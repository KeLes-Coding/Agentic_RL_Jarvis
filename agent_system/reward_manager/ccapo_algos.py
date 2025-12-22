# agent_system/reward_manager/ccapo_algos.py

import torch
import numpy as np
import collections
from typing import List, Dict, Any
from sentence_transformers import util
import logging
import os
import json
import datetime

# --- 1. 标准日志器 ---
logger = logging.getLogger(__name__)

# --- 2. 专用文件日志器 ---
ccapo_file_logger = logging.getLogger("CCAPO_FILE")
ccapo_file_logger.setLevel(logging.INFO)
ccapo_file_logger.propagate = False

if not ccapo_file_logger.handlers:
    try:
        log_dir = "logger/CCAPO"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "ccapo_operations.log")
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - [CCAPO_FILE] - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        ccapo_file_logger.addHandler(file_handler)
        ccapo_file_logger.info("--- CCAPO 专用文件日志器已初始化 ---")
    except Exception as e:
        logger.error(f"[CCAPO] 无法创建专用文件日志器: {e}")

# --- [新增] 3. 匹配详情调试日志器 (MATCH_DEBUG) ---
match_debug_logger = logging.getLogger("MATCH_DEBUG")
match_debug_logger.setLevel(logging.INFO)
match_debug_logger.propagate = False

if not match_debug_logger.handlers:
    try:
        log_dir = "logger/CCAPO"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "ccapo_match_debug.log")
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        # 纯净格式，方便阅读
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        match_debug_logger.addHandler(file_handler)
        match_debug_logger.info(f"=== Match Debug Logger Init: {datetime.datetime.now()} ===")
    except Exception as e:
        logger.error(f"[CCAPO] 无法创建匹配调试日志器: {e}")

# =============================================================================
# 基础辅助函数
# =============================================================================

def _standardize(values: List[float]) -> List[float]:
    """Z-score standardization"""
    if not values: return []
    mean = np.mean(values)
    std = np.std(values) + 1e-6
    if std < 1e-6: return [0.0] * len(values)
    return [(v - mean) / std for v in values]

def _group_steps_by_traj(steps_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按 traj_uid 分组并排序"""
    trajs = collections.defaultdict(list)
    for step in steps_list:
        traj_uid_val = step['traj_uid']
        if isinstance(traj_uid_val, (np.ndarray, torch.Tensor)):
            traj_uid_val = traj_uid_val.item()
        trajs[traj_uid_val].append(step)
    for traj_uid in trajs:
        trajs[traj_uid].sort(key=lambda s: s.get('step_index', 0))
    return trajs

def _get_deterministic_id(step: Dict[str, Any], use_fine_grained: bool) -> str:
    """生成确定性的动作标识符"""
    action_type = step.get('action_type', '')
    
    if not use_fine_grained:
        return action_type
    
    raw_act = step.get('parsed_action')
    if raw_act is None:
        return f"{action_type}::None"

    if isinstance(raw_act, dict):
        try:
            sorted_items = sorted([(k, str(v)) for k, v in raw_act.items()])
            content = "|".join([f"{k}={v}" for k, v in sorted_items])
            return f"{action_type}::{content}"
        except Exception:
            pass

    return f"{action_type}::{str(raw_act)}"

# =============================================================================
# [轨迹重建组件] Trajectory Reconstructor
# =============================================================================

def _reconstruct_trajectory_from_disk(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    [Reconstructor V3]
    基于磁盘文件重建完整轨迹，解决内存数据截断/丢失问题。
    """
    if not steps: return steps
    
    log_dir = steps[0].get('log_dir_path')
    if not log_dir or not os.path.exists(log_dir):
        return steps

    input_step_map = {s.get('step_index'): s for s in steps}
    reconstructed_steps = []
    
    # 动态探测最大步数
    max_probe_limit = 100 
    start_idx = 0
    
    for i in range(start_idx, max_probe_limit):
        step_file = os.path.join(log_dir, f"step_{i}", "step_details.json")
        
        if not os.path.exists(step_file):
            if i == 0: continue # 允许 step 0 缺失
            break # 连续缺失则停止
            
        if i in input_step_map:
            # 内存里有，优先使用
            reconstructed_steps.append(input_step_map[i])
        else:
            # 内存没有，从磁盘加载
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                healed_step = {
                    'step_index': i,
                    'action_type': data.get('action_type', 'unknown'),
                    'parsed_action': data.get('parsed_action', ''),
                    'action_success': data.get('action_success', False),
                    'action_status': data.get('action_status', ''),
                    'is_human_marked': False,
                    'is_healed_data': True,
                    'log_dir_path': log_dir,
                    'R_core': steps[0].get('R_core'), 
                    'R_tau': steps[0].get('R_tau'),
                    'traj_uid': steps[0].get('traj_uid'),
                    'is_buffer_data': steps[0].get('is_buffer_data', False)
                }
                
                match_debug_logger.info(f"  [RECONSTRUCT] Recovered Step {i} from disk.")
                reconstructed_steps.append(healed_step)
                
            except Exception as e:
                match_debug_logger.warning(f"  [RECONSTRUCT] Failed to load Step {i}: {e}")

    if not reconstructed_steps:
        return steps
        
    return reconstructed_steps

# =============================================================================
# 核心逻辑组件
# =============================================================================

def _calculate_R_tau(g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    """[Sec 4] 计算 R_tau, R_core"""
    for traj_uid, steps in g_calc_trajs.items():
        if not steps: continue
        first_step = steps[0]
        
        if first_step.get('is_buffer_data', False):
            R_success = 1.0
        else:
            R_success = 1.0 if first_step.get('traj_task_completed', False) else -1.0
            
        total_steps = first_step.get('traj_total_steps', 0)
        total_tokens = first_step.get('traj_total_tokens', 0)
        
        is_shortcut = (R_success > 0 and total_steps < config.min_reasonable_steps)
        
        R_core = 0.0
        if R_success <= 0:
            R_core = -1.0
        elif not is_shortcut:
            R_core = 1.0
        
        R_tau = R_core 
        m_token_ratio = total_tokens / (config.max_tokens + 5)
        token_cost = m_token_ratio 
        
        if R_core == 1.0:
            m_steps_ratio = total_steps / config.max_steps
            M_steps = (max(0.0, 1.0 - m_steps_ratio))**0.5
            M_token = (max(0.0, 1.0 - m_token_ratio))**0.5
            R_tau = R_core * M_steps * M_token
        
        for step in steps:
            step['R_tau'] = R_tau
            step['R_core'] = R_core
            step['TokenCost'] = token_cost

def _reconstruct_trajectory_from_disk(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    [Reconstructor V3.1] 带日志增强
    """
    if not steps: return steps
    
    # 获取 UID 用于日志
    uid = steps[0].get('traj_uid', 'UNKNOWN')
    log_dir = steps[0].get('log_dir_path')
    
    if not log_dir or not os.path.exists(log_dir):
        match_debug_logger.warning(f"  [RECONSTRUCT] Skip {uid}: path invalid {log_dir}")
        return steps

    input_step_map = {s.get('step_index'): s for s in steps}
    reconstructed_steps = []
    
    max_probe_limit = 100 
    start_idx = 0
    
    for i in range(start_idx, max_probe_limit):
        step_file = os.path.join(log_dir, f"step_{i}", "step_details.json")
        
        if not os.path.exists(step_file):
            if i == 0: continue 
            break 
            
        if i in input_step_map:
            reconstructed_steps.append(input_step_map[i])
        else:
            # 内存没有，从磁盘加载
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # [日志] 确认磁盘数据身份
                disk_path = data.get('log_dir_path', 'N/A')
                if disk_path != 'N/A' and os.path.basename(disk_path) != os.path.basename(log_dir):
                     match_debug_logger.error(f"  [RECONSTRUCT] !!! IDENTITY MISMATCH !!!")
                     match_debug_logger.error(f"     Memory Expects: {log_dir}")
                     match_debug_logger.error(f"     Disk Contains:  {disk_path}")

                healed_step = {
                    'step_index': i,
                    'action_type': data.get('action_type', 'unknown'),
                    'parsed_action': data.get('parsed_action', ''),
                    'action_success': data.get('action_success', False),
                    'action_status': data.get('action_status', ''),
                    'is_human_marked': False,
                    'is_healed_data': True,
                    'log_dir_path': log_dir,
                    'R_core': steps[0].get('R_core'), 
                    'R_tau': steps[0].get('R_tau'),
                    'traj_uid': steps[0].get('traj_uid'),
                    'is_buffer_data': steps[0].get('is_buffer_data', False)
                }
                reconstructed_steps.append(healed_step)
                
            except Exception as e:
                match_debug_logger.warning(f"  [RECONSTRUCT] Failed to load Step {i}: {e}")

    if not reconstructed_steps:
        return steps
        
    return reconstructed_steps

def _calculate_A_traj(g_calc_steps: List[Dict[str, Any]], current_sr: float):
    """
    [Sec 4.2 Fixed] 修复版 A_traj 计算
    1. 修正统计方式：按 Trajectory 聚合，消除 Step Count Bias。
    2. 修正激励逻辑：成功样本底线提升至 0.0 (或更高)，严禁给予负分(-0.2)导致遗忘。
    """
    # 打印所有 uid 和对应的 R_tau，检查是否有重复 UID 对应不同 R_tau 的情况
    debug_uid_check = collections.defaultdict(set)
    for step in g_calc_steps:
        uid = step['traj_uid']
        if isinstance(uid, torch.Tensor): uid = uid.item()
        debug_uid_check[uid].add(step.get('R_tau', 0.0))

    for uid, r_set in debug_uid_check.items():
        if len(r_set) > 1:
            match_debug_logger.error(f"!!! CRITICAL WARNING: UID Collision Detected !!! UID {uid} maps to multiple R_tau values: {r_set}")

    if not g_calc_steps: return

    # --- 1. 数据重组：按轨迹聚合 (Key Fix 1) ---
    # 必须把 step 列表压缩成轨迹列表，否则长轨迹会主导均值
    traj_map = {} # uid -> R_tau
    for step in g_calc_steps:
        # 兼容 Tensor 或普通类型
        uid = step['traj_uid']
        if isinstance(uid, (np.ndarray, torch.Tensor)):
            uid = uid.item()
        
        # 只记录一次 R_tau (假设同轨迹内 R_tau 相同)
        if uid not in traj_map:
            traj_map[uid] = step.get('R_tau', 0.0)
    
    unique_r_taus = list(traj_map.values())
    if not unique_r_taus: return

    # 基于轨迹的统计量
    mean_R = np.mean(unique_r_taus)
    std_R = np.std(unique_r_taus)
    r_ptp = np.ptp(unique_r_taus)

    # --- 2. 动态底线 (Key Fix 2) ---
    # 只要是成功样本，绝不给负 Advantage。
    # 早期 (SR低) 给强激励 (+0.2)，后期给中性激励 (0.0)。
    # 只有当你明确想要模型“放弃”某种成功路径时，才给负分，但这非常危险。
    if current_sr < 0.25:
        success_floor = 0.2
    elif current_sr < 0.5:
        success_floor = 0.0
    else:
        success_floor = -0.2

    # 计算 Advantage
    traj_adv_map = {}
    
    # 异常处理：全同分
    if r_ptp < 1e-6:
        fill_val = 0.2 if mean_R > 0 else -0.5
        for uid in traj_map:
            traj_adv_map[uid] = fill_val
    else:
        min_std = 0.02
        safe_std = max(std_R, min_std)
        
        for uid, r_val in traj_map.items():
            # 标准 Z-Score
            adv = (r_val - mean_R) / safe_std
            
            # --- [逻辑修正核心] ---
            if r_val > 0.0:
                # 成功样本：如果算出来是负数（比如 -1.0），强行拉回 Floor (0.0 或 0.2)
                # 这样 Traj 20 即使比 Traj 17 差，也依然是“正向”的。
                if adv < success_floor:
                    adv = success_floor
            elif r_val <= 0.0:
                # 失败样本：封顶 -0.05
                if adv > -0.05:
                    adv = -0.05
            
            traj_adv_map[uid] = adv

    # --- 3. 回写 ---
    final_advantages = []
    for step in g_calc_steps:
        uid = step['traj_uid']
        if isinstance(uid, (np.ndarray, torch.Tensor)):
            uid = uid.item()
        
        adv = traj_adv_map.get(uid, 0.0)
        # Clip
        adv = float(np.clip(adv, -3.0, 3.0))
        step['A_traj'] = adv


# =============================================================================
# [新增/修改] 辅助函数：全局计数持久化
# =============================================================================

def _load_global_counts(save_path: str) -> Dict[str, int]:
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[CCAPO] Failed to load global counts: {e}")
    return {}

def _save_global_counts(save_path: str, counts: Dict[str, int]):
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(counts, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[CCAPO] Failed to save global counts: {e}")

# =============================================================================
# [修改] 格式奖励/惩罚组件 (优化：增加正向引导)
# =============================================================================

def _calculate_R_format_penalty(g_calc_steps: List[Dict[str, Any]], config):
    """
    [优化版] 计算格式惩罚与奖励
    Change: 不仅惩罚错误，还奖励正确 (Format Compliance Bonus)，加速前期指令遵循的学习。
    """
    # 获取配置中的奖励值，如果未配置则默认为 0.0 (建议设为 0.1~0.2)
    compliance_bonus = getattr(config, 'reward_format_compliance', 0.1)
    
    for step in g_calc_steps:
        action_status = step.get('action_status', '')
        is_success = step.get('action_success', False)
        
        # 判断是否为格式/执行合规
        # 只要没有明显的 FORMAT_ERROR 或 UNKNOWN_ACTION，即使执行失败(FAILURE)也视为“格式正确”
        # 这里逻辑需根据你的 action_status 定义微调
        is_format_compliant = (
            is_success is True or 
            action_status == 'true' or 
            action_status == '' or 
            action_status == 'SUCCESS' or
            action_status.startswith('FAILURE') or # 动作执行失败通常意味着格式是对的，只是环境没跑通
            action_status.startswith('EXECUTION_ERROR')
        )
        
        penalty = 0.0
        bonus = 0.0
        
        if is_format_compliant:
            # 格式正确，给予微小奖励，抵消 R_core 的部分负值
            bonus = compliance_bonus
            # 如果是明确的成功，不扣分；如果是执行失败，可能扣分也可能不扣，取决于配置
            # 这里保持原逻辑：执行层面的失败是否扣分？
            if action_status.startswith('FAILURE') or action_status.startswith('EXECUTION_ERROR'):
                penalty = config.penalty_failure
            else:
                penalty = 0.0
        else:
            # 明确的格式错误
            if (action_status.startswith('FORMAT_ERROR') or 
                action_status.startswith('UNKNOWN_ACTION') or 
                action_status.startswith('ARGUMENT_ERROR')):
                penalty = config.penalty_format_error
            else:
                penalty = config.penalty_failure # Fallback
        
        # 最终写入 R_format_penalty 字段 (正数表示净奖励，负数表示净惩罚)
        step['R_format_penalty'] = bonus + penalty

# =============================================================================
# [修改] 动作新颖性组件
# =============================================================================
def _calculate_action_novelty(g_calc_steps: List[Dict[str, Any]], config):
    """
    [CCAPO V10 - Simplified] 局部 Batch 级新颖性奖励
    不再读写磁盘，仅计算当前 Batch 内的动作频率。
    公式: Bonus = base / sqrt(Batch_Count(a))
    """
    use_fine_grained = getattr(config, 'use_fine_grained_action', True)
    base_bonus = getattr(config, 'base_bonus', 0.2) 
    
    # 1. 统计当前 Batch 内所有动作的出现次数
    batch_counts = collections.defaultdict(int)
    
    # 第一遍遍历：统计
    for step in g_calc_steps:
        # 只统计成功的动作，失败动作不配拥有新颖性
        if not step.get('action_success', False): 
            continue
            
        act_id = _get_deterministic_id(step, use_fine_grained)
        batch_counts[act_id] += 1
        
    # 2. 第二遍遍历：分配奖励
    for step in g_calc_steps:
        if not step.get('action_success', False):
            step['Z_novelty'] = 0.0
            continue
            
        act_id = _get_deterministic_id(step, use_fine_grained)
        count = batch_counts.get(act_id, 1)
        
        # 局部衰减公式：出现次数越多，奖励越低
        # sqrt 使得衰减比较温和，不会因为出现 2 次就变成 0
        novelty_score = base_bonus / np.sqrt(count)
        
        step['Z_novelty'] = novelty_score

def _calculate_repetition_penalty(g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    """
    [新增] 轨迹内重复动作惩罚 (指数级 + 容忍度)
    逻辑:
      - 允许连续重复 N 次 (tolerance)
      - 超过后，惩罚 = base * (factor ^ (excess_count))
      - 剥夺当步 Format Bonus
    """
    use_fine_grained = getattr(config, 'use_fine_grained_action', True)
    
    # 容忍度: 允许连续出现 2 次 (即第3次才罚)
    tolerance = getattr(config, 'repetition_tolerance', 2) 
    # 基础惩罚值
    base_penalty = getattr(config, 'penalty_repetition_base', 1.0)
    # 指数增长因子
    exp_factor = getattr(config, 'penalty_repetition_factor', 2.0)
    
    for uid, steps in g_calc_trajs.items():
        if not steps: continue
        
        last_act_id = None
        consecutive_count = 1
        
        for step in steps:
            curr_act_id = _get_deterministic_id(step, use_fine_grained)
            
            # 检测重复
            if last_act_id is not None and curr_act_id == last_act_id:
                consecutive_count += 1
            else:
                consecutive_count = 1 # 重置计数
            
            # 判断是否超过容忍度
            if consecutive_count > tolerance:
                # 计算超出的次数 (从 0 开始)
                excess = consecutive_count - tolerance - 1 
                # 指数惩罚: 1.0, 2.0, 4.0, 8.0...
                current_penalty = base_penalty * (exp_factor ** excess)
                
                # 写入负分
                step['R_repetition'] = -current_penalty
                
                # 【关键】剥夺 Format Bonus
                # 既然触发了恶意复读，就不配拿格式分
                if step.get('R_format_penalty', 0.0) > 0:
                      step['R_format_penalty'] = 0.0
                      
                step['action_status'] = f"REP({consecutive_count})::{step.get('action_status','')}"
            else:
                step['R_repetition'] = 0.0
            
            last_act_id = curr_act_id

def _calculate_R_step_success(g_calc_steps: List[Dict[str, Any]], g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    """
    [CCAPO Simplified] 极简版 LCS + Q_step
    增加详细日志：为什么选它做 Anchor？
    """
    match_debug_logger.info(f"\n=== [Anchor Selection & LCS Start] ===")

    # --- 1. 数据清洗与筛选 ---
    success_traj_info = {}
    
    # 计数器：发现了多少成功轨迹
    found_success_count = 0
    
    for traj_uid, raw_steps in g_calc_trajs.items():
        if not raw_steps: continue
        
        # 检查是否成功
        r_core = raw_steps[0].get('R_core')
        if r_core != 1.0: 
            continue
            
        found_success_count += 1

        # 轨迹重建
        steps = _reconstruct_trajectory_from_disk(raw_steps)
        
        # 提取指纹
        act_identifiers = []
        valid_steps = []
        n_success_count = 0 

        for step in steps:
            is_valid_success = step.get('action_success', False)
            if step.get('is_buffer_data'):
                is_valid_success = True

            if not is_valid_success:
                continue
                
            n_success_count += 1
            s_type = step.get('action_type', '')
            s_parsed = step.get('parsed_action')
            act_str = str(s_parsed).strip() if s_parsed else ""
            act_id = f"{s_type}::{act_str}"
            
            act_identifiers.append(act_id)
            valid_steps.append(step)

        if valid_steps:
            r_tau_val = steps[0].get('R_tau', 0.0)
            success_traj_info[traj_uid] = {
                'steps': valid_steps, 
                'all_steps_raw': steps, 
                'act_ids': act_identifiers, 
                'n_steps': len(valid_steps),
                'n_success_count': n_success_count,
                'n_tokens': steps[0].get('traj_total_tokens', 0),
                'r_tau': r_tau_val
            }

    match_debug_logger.info(f"  -> Total Success Trajs Found in Batch: {found_success_count}")
    
    if not success_traj_info: 
        match_debug_logger.info(f"  -> No valid success trajectories for Anchor (Valid Count=0). Skip.")
        return

    # --- 2. 锚点选择 (Anchor Selection) ---
    # 打印候选名单
    match_debug_logger.info(f"  [Election] Candidates:")
    
    candidate_list = []
    for uid, info in success_traj_info.items():
        candidate_list.append({
            'uid': uid,
            'R_tau': info['r_tau'],
            'Steps': info['n_steps']
        })
    
    # 按照排序逻辑打印
    # 排序逻辑: R_tau 降序, Steps 升序
    sorted_candidates = sorted(
        candidate_list,
        key=lambda x: (-x['R_tau'], x['Steps'])
    )
    
    for rank, cand in enumerate(sorted_candidates):
        match_debug_logger.info(f"    Rank {rank+1}: UID={cand['uid']}, R_tau={cand['R_tau']:.4f}, Steps={cand['Steps']}")

    # 实际排序
    sorted_trajs = sorted(
        success_traj_info.items(), 
        key=lambda item: (-item[1]['r_tau'], item[1]['n_steps']) 
    )
    
    anchor_uid, anchor_data = sorted_trajs[0]
    anchor_seq = anchor_data['act_ids']
    anchor_len = max(1.0, float(anchor_data['n_success_count'])) 
    
    match_debug_logger.info(f"  [Decision] Selected Anchor: UID {anchor_uid} (Len={anchor_len:.1f})")

    # 标记 Anchor
    for s in anchor_data['all_steps_raw']:
        s['is_anchor'] = True
        s['anchor_uid'] = str(anchor_uid)

    # --- 3. LCS 计算与奖励分配 ---
    def get_lcs_match_indices(seq_a, seq_b):
        m, n = len(seq_a), len(seq_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i-1] == seq_b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        matched_b_indices = set()
        i, j = m, n
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                matched_b_indices.add(j-1)
                i -= 1; j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        return matched_b_indices

    for traj_uid, info in success_traj_info.items():
        current_seq = info['act_ids']
        raw_steps_to_update = info['all_steps_raw']
        
        matched_indices_curr = get_lcs_match_indices(anchor_seq, current_seq)
        
        n_success = info['n_success_count']
        m_steps_ratio = n_success / config.max_steps
        q_step = max(0.0, 1.0 - config.alpha_step * m_steps_ratio)
        
        current_len = max(1.0, float(n_success))
        len_ratio = anchor_len / current_len
        len_ratio = min(1.0, len_ratio) 
        mismatch_coeff = 0.95 * len_ratio

        valid_idx_counter = 0
        match_count = 0
        
        for step in raw_steps_to_update:
            step['R_core_raw'] = 0.0
            step['meta_anchor_uid'] = str(anchor_uid)
            
            should_eval = False
            if step.get('action_success', False) or step.get('is_buffer_data'):
                should_eval = True
            
            if should_eval:
                is_match = (valid_idx_counter in matched_indices_curr)
                step['Q_step'] = q_step 
                
                if is_match:
                    match_count += 1
                    r_core_raw = 1.0 * q_step
                    step['R_core_raw'] = r_core_raw
                    step['S_utility'] = 1.0 
                    step['I_action'] = 1.0 
                    step['meta_lcs_id'] = f"MATCH_IDX_{valid_idx_counter}"
                else:
                    dynamic_reward = mismatch_coeff * q_step
                    r_core_raw = max(0.5 * q_step, dynamic_reward)
                    step['R_core_raw'] = r_core_raw
                    step['S_utility'] = mismatch_coeff 
                    step['I_action'] = 0.0 
                    step['meta_lcs_id'] = f"ALT_PATH_{mismatch_coeff:.2f}"

                valid_idx_counter += 1
            else:
                step['R_core_raw'] = 0.0
                step['Q_step'] = q_step
                step['I_action'] = 0.0
                step['meta_lcs_id'] = "N/A"
        
        if str(traj_uid) != str(anchor_uid):
             match_debug_logger.info(f"    Compared Traj {traj_uid} to Anchor. Matched Steps: {match_count}/{len(current_seq)}")

    match_debug_logger.info(f"=== [Anchor Selection End] ===\n")

def _calculate_R_step_fail(g_calc_steps: List[Dict[str, Any]], g_buffer_steps: List[Dict[str, Any]], embedding_model, config):
    """
    [Sec 5.2] 失败挽救
    """
    stdb_step_scores = collections.defaultdict(list)
    stdb_thoughts = []
    stdb_map = {} 
    idx = 0
    
    use_fine_grained = getattr(config, 'use_fine_grained_action', True)

    for step in g_buffer_steps:
        if step.get('R_core') != 1.0: continue
        if not step.get('action_success', False): continue
        
        action_key = _get_deterministic_id(step, use_fine_grained)
        
        stdb_step_scores[action_key].append({'score': step.get('R_step', 0.0)})
        stdb_thoughts.append(step.get('thought', ''))
        stdb_map[idx] = (action_key, len(stdb_step_scores[action_key]) - 1)
        idx += 1

    if not stdb_thoughts: return
    stdb_embeddings = embedding_model.encode(stdb_thoughts, convert_to_tensor=True)
    for i, emb in enumerate(stdb_embeddings):
        action_key, list_idx = stdb_map[i]
        stdb_step_scores[action_key][list_idx]['embedding'] = emb

    fail_steps, fail_indices = [], []
    for i, step in enumerate(g_calc_steps):
        if step.get('R_core') != -1.0: continue
        if not step.get('action_success', False): continue
        
        action_key = _get_deterministic_id(step, use_fine_grained)
        if action_key not in stdb_step_scores: continue
        
        fail_steps.append(step.get('thought', ''))
        fail_indices.append(i)
        
    if not fail_steps: return
    fail_embeddings = embedding_model.encode(fail_steps, convert_to_tensor=True)
    
    for i, emb_t in enumerate(fail_embeddings):
        step_idx = fail_indices[i]
        step = g_calc_steps[step_idx]
        
        action_key = _get_deterministic_id(step, use_fine_grained)
        
        matches = stdb_step_scores[action_key]
        valid_m = [m for m in matches if 'embedding' in m]
        if not valid_m: continue
            
        comp_embs = torch.stack([m['embedding'] for m in valid_m]).to(emb_t.device)
        comp_scores = torch.tensor([m['score'] for m in valid_m], device=emb_t.device)
        
        cos_sims = util.cos_sim(emb_t, comp_embs)[0]
        found = torch.where(cos_sims > config.similarity_threshold)[0]
        
        if len(found) > 0:
            step['R_match_raw'] = torch.max(comp_scores[found]).item()

def _calculate_A_step(g_calc_steps: List[Dict[str, Any]]):
    """计算 A_step"""
    if not g_calc_steps: return
    all_R_step = [step.get('R_step', 0.0) for step in g_calc_steps]
    mean_s = np.mean(all_R_step)
    std_s = np.std(all_R_step) + 1e-6
    for step in g_calc_steps:
        step['A_step'] = (step.get('R_step', 0.0) - mean_s) / std_s

def _calculate_separated_advantages(steps: List[Dict[str, Any]], omega: float, current_sr: float):
    """
    [关键] 分离计算 Advantage 并进行最终归一化
    """
    if not steps: return
    
    # 1. 计算轨迹级优势 (可能为全0)
    _calculate_A_traj(steps, current_sr)
    
    # 2. 计算步骤级优势
    _calculate_A_step(steps)
    
    # 3. 加权融合
    raw_advs = []
    for s in steps:
        # 如果 A_traj 为 0 (全错场景)，则 Advantage 完全由 A_step 驱动
        raw = s['A_traj'] + omega * s['A_step']
        s['A_final_raw'] = raw
        raw_advs.append(raw)
    
    # 4. 最终 Batch 级归一化 (Whitening)
    # 这步至关重要：即使 raw_advs 很小，归一化后也能提供有效的梯度信号
    mean_adv = np.mean(raw_advs)
    std_adv = np.std(raw_advs)
    
    # 再次使用较小的 epsilon，防止数值不稳定
    safe_std_final = std_adv + 1e-8
    
    for s in steps:
        s['advantages'] = (s['A_final_raw'] - mean_adv) / safe_std_final

# =============================================================================
# [新增] 5. Dr. GRPO 长度加权组件 (解决数量偏差)
# =============================================================================

def _apply_dr_grpo_v3(steps_list: List[Dict[str, Any]]):
    """
    [Dr. GRPO V3.2 - Prompt-Specific Grouping]
    根据 'prompt_index' 对 Prompt 进行严格分组。
    同一 Prompt 下的所有轨迹（无论成功失败）参与该组的长度统计。
    """
    if not steps_list: return

    # --- 1. 按 Prompt (Task) 分组 ---
    # 结构: prompt_idx -> [ {uid: str, steps: list, is_success: bool, length: int} ]
    prompt_groups = collections.defaultdict(list)
    
    # 先按轨迹聚合，方便提取每条轨迹的属性
    trajs = _group_steps_by_traj(steps_list)
    
    for uid, steps in trajs.items():
        if not steps: continue
        
        # [关键] 直接获取 upstream 传递下来的 prompt_index
        # ray_trainer.py 中已将其从 non_tensor_batch pop 出来
        p_idx = steps[0].get('prompt_index')
        
        # 容错处理：如果 tensor 类型则取值，如果不存在则归为 'unknown'
        if hasattr(p_idx, 'item'):
            p_idx = int(p_idx.item())
        elif p_idx is None:
            # 这是一个极端情况，理论上不应发生，除非数据流中有丢失
            p_idx = 'unknown'
        
        r_core = steps[0].get('R_core', 0.0)
        is_success = (r_core == 1.0)
        length = len(steps)
        
        prompt_groups[p_idx].append({
            'uid': uid,
            'is_success': is_success,
            'length': length
        })

    # --- 2. 对每个 Group 计算统计量并分配权重 ---
    uid_weight_map = {} # uid -> weight
    
    for p_idx, group_items in prompt_groups.items():
        # 2.1 提取该组内 *成功* 样本的长度
        succ_lengths = [item['length'] for item in group_items if item['is_success']]
        
        # 2.2 计算统计量 (Mean, Std)
        # 只有当该 Prompt 至少有 2 条成功轨迹时，比较才有意义
        # 如果只有 1 条成功，或者全是失败，无法计算相对优势
        can_calculate_stats = len(succ_lengths) > 1
        
        if can_calculate_stats:
            arr = np.array(succ_lengths, dtype=float)
            mean_succ = np.mean(arr)
            std_succ = np.std(arr) + 1e-6
        else:
            # 无法比较时，不进行缩放
            mean_succ = 0.0
            std_succ = 1.0
            
        # 2.3 计算每个轨迹的权重
        for item in group_items:
            uid = item['uid']
            is_succ = item['is_success']
            length = item['length']
            
            weight = 1.0 # 默认为 1.0 (不改变)
            
            if is_succ and can_calculate_stats:
                # Dr. GRPO 核心公式
                # 只有成功样本参与“内卷”
                z_score = (mean_succ - float(length)) / std_succ
                
                # Tanh Gating: 映射到 (0, 2) 区间，中心为 1.0
                # 短于平均 -> z > 0 -> tanh > 0 -> weight > 1.0 (奖励放大)
                # 长于平均 -> z < 0 -> tanh < 0 -> weight < 1.0 (奖励抑制)
                weight = 1.0 + np.tanh(z_score)
                
                # 安全下限：防止权重过小导致样本失效（虽然是成功的）
                weight = max(0.2, weight)
            
            uid_weight_map[uid] = weight
            
            # [可选] 日志记录：看看哪些 Prompt 触发了加权
            if is_succ and can_calculate_stats and abs(weight - 1.0) > 0.1:
                match_debug_logger.info(f"  [Dr.GRPO] Prompt {p_idx} | Len {length} (Avg {mean_succ:.1f}) -> W {weight:.2f}")

    # --- 3. 将权重写回步骤列表 ---
    for step in steps_list:
        uid_val = step['traj_uid']
        if isinstance(uid_val, (np.ndarray, torch.Tensor)):
             uid_val = uid_val.item()
        
        weight = uid_weight_map.get(uid_val, 1.0)
        
        step['W_length'] = weight
        if 'advantages' in step:
            step['advantages'] *= weight
            # PPO 最终防爆裁剪
            step['advantages'] = np.clip(step['advantages'], -4.0, 4.0).item()

# =============================================================================
# 主入口
# =============================================================================
def compute_ccapo_advantages(g_calc_steps: List[Dict[str, Any]], 
                             g_online_steps: List[Dict[str, Any]], 
                             g_buffer_steps: List[Dict[str, Any]], 
                             embedding_model, 
                             config):
    """
    [CCAPO V9.4] Integration with Dr. GRPO (Trajectory-Length Normalization)
    Updates:
    1. Integrated `_apply_length_weighting` to fix Quantity Bias.
    2. Maintained `_calculate_repetition_penalty` and all legacy logic.
    """
    ccapo_file_logger.info("=== [CCAPO V9.4] Start Calculation ===")
    
    # 1. 初始化键值 (新增 R_repetition, meta_anchor_uid, meta_lcs_id)
    keys_defaults = {
        'R_core_raw': 0.0, 'R_match_raw': 0.0, 'R_format_penalty': 0.0, 
        'S_necessity': 0.0, 'S_utility': 0.0, 'I_action': 0.0, 
        'Q_step': 0.0, 'Z_novelty': 0.0, 'Z_core': 0.0, 'Z_match': 0.0, 
        'TokenCost': 0.0, 'b_stage': 'N/A', 'R_novelty_bonus': 0.0,
        'anchor_uid': 'N/A', 'is_anchor': False,
        'R_step': 0.0,
        # [新增] 重复惩罚字段
        'R_repetition': 0.0,
        # [新增] Dr. GRPO 权重字段
        'W_length': 1.0,
        # [修复] 必须初始化这些用于 Tooltip 可视化的元数据，防止 DataProto 长度不一致报错
        'meta_anchor_uid': 'N/A',
        'meta_lcs_id': 'N/A'
    }

    for step in g_calc_steps:
        for k, v in keys_defaults.items():
            step[k] = v

    # 初始化轨迹统计 (保持不变)
    for step in g_calc_steps:
        if step.get('is_buffer_data', False):
            if 'traj_task_completed' not in step: step['traj_task_completed'] = True
            if 'traj_total_steps' not in step: step['traj_total_steps'] = step.get('step_index', 0)
            if 'traj_total_tokens' not in step: step['traj_total_tokens'] = 0

    g_calc_trajs = _group_steps_by_traj(g_calc_steps)
    _calculate_R_tau(g_calc_trajs, config)
    
    # 2. 计算 SR 和 动态权重 w_N
    online_trajs = _group_steps_by_traj(g_online_steps)
    online_succ = sum(1 for steps in online_trajs.values() if steps and steps[0].get('R_core') == 1.0)
    sr = online_succ / (len(online_trajs) + 1e-6)
    
    max_w = config.get("max_w_N", 0.8)
    min_w = config.get("min_w_N", 0.2)
    # w_N = min_w + (max_w - min_w) * (1.0 - sr)
    w_N = 0.2
    
    ccapo_file_logger.info(f"SR: {sr:.4f}, w_N: {w_N:.4f}")

    # 3. [关键步骤] 奖励计算流水线
    
    # Step A: 计算格式分 (含 Bonus)
    _calculate_R_format_penalty(g_calc_steps, config)

    # Step B: 计算动作新颖性
    _calculate_action_novelty(g_calc_steps, config)

    # === [新增] Step B.2: 轨迹内重复惩罚 ===
    # 必须在 Step A 之后执行，因为要剥夺恶意复读的格式分
    _calculate_repetition_penalty(g_calc_trajs, config)

    # Step C: 计算基于 LCS 的共识奖励
    # 注意：这里只会更新“成功”轨迹的 meta_anchor_uid，失败轨迹保持默认值 'N/A'
    _calculate_R_step_success(g_calc_steps, g_calc_trajs, config)
    
    # 4. 最终奖励聚合 (Aggregating Rewards)
    
    # 处理成功轨迹
    success_steps = [s for s in g_calc_steps if s.get('R_core') == 1.0]
    for s in success_steps:
        z_val = s.get('R_core_raw', 0.0) 
        s['Z_core'] = 0.0 
        
        # [公式更新] 加上 R_repetition (通常为 0 或 负数)
        s['R_step'] = z_val + w_N * s.get('Z_novelty', 0.0) + \
                      s['R_format_penalty'] + s.get('R_repetition', 0.0)

    # 处理失败轨迹 (尝试 STDB 挽救)
    _calculate_R_step_fail(g_calc_steps, g_buffer_steps, embedding_model, config)
    
    fail_steps = [s for s in g_calc_steps if s.get('R_core') == -1.0]
    for s in fail_steps:
        z_match = s.get('R_match_raw', 0.0)
        s['Z_match'] = 0.0 
        
        # 获取该步的重复惩罚
        rep_pen = s.get('R_repetition', 0.0)
        
        # [公式更新] 包含重复惩罚
        if z_match > config.similarity_threshold:
            s['R_step'] = z_match + w_N * s.get('Z_novelty', 0.0) + \
                          s['R_format_penalty'] + rep_pen
        else:
            # 即使没 Match 上，也要把重复惩罚算进去
            s['R_step'] = s.get('R_format_penalty', 0.0) + rep_pen

    # 处理其他情况
    for step in g_calc_steps:
        if step.get('R_core') not in [1.0, -1.0]:
            step['R_step'] = step.get('R_format_penalty', 0.0)

    # 5. 分离 Advantage 计算
    online_subset = [s for s in g_calc_steps if not s.get('is_buffer_data', False)]
    buffer_subset = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    
    _calculate_separated_advantages(online_subset, config.omega, sr)
    _calculate_separated_advantages(buffer_subset, config.omega, sr)
    
    # === [新增] Step 6: Dr. GRPO 长度加权 (Quantity Bias Fix) ===
    # 仅对 Online 数据进行长度归一化加权，确保长轨迹不会因为 Token 多而主导梯度
    _apply_dr_grpo_v3(online_subset)
    
    ccapo_file_logger.info("=== [CCAPO V9.4] Done ===")
    return g_calc_steps, sr