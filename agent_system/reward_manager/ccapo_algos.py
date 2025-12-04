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
        m_token_ratio = total_tokens / config.max_tokens
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

def _calculate_A_traj(g_calc_steps: List[Dict[str, Any]], current_sr: float):
    """
    [Sec 4.2 Optimized V2] 计算 A_traj
    修改点：
    1. 全错一致 (All Fail Same) -> Adv = -0.5 (软惩罚，防止因 R_step 正向导致刷分)
    2. 全对一致 (All Success Same) -> Adv = 0.0 (完美收敛)
    3. 正常差异 -> 微小 epsilon
    """
    if not g_calc_steps: return
    all_R_tau = [step.get('R_tau', 0.0) for step in g_calc_steps]
    
    mean_R = np.mean(all_R_tau)
    std_R = np.std(all_R_tau)
    r_ptp = np.ptp(all_R_tau)
    
    # --- 核心数学修正 ---
    
    # Case 1: 全错且一致 (All Fail Same)
    # [用户修正] 不能设为 0.0，否则模型会为了追求 R_step 而故意失败。
    # 设为 -0.5，给予基础负反馈，保证 Success (>0) 永远优于 Failure (-0.5)。
    if r_ptp < 1e-6 and mean_R <= 0:
        raw_advantages = [-0.5 for _ in all_R_tau]
        match_debug_logger.info(f"  [A_traj] All Fail Same (Mean={mean_R:.2f}). Adv=-0.5 (Soft Penalty)")

    # Case 2: 全对且一致 (All Success Same)
    # 既然都满分且效率一样，保持静默，防止数值噪音。
    elif r_ptp < 1e-6 and mean_R > 0:
        raw_advantages = [0.1 for _ in all_R_tau]
        match_debug_logger.info(f"  [A_traj] All Success Same. Adv=0.0")
             
    # Case 3: 存在差异 (正常情况)
    else:
        # epsilon = 1e-8，敏锐捕捉效率差
        safe_std = std_R + 1e-8
        raw_advantages = [(v - mean_R) / safe_std for v in all_R_tau]

    # --- 保护机制 (保持不变) ---
    protection_factor = min(max(current_sr, 0.2), 1.0)

    final_advantages = []
    for r_val, adv_val in zip(all_R_tau, raw_advantages):
        if r_val > 0.0 and adv_val < 0.0:
            adv_val = adv_val * protection_factor
        elif r_val < 0.0 and adv_val > 0.0:
            # 即使是"优秀的失败"，也要限制其优势不能超过 -0.05
            # 防止其看起来比"差劲的成功"还要好
            adv_val = -0.05 
        final_advantages.append(adv_val)

    # Clip 范围限制，防止单样本梯度爆炸
    final_advantages = np.clip(final_advantages, -3.0, 3.0).tolist()

    for step, adv in zip(g_calc_steps, final_advantages):
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
# [修改] 动作新颖性组件 (符合文稿：全局计数 + 成功筛选)
# =============================================================================

def _calculate_action_novelty(g_calc_steps: List[Dict[str, Any]], config):
    """
    [CCAPO V9.2 Fix] 基于全局成功计数的动作新颖性 + 格式奖励饱和机制
    
    修复逻辑:
    针对 "tap(1) 刷屏" 现象。原因为: 新颖性趋近于0，但格式奖励恒定为+0.1，导致 Agent 
    依然能从烂大街的动作中获利。
    
    Fix: 引入 saturation_threshold (饱和阈值)。
    当一个动作的全局计数超过阈值(如50次)时，认为 Agent 已熟练掌握该格式，
    强制移除该步的 Format Bonus，迫使 Agent 探索新动作以获取奖励。
    """
    use_fine_grained = getattr(config, 'use_fine_grained_action', True)
    base_bonus = getattr(config, 'base_bonus', 0.2) 
    
    # [新增] 饱和阈值：超过这个次数，该动作不再发放“低保”(Format Bonus)
    # 建议设为 30-100 之间，给予 Agent 足够的练习次数，但不允许无限刷
    saturation_threshold = getattr(config, 'novelty_saturation_threshold', 50)

    # 1. 路径定义与加载
    count_file = "logger/CCAPO/global_action_counts.json"
    global_counts = _load_global_counts(count_file)
    
    new_counts_update = collections.defaultdict(int)
    
    for step in g_calc_steps:
        # [Filter] 必须是执行成功的动作
        if not step.get('action_success', False):
            step['Z_novelty'] = 0.0 
            continue

        # 生成动作指纹
        act_id = _get_deterministic_id(step, use_fine_grained)
        
        # 获取当前全局计数 + 本 Batch 内已累积的计数
        current_global = global_counts.get(act_id, 0)
        current_batch_inc = new_counts_update[act_id]
        total_count = current_global + current_batch_inc
        
        # --- Part A: 计算新颖性奖励 (不变) ---
        calc_denom = np.sqrt(total_count + 1)
        novelty_score = base_bonus / calc_denom
        step['Z_novelty'] = novelty_score
        
        # --- Part B [CRITICAL FIX]: 格式奖励饱和剥离 ---
        # 如果动作已经太老旧 (total_count > threshold)，说明 Agent 只是在刷单
        # 此时不仅没有新颖性，连之前的 Format Bonus 也要收回
        if total_count > saturation_threshold:
            current_fmt_penalty = step.get('R_format_penalty', 0.0)
            # 只剥离正向的 Bonus，保留负向的 Penalty (报错依然要扣分)
            if current_fmt_penalty > 0:
                step['R_format_penalty'] = 0.0
                # (可选) 标记状态，方便调试
                # step['action_status'] = f"SATURATED::{step.get('action_status','')}"

        # 记录增量
        new_counts_update[act_id] += 1

    # 3. 更新并保存全局计数
    if new_counts_update:
        for k, v in new_counts_update.items():
            global_counts[k] = global_counts.get(k, 0) + v
            
        _save_global_counts(count_file, global_counts)
        ccapo_file_logger.info(f"[NOVELTY] Updated global counts for {len(new_counts_update)} unique actions.")

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
    [Sec 5.1 Revised] 基于结构化共识的 LCS 奖励
    
    Update Logic (V5.2 - Explicit Anchor Persistence):
    1. Anchor Selection: Max R_tau -> Min Steps -> Min Tokens.
    2. Data Injection: Inject 'anchor_uid' and 'is_anchor' into steps for disk persistence.
    """
    use_fine_grained = getattr(config, 'use_fine_grained_action', True)
    beta = getattr(config, 'redundancy_penalty', 0.2)

    batch_id = datetime.datetime.now().strftime("%H:%M:%S.%f")
    match_debug_logger.info(f"\n\n{'='*20} [Batch {batch_id}] Start Consensus LCS (V5.2 Explicit Anchor) {'='*20}")

    # --- 1. 数据准备与 P_global 统计准备 ---
    success_traj_info = {}
    action_type_counts = collections.defaultdict(int) 
    
    for traj_uid, raw_steps in g_calc_trajs.items():
        if not raw_steps: continue
        
        # 只处理 R_core == 1.0 的成功轨迹
        if raw_steps[0].get('R_core') != 1.0: continue
        
        # 轨迹重建
        steps = _reconstruct_trajectory_from_disk(raw_steps)
        is_buffer = steps[0].get('is_buffer_data', False)
        
        valid_steps = []
        act_identifiers = []
        traj_action_types = set()
        
        for step in steps:
            s_type = step.get('action_type', '')
            s_parsed = step.get('parsed_action')
            s_success = step.get('action_success', False)
            is_healed = step.get('is_healed_data', False)
            status = step.get('action_status', '')
            
            keep = False
            if is_buffer or is_healed:
                if not (status.startswith('FAILURE') or status.startswith('ERROR') or status.startswith('FORMAT')):
                    keep = True
            else:
                if s_success:
                    keep = True
            
            if keep:
                valid_steps.append(step)
                # 构造 ID
                if use_fine_grained:
                    act_str = str(s_parsed).strip() if s_parsed else ""
                    act_identifier = f"{s_type}::{act_str}"
                else:
                    act_identifier = s_type
                act_identifiers.append(act_identifier)
                
                traj_action_types.add(s_type)

        if valid_steps:
            for at in traj_action_types:
                action_type_counts[at] += 1

            success_traj_info[traj_uid] = {
                'steps': valid_steps, 
                'act_ids': act_identifiers, 
                'n_steps': len(valid_steps),
                'n_tokens': steps[0].get('traj_total_tokens', 0),
                'r_tau': steps[0].get('R_tau', 0.0), # 确保 R_tau 被记录
                'is_buffer': is_buffer
            }

    if not success_traj_info: 
        match_debug_logger.info("No successful trajectories found for consensus.")
        return

    total_samples = len(success_traj_info)

    # --- 2. 锚点进化 (Anchor Evolution - Revised per User Request) ---
    # 规则：优先 R_tau 高 (Descending)，其次 步数少 (Ascending)，最后 Token 少 (Ascending)
    sorted_trajs = sorted(
        success_traj_info.items(), 
        key=lambda item: (-item[1]['r_tau'], item[1]['n_steps'], item[1]['n_tokens']) 
    )
    
    anchor_uid, anchor_data = sorted_trajs[0]
    anchor_seq = anchor_data['act_ids']
    
    match_debug_logger.info(f"\n>>> SELECTED ANCHOR: {anchor_uid}")
    match_debug_logger.info(f"    R_tau: {anchor_data['r_tau']:.4f}, Len: {anchor_data['n_steps']}")
    match_debug_logger.info(f"    Seq: {anchor_seq}")

    # --- 3. 计算 S_necessity (共识投票) ---
    anchor_len = len(anchor_seq)
    anchor_votes = [0] * anchor_len
    
    def get_lcs_match_indices(seq_a, seq_b):
        m, n = len(seq_a), len(seq_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i-1] == seq_b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        matched_b = set()
        mapping_b_to_a = {}
        
        i, j = m, n
        while i > 0 and j > 0:
            if seq_a[i-1] == seq_b[j-1]:
                matched_b.add(j-1)
                mapping_b_to_a[j-1] = i-1
                i -= 1; j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        return matched_b, None, mapping_b_to_a

    # 投票阶段
    for uid, info in success_traj_info.items():
        curr_seq = info['act_ids']
        if uid == anchor_uid:
            for k in range(anchor_len):
                anchor_votes[k] += 1
        else:
            _, _, mapping_b_to_a = get_lcs_match_indices(anchor_seq, curr_seq)
            for k_b, k_a in mapping_b_to_a.items():
                anchor_votes[k_a] += 1
    
    s_necessity_vec = [(v / (total_samples + 1e-6)) for v in anchor_votes]
    
    # --- 4. 奖励回写 (Write Back) ---
    for traj_uid, info in success_traj_info.items():
        current_seq = info['act_ids']
        current_steps = info['steps']
        
        # [Explicit Persistence] 标记锚点身份
        is_anchor_traj = (traj_uid == anchor_uid)
        
        m_steps_ratio = info['n_steps'] / config.max_steps
        q_step = max(0.0, 1.0 - config.alpha_step * m_steps_ratio)
        
        matched_indices_curr, _, mapping_b_to_a = get_lcs_match_indices(anchor_seq, current_seq)
        
        for idx, step in enumerate(current_steps):
            # 注入锚点元数据 (用于 dp_actor 存盘)
            step['anchor_uid'] = str(anchor_uid)
            step['is_anchor'] = is_anchor_traj
            
            if step.get('is_healed_data'): continue 

            is_match = (idx in matched_indices_curr)
            s_type = step.get('action_type', 'unknown')
            
            s_nec = 0.0
            s_util = 0.0
            
            if is_match:
                # [Case 1] Match
                anchor_idx = mapping_b_to_a[idx]
                s_nec = s_necessity_vec[anchor_idx]
                s_util = 1.0 
                
                i_action = s_nec 
                r_core_raw = i_action * q_step
            else:
                # [Case 2] Mismatch
                p_global = action_type_counts.get(s_type, 0) / (total_samples + 1e-6)
                forgiveness_factor = 1.0 - p_global
                penalty = -beta * forgiveness_factor
                
                i_action = penalty 
                r_core_raw = penalty

            step['R_core_raw'] = r_core_raw
            step['I_action'] = i_action
            step['Q_step'] = q_step
            step['S_necessity'] = s_nec
            step['S_utility'] = s_util

    match_debug_logger.info(f"{'='*20} [Batch {batch_id}] End {'='*20}\n")

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
# 主入口
# =============================================================================
def compute_ccapo_advantages(g_calc_steps: List[Dict[str, Any]], 
                             g_online_steps: List[Dict[str, Any]], 
                             g_buffer_steps: List[Dict[str, Any]], 
                             embedding_model, 
                             config):
    """
    [CCAPO V9.1] Integration with Global Novelty, Dense Format Rewards & Anti-Hacking
    Updates:
    1. Integrated `_calculate_repetition_penalty` to stop reward hacking.
    2. Updated `_calculate_A_traj` with soft penalty (-0.5) for all-fail scenarios.
    """
    ccapo_file_logger.info("=== [CCAPO V9.1] Start Calculation ===")
    
    # 1. 初始化键值 (新增 R_repetition)
    keys_defaults = {
        'R_core_raw': 0.0, 'R_match_raw': 0.0, 'R_format_penalty': 0.0, 
        'S_necessity': 0.0, 'S_utility': 0.0, 'I_action': 0.0, 
        'Q_step': 0.0, 'Z_novelty': 0.0, 'Z_core': 0.0, 'Z_match': 0.0, 
        'TokenCost': 0.0, 'b_stage': 'N/A', 'R_novelty_bonus': 0.0,
        'anchor_uid': 'N/A', 'is_anchor': False,
        'R_step': 0.0,
        # [新增] 重复惩罚字段
        'R_repetition': 0.0 
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
    
    ccapo_file_logger.info("=== [CCAPO V9.1] Done ===")
    return g_calc_steps, sr