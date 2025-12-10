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
    [Sec 4.2 Optimized V4] 计算 A_traj (修复 SR=1 时的崩盘震荡)
    修改点：
    1. 保留 safe_std_threshold 防止 0.0004 分差导致的噪声放大。
    2. [关键新增] 成功样本底线保护 (Success Floor): 即使是 batch 里效率最差的成功轨迹，
       其 Advantage 也不允许低于 -0.2。防止模型把它当成失败样本去"矫枉过正"。
    """
    if not g_calc_steps: return
    all_R_tau = [step.get('R_tau', 0.0) for step in g_calc_steps]
    
    mean_R = np.mean(all_R_tau)
    std_R = np.std(all_R_tau)
    r_ptp = np.ptp(all_R_tau)
    
    # --- 核心数学修正 ---
    
    # Case 1: 全错且一致 (All Fail Same)
    if r_ptp < 1e-6 and mean_R <= 0:
        raw_advantages = [-0.5 for _ in all_R_tau]
        match_debug_logger.info(f"  [A_traj] All Fail Same (Mean={mean_R:.2f}). Adv=-0.5 (Soft Penalty)")

    # Case 2: 全对且一致 (All Success Same)
    elif r_ptp < 1e-6 and mean_R > 0:
        raw_advantages = [0.1 for _ in all_R_tau]
        match_debug_logger.info(f"  [A_traj] All Success Same. Adv=0.1")
             
    # Case 3: 存在差异 (正常情况)
    else:
        # [保留 V3] 最小标准差阈值，防止微小分差爆炸
        min_std_threshold = 0.02
        safe_std = max(std_R, min_std_threshold)
        
        raw_advantages = [(v - mean_R) / safe_std for v in all_R_tau]

    final_advantages = []
    for r_val, adv_val in zip(all_R_tau, raw_advantages):
        
        # [关键修改] 成功样本底线保护 (Success Floor)
        # 针对你观察到的 "6 vs 25" 导致的 SR 崩盘问题。
        # 如果任务成功(R > 0)，哪怕效率极低(R_tau 很低)，归一化后 Adv 变成了 -2.0，
        # 我们也要把它拉回 -0.2。
        # 理由：它完成了任务。Adv -2.0 会让模型以为这是个极其严重的错误(甚至比失败还严重)，
        # 从而导致模型遗忘"如何完成任务"，导致下一轮 SR 暴跌。
        if r_val > 0.0:
            if adv_val < -0.2:
                adv_val = -0.2
        
        # [保留] 失败样本限制 (Failure Ceiling)
        # 如果任务失败，即使它在失败者里排第一，Advantage 也不应该变成正数鼓励它。
        # 限制其上限为 -0.05。
        elif r_val <= 0.0:
            if adv_val > -0.05:
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
    [CCAPO Simplified] 极简版 LCS + Q_step (Fixed for Trajectory Efficiency)
    
    逻辑:
    1. 选出 Anchor (Batch内 R_tau 最高 & 长度最短)。
    2. 计算轨迹的全局效率乘数 Q_step (由 N_success 决定，全轨迹共享)。
    3. 如果当前步在 Anchor 的 LCS 序列中 -> R = 1.0 * Q_step
    4. [Updated] 如果当前步不在 LCS 序列中:
       不再给予固定的 0.5 惩罚，而是基于相对长度 (L_anchor / L_current) 给予动态分。
       R = 0.95 * min(1.0, L_anchor/L_current) * Q_step
       这允许"平替"路径获得高分，消除僵化，同时通过 0.95 保持 Anchor 的收敛引力。
    """
    
    # --- 1. 数据清洗与筛选 ---
    success_traj_info = {}
    
    for traj_uid, raw_steps in g_calc_trajs.items():
        if not raw_steps: continue
        # 只处理成功的轨迹
        if raw_steps[0].get('R_core') != 1.0: continue
        
        # 轨迹重建 (内存/磁盘)
        steps = _reconstruct_trajectory_from_disk(raw_steps)
        
        # 提取用于 LCS 对比的指纹序列
        act_identifiers = []
        valid_steps = []
        
        # [修改] 用于统计 N_success 的计数器
        n_success_count = 0 

        for step in steps:
            # 过滤掉 Buffer 中可能的无效步或 System 步
            # 注意：按照定义，N_success 只包含 action_success=True 的步骤
            is_valid_success = step.get('action_success', False)
            
            # 兼容 Buffer Data (通常默认视为 Success)
            if step.get('is_buffer_data'):
                is_valid_success = True

            if not is_valid_success:
                continue
                
            n_success_count += 1
                
            # 构造指纹 (Action Type + Parsed Content)
            s_type = step.get('action_type', '')
            s_parsed = step.get('parsed_action')
            # 简单处理，确保指纹稳定性
            act_str = str(s_parsed).strip() if s_parsed else ""
            act_id = f"{s_type}::{act_str}"
            
            act_identifiers.append(act_id)
            valid_steps.append(step)

        if valid_steps:
            success_traj_info[traj_uid] = {
                'steps': valid_steps, # 注意：这是过滤后的 steps，回写时需要映射回 raw_steps
                'all_steps_raw': steps, # 保留原始引用以便回写
                'act_ids': act_identifiers, 
                'n_steps': len(valid_steps),
                'n_success_count': n_success_count, # [新增] 明确记录 N_success
                'n_tokens': steps[0].get('traj_total_tokens', 0),
                'r_tau': steps[0].get('R_tau', 0.0)
            }

    if not success_traj_info: 
        return

    # --- 2. 锚点选择 (Anchor Selection) ---
    # 策略：R_tau 越大越好，Step 越少越好
    sorted_trajs = sorted(
        success_traj_info.items(), 
        key=lambda item: (-item[1]['r_tau'], item[1]['n_steps']) 
    )
    anchor_uid, anchor_data = sorted_trajs[0]
    anchor_seq = anchor_data['act_ids']
    
    # [新增] 获取 Anchor 的关键长度，用于后续计算比率
    # 防止除以0，虽然 Anchor 必然 > 0
    anchor_len = max(1.0, float(anchor_data['n_success_count'])) 
    
    # [可视化修复] 标记 Anchor 轨迹
    for s in anchor_data['all_steps_raw']:
        s['is_anchor'] = True
        s['anchor_uid'] = str(anchor_uid) # 写入顶层便于概览

    # --- 3. LCS 计算辅助函数 (标准 DP) ---
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

    # --- 4. 奖励计算 (Binary LCS + Trajectory-Level Q_step) ---
    for traj_uid, info in success_traj_info.items():
        current_seq = info['act_ids']
        raw_steps_to_update = info['all_steps_raw']
        
        # 计算 LCS 匹配索引 (相对于 valid_steps)
        matched_indices_curr = get_lcs_match_indices(anchor_seq, current_seq)
        
        # [修改] 计算全局共享的 Q_step (Trajectory Efficiency Multiplier)
        # 公式: max(0, 1 - alpha * (N_success / MaxSteps))
        n_success = info['n_success_count']
        m_steps_ratio = n_success / config.max_steps
        q_step = max(0.0, 1.0 - config.alpha_step * m_steps_ratio)
        
        # === [新增] 计算动态惩罚比率 (Dynamic Mismatch Ratio) ===
        # 逻辑: (L_anchor / L_current)
        # 如果当前轨迹比 Anchor 长，ratio < 1.0，得分自然下降
        # 如果当前轨迹比 Anchor 短，ratio > 1.0，但在 Anchor 更新前，我们将其截断为 1.0
        # 配合 0.95 的系数，构成"软性约束"
        current_len = max(1.0, float(n_success))
        len_ratio = anchor_len / current_len
        # 上限截断为 1.0，防止在 Anchor 未切换时给出 > 1.0 的 Step 奖励导致值爆炸
        len_ratio = min(1.0, len_ratio) 
        
        # 设定非 Match 步骤的折扣系数 (0.95)
        # 这个 gap 保证了同等长度下，Anchor 依然优于非 Anchor (1.0 vs 0.95)
        mismatch_coeff = 0.95 * len_ratio

        # 遍历更新步骤
        valid_idx_counter = 0
        
        for step in raw_steps_to_update:
            # 初始化
            step['R_core_raw'] = 0.0
            # [可视化修复] 写入 Anchor 元数据，供前端 Tooltip 显示
            step['meta_anchor_uid'] = str(anchor_uid)
            
            should_eval = False
            if step.get('action_success', False) or step.get('is_buffer_data'):
                should_eval = True
            
            if should_eval:
                is_match = (valid_idx_counter in matched_indices_curr)
                
                # [修改] 不再根据 step_index 计算，直接使用计算好的全局 q_step
                step['Q_step'] = q_step 
                
                if is_match:
                    # --- [HIT] 命中锚点关键路径 ---
                    # 奖励 = 1.0 (满) * Q_step
                    r_core_raw = 1.0 * q_step
                    
                    step['R_core_raw'] = r_core_raw
                    step['S_utility'] = 1.0 
                    
                    # [可视化修复] 这里的赋值决定了前端显示 "Match"
                    step['I_action'] = 1.0 
                    step['meta_lcs_id'] = f"MATCH_IDX_{valid_idx_counter}"
                else:
                    # --- [MISS] 偏离锚点 (但依然成功) ---
                    # 旧逻辑: 0.5 * q_step (固定惩罚)
                    # 新逻辑: 0.95 * (L_anchor/L_curr) * q_step (动态效用)
                    # 保底: max(0.5, ...) 防止极长轨迹导致负面影响太大，虽不常见
                    
                    dynamic_reward = mismatch_coeff * q_step
                    r_core_raw = max(0.5 * q_step, dynamic_reward)
                    
                    step['R_core_raw'] = r_core_raw
                    
                    # 记录实际得分比率，方便调试
                    step['S_utility'] = mismatch_coeff 
                    
                    # [可视化修复] 这里的赋值决定了前端显示 "Redundant"
                    # 这里改为 0.0 (中性)，表示它不是错误，只是"非标准"
                    step['I_action'] = 0.0 
                    step['meta_lcs_id'] = f"ALT_PATH_{mismatch_coeff:.2f}"

                valid_idx_counter += 1
            else:
                # 失败步骤 (Action Success = False)
                # 根据你的设计，失败步骤的 R_core_raw 为 0
                step['R_core_raw'] = 0.0
                step['Q_step'] = q_step # 即使失败，也记录下当轮的 Q_step 供参考
                step['I_action'] = 0.0
                step['meta_lcs_id'] = "N/A"

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
    [Dr. GRPO V3 - Success-Conditional Tanh-Gating]
    
    Innovation:
    1. Failure Invariance: Failed trajectories always get W=1.0. 
       (Prevents weakening the penalty for long failures).
    2. Success-Only Statistics: Length stats are calculated ONLY from successful trajectories.
    3. Parameter-Free: Uses tanh to naturally bound weights to (0, 2).
    
    Logic:
       if is_fail: W = 1.0
       if is_success: W = 1.0 + tanh( (Mean_Succ - L_i) / Std_Succ )
    """
    if not steps_list: return

    # 1. Group by Trajectory
    trajs = _group_steps_by_traj(steps_list)
    
    # 2. Extract Success Trajectory Lengths
    succ_lengths = []
    traj_info_map = {} # uid -> {'length': int, 'is_success': bool}

    for uid, steps in trajs.items():
        if not steps: continue
        
        # Check success from the first step (which holds trajectory-level info)
        # Assuming R_core == 1.0 means success
        r_core = steps[0].get('R_core', 0.0)
        is_success = (r_core == 1.0)
        length = len(steps)
        
        traj_info_map[uid] = {'length': length, 'is_success': is_success}
        
        if is_success:
            succ_lengths.append(length)

    # 3. Calculate Statistics (Only if we have successes)
    mean_succ = 0.0
    std_succ = 1.0 # default to avoid div by zero
    has_success = len(succ_lengths) > 0
    
    if has_success:
        arr = np.array(succ_lengths, dtype=float)
        mean_succ = np.mean(arr)
        std_succ = np.std(arr) + 1e-6 # epsilon

    # 4. Apply Weights
    for step in steps_list:
        uid_val = step['traj_uid']
        if isinstance(uid_val, (np.ndarray, torch.Tensor)):
             uid_val = uid_val.item()
        
        info = traj_info_map.get(uid_val)
        if not info: continue
        
        # Default weight
        weight = 1.0
        z_score = 0.0
        
        # --- Core Logic ---
        if not info['is_success']:
            # Case A: Failure Invariance
            # 失败者保持原样，确保吃到满额的 Negative Advantage
            weight = 1.0
        elif not has_success:
             # Should not happen if is_success is true, but for safety
             weight = 1.0
        else:
            # Case B: Success-Conditional Tanh-Gating
            # 只在成功者内部卷效率
            l_i = info['length']
            
            # Z-Score: (群体平均 - 我的长度) / 标准差
            # 短于平均 -> Z > 0 -> W > 1
            # 长于平均 -> Z < 0 -> W < 1
            z_score = (mean_succ - float(l_i)) / std_succ
            
            # Tanh Mapping: Range (0, 2)
            # Center at 1.0
            weight = 1.0 + np.tanh(z_score)
            
            # [Safety] 防止权重过小导致样本失效
            # 虽然 25步的成功(vs 5步平均)确实很烂，但好歹是成功，给个保底 0.2
            weight = max(0.2, weight)

        # Write Metadata
        step['W_length'] = weight
        step['meta_z_score'] = z_score
        
        # Apply to Advantages
        if 'advantages' in step:
            # Apply Dr. GRPO weight
            step['advantages'] *= weight
            
            # [Final Safety Clip] 
            # 无论如何，防止最终梯度爆炸，这是 PPO 的最后一道防线
            step['advantages'] = np.clip(step['advantages'], -4.0, 4.0).item()
            
    match_debug_logger.info(f"  [Dr. GRPO V3] Success Count: {len(succ_lengths)}. "
                           f"Mean Succ Len: {mean_succ:.2f}. Applied Weights.")

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