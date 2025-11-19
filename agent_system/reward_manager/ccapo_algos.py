# agent_system/reward_manager/ccapo_algos.py

import torch
import numpy as np
import collections
from typing import List, Dict, Any
from sentence_transformers import util
import logging
import os # <-- ✅ [日志] 新增

# --- 1. 标准日志器 (用于 STDOUT / 主日志) ---
logger = logging.getLogger(__name__)

# --- 2. ✅ [日志] 专用文件日志器 (用于 logger/CCAPO/ccapo_operations.log) ---
ccapo_file_logger = logging.getLogger("CCAPO_FILE")
ccapo_file_logger.setLevel(logging.INFO) # 捕获 INFO 及以上级别
ccapo_file_logger.propagate = False      # 防止重复记录到 root logger

# 仅在日志器没有处理器时才添加，以防止重复
if not ccapo_file_logger.handlers:
    try:
        log_dir = "logger/CCAPO"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "ccapo_operations.log")
        
        # 创建文件处理器 (追加模式)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - [CCAPO_FILE] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        ccapo_file_logger.addHandler(file_handler)
        ccapo_file_logger.info("--- CCAPO 专用文件日志器已初始化 ---")
        
    except Exception as e:
        logger.error(f"[CCAPO] 无法创建专用文件日志器: {e}")
# --- 日志设置结束 ---


# --- 辅助函数：分组 ---
def _group_steps_by_traj(steps_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按 traj_uid 将扁平化的步骤列表重新组合成轨迹字典"""
    trajs = collections.defaultdict(list)
    for step in steps_list:
        # 确保我们处理的是PyTorch张量或Numpy数组
        traj_uid_val = step['traj_uid']
        if isinstance(traj_uid_val, np.ndarray):
            traj_uid_val = traj_uid_val.item()
        elif isinstance(traj_uid_val, torch.Tensor):
            traj_uid_val = traj_uid_val.item()
            
        trajs[traj_uid_val].append(step)
    
    # 确保每个轨迹内的步骤是排序的
    for traj_uid in trajs:
        trajs[traj_uid].sort(key=lambda s: s.get('step_index', 0))
    return trajs

# --- ✅ [CCAPO V2] 新增: 辅助函数：标准化 (Z-Score) ---
def _standardize(values: List[float]) -> List[float]:
    """Z-score standardization (mathcal{Z}(X))"""
    if not values:
        return []
    mean = np.mean(values)
    std = np.std(values) + 1e-6
    # 防止 std 接近 0 导致 nan/inf
    if std < 1e-6:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]

# --- Sec 4: 宏观轨迹奖励 R_tau ---
def _calculate_R_tau(g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    """
    计算 R_tau, R_core 并将其写回每个步骤
    (✅ [CCAPO V2] 修改: 采用 R_core 和效率乘数)
    """
    for traj_uid, steps in g_calc_trajs.items():
        if not steps: continue
        first_step = steps[0]
        
        R_success = 1.0 if first_step['traj_task_completed'] else -1.0
        total_steps = first_step['traj_total_steps']
        total_tokens = first_step['traj_total_tokens']
        
        # --- ✅ [CCAPO V2] 新版逻辑 (Sec 4) ---
        is_shortcut = (R_success > 0 and total_steps < config.min_reasonable_steps)
        
        R_core = 0.0
        if R_success <= 0:
            R_core = -1.0  # 任务失败
        elif not is_shortcut:
            R_core = 1.0   # 有效成功
        # else: 作弊成功 (IsShortcut=True), R_core = 0.0 (默认)
        
        R_tau = R_core # 默认值 (失败或作弊)
        
        if R_core == 1.0:
            # 只有有效成功才考虑效率
            # M_steps = (1.0 - total_steps / config.max_steps)
            # print(f"Traj {traj_uid}: total_steps={total_steps}, M_steps={M_steps}") # <-- ✅ [日志] 替换
            # M_token = (1.0 - total_tokens / config.max_tokens)
            # print(f"Traj {traj_uid}: total_tokens={total_tokens}, M_token={M_token}") # <-- ✅ [日志] 替换
            # # 确保效率乘数不为负
            # M_steps = max(0.0, M_steps)
            # print(f"Traj {traj_uid}: Clipped M_steps={M_steps}") # <-- ✅ [日志] 替换
            # M_token = max(0.0, M_token)
            # print(f"Traj {traj_uid}: Clipped M_token={M_token}") # <-- ✅ [日志] 替换

            m_steps_ratio = total_steps / config.max_steps
            m_token_ratio = total_tokens / config.max_tokens

            M_steps = (max(0.0, 1.0 - m_steps_ratio))**0.5
            ccapo_file_logger.info(f"[R_tau] Traj {traj_uid}: total_steps={total_steps}, m_steps_ratio={m_steps_ratio:.2f}, M_steps={M_steps:.4f} (sqrt scaled)") # <-- ✅ [日志] 替换
            M_token = (max(0.0, 1.0 - m_token_ratio))**0.5
            ccapo_file_logger.info(f"[R_tau] Traj {traj_uid}: total_tokens={total_tokens}, m_token_ratio={m_token_ratio:.2f}, M_token={M_token:.4f} (sqrt scaled)") # <-- ✅ [日志] 替换
        
            R_tau = R_core * M_steps * M_token
        # --- 新版 CCAPO 逻辑结束 ---
        
        for step in steps:
            step['R_tau'] = R_tau
            step['R_core'] = R_core # 存储 R_core 供 R_step 使用
            ccapo_file_logger.info(f"[R_tau] Traj {traj_uid} Step {step.get('step_index', 0)}: R_tau={R_tau}, R_core={R_core}") # <-- ✅ [日志] 替换

# --- Sec 4: 宏观轨迹优势 A_traj ---
def _calculate_A_traj(g_calc_steps: List[Dict[str, Any]]):
    """计算 A_traj 并将其写回每个步骤"""
    # 注意：我们应该只在 G_calc *轨迹* 级别上进行标准化
    # 为了简化，我们收集 G_calc 中所有 R_tau 的 unique 值
    all_R_tau = list(set(step['R_tau'] for step in g_calc_steps if 'R_tau' in step))
    if not all_R_tau:
        for step in g_calc_steps: step['A_traj'] = 0.0
        return
        
    mean_R_tau = np.mean(all_R_tau)
    std_R_tau = np.std(all_R_tau) + 1e-6
    
    for step in g_calc_steps:
        step['A_traj'] = (step.get('R_tau', mean_R_tau) - mean_R_tau) / std_R_tau

# --- Sec 5: R_format_novelty ---
# --- Sec 5: R_format_penalty (V3: 解耦) ---
def _calculate_R_format_penalty(g_calc_steps: List[Dict[str, Any]], config):
    """
    计算 R_format_penalty (硬惩罚)
    并将其作为 'R_format_penalty' 键写入每个步骤。
    (✅ [CCAPO V3] 新增: 将惩罚与新颖度解耦)
    """
    for step in g_calc_steps:
        action_status = step.get('action_status', '')
        R_format_penalty = 0.0
        
        if action_status.startswith('FORMAT_ERROR'):
            R_format_penalty = config.penalty_format_error
            ccapo_file_logger.info(f"[R_format] Step {step.get('step_index', 0)}: FORMAT_ERROR detected, applying penalty {R_format_penalty}") # <-- ✅ [日志] 替换
        elif action_status.startswith('FAILURE'):
            R_format_penalty = config.penalty_failure
            ccapo_file_logger.info(f"[R_format] Step {step.get('step_index', 0)}: FAILURE detected, applying penalty {R_format_penalty}") # <-- ✅ [日志] 替换
            
        step['R_format_penalty'] = R_format_penalty

# --- Sec 5: R_novelty_bonus (V3: 解耦) ---
def _calculate_R_novelty_bonus(g_calc_steps: List[Dict[str, Any]], config):
    """
    计算 R_novelty_bonus (仅针对成功动作)
    并将其作为 'R_novelty_bonus' 键写入每个步骤。
    (✅ [CCAPO V3] 修改: 从 _calculate_R_format_novelty 拆分)
    """
    # 1. (全局) 计算 G_calc 中所有成功动作的计数
    ActionSuccessCount = collections.defaultdict(int)
    for step in g_calc_steps:
        if step.get('action_success', False):
            action_type = step.get('action_type')
            if action_type:
                ActionSuccessCount[action_type] += 1
                ccapo_file_logger.info(f"[R_novelty] Counting success for action '{action_type}': total now {ActionSuccessCount[action_type]}") # <-- ✅ [日志] 替换
    
    # 2. 遍历所有步骤，计算 R_novelty_bonus
    for step in g_calc_steps:
        R_novelty = 0.0
        # 仅当动作成功时才计算新颖度奖励
        if step.get('action_success', False):
            action_type = step.get('action_type')
            if action_type:
                count = ActionSuccessCount[action_type]
                R_novelty = config.base_bonus / (count**0.5 + 1e-6)
                ccapo_file_logger.info(f"[R_novelty] Calculating R_novelty_bonus for action '{action_type}' with count {count}: {R_novelty}") # <-- ✅ [日志] 替换
        
        step['R_novelty_bonus'] = R_novelty

# --- Sec 5.1: 微观步骤奖励 (成功轨迹) ---
def _calculate_R_step_success(g_calc_steps: List[Dict[str, Any]], g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    """
    计算 R_step 的“核心”原始分量 (R_core_raw)
    (✅ [CCAPO V2] 修改：不再计算 R_step，只存储 R_core_raw 供后续标准化)
    (✅ [CCAPO V4] 修复: 重新计算 n_success 来修复 b_stage bug; 升级日志)
    (✅ [V5 修复]：使用 Q_step 效率加权 S_necessity)
    """
    ccapo_file_logger.info("--- [CCAPO] Phase 3.1: 计算 R_core_raw (成功轨迹) ---")
    
    # --- Sec 5.1.3.2: Q_economy ---
    all_token_costs = []
    if not g_calc_steps:
        ccapo_file_logger.warning("[R_success] g_calc_steps 为空, 跳过 Q_economy 计算。")
        avg_token_cost = 1e-6 # 避免除以零
    else:
        for step in g_calc_steps:
            usage = step.get('step_token_usage', {})
            cost = (config.w_out * usage.get('completion_tokens', 0) +
                    config.w_in * usage.get('prompt_tokens', 0))
            step['TokenCost'] = cost
            all_token_costs.append(cost)
        avg_token_cost = np.mean(all_token_costs) + 1e-6
    
    for step in g_calc_steps:
        # Q_economy 使用 tanh，范围 (0, 2)
        q_econ = 1.0 - np.tanh((step.get('TokenCost', 0) - avg_token_cost) / avg_token_cost)
        step['Q_economy'] = q_econ
        # ✅ [日志] 调试 Q_economy
        ccapo_file_logger.debug(f"  [Q_econ] Step (traj_uid={step.get('traj_uid')}, t={step.get('step_index')}): "
                                f"Cost={step.get('TokenCost', 0):.2f}, AvgCost={avg_token_cost:.2f}, Q_economy={q_econ:.4f}")

    # --- Sec 5.1.2: I_action (S_necessity, S_utility) ---
    
    # ======================= ✅ [ Bug 1 修复 ] =======================
    # 1. (Trajectory-local) 预计算 N_success(tau)
    # 我们必须重新计算，因为 steps[0] 的元数据可能是 G_online/G_buffer 混合而不可靠。
    n_success_map = {}
    for traj_uid, steps in g_calc_trajs.items():
        n_success = sum(1 for s in steps if s.get('action_success', False))
        n_success_map[traj_uid] = n_success
        if n_success == 0:
             ccapo_file_logger.debug(f"  [b_stage] Traj {traj_uid}: 预计算 n_success = 0")
    # ======================= 修复结束 =======================

    # ======================= ✅ [ V5 效率加权修复 ] =======================
    # 1.1 (Trajectory-local) 预计算 Q_step(tau) 作为 S_necessity 的权重
    ccapo_file_logger.info("  [I_action] 预计算 Q_step 作为 S_necessity 的轨迹权重:")
    traj_q_step_weights = {}
    total_q_step_weight_sum = 0.0  # S_necessity 的新分母
    
    for traj_uid, steps in g_calc_trajs.items():
        if not steps or steps[0].get('R_core') != 1.0:
            traj_q_step_weights[traj_uid] = 0.0
            continue
            
        n_success = n_success_map.get(traj_uid, 0)
        
        # Sec 5.1.3.1: 计算 Q_step(tau)
        q_step = max(0, 1.0 - config.alpha_step * (n_success / config.max_steps))
        
        traj_q_step_weights[traj_uid] = q_step
        total_q_step_weight_sum += q_step
        
        ccapo_file_logger.debug(f"    - Traj {traj_uid}: n_success={n_success}, Q_step (Weight)={q_step:.4f}")

    ccapo_file_logger.info(f"  [I_action] 总成功权重 (Sum of Q_step): {total_q_step_weight_sum:.4f}")
    if total_q_step_weight_sum < 1e-6:
        total_q_step_weight_sum = 1e-6 # 防止除以零
    # ======================= V5 修复结束 =======================

    # 1.2 计算 S_utility 的全局基线 (保持不变)
    total_trajs_in_calc = len(g_calc_trajs)
    successful_trajs_in_calc = set(uid for uid, s in g_calc_trajs.items() if s[0].get('R_core') == 1.0)
    P_success_global = len(successful_trajs_in_calc) / (total_trajs_in_calc + 1e-6)
    
    ccapo_file_logger.info(f"  [I_action] P_success_global (G_calc): {len(successful_trajs_in_calc)} / {total_trajs_in_calc} = {P_success_global:.4f}")

    # ======================= ✅ [ V5 效率加权修复 ] =======================
    # 1.3 定义新的 stats 结构
    stats = collections.defaultdict(lambda: {
        "total_steps": 0,
        "success_trajs": set(),            # 保留: 用于 S_utility
        "total_trajs": set(),            # 保留: 用于 S_utility
        "weighted_success_sum": 0.0,       # 新增: 用于 S_necessity
        "seen_in_success_trajs": set()     # 辅助: 防止(action, stage)在同一轨迹中被重复加权
    })
    # ======================= V5 修复结束 =======================

    # 1.4 计算 b_stage 并收集统计数据
    for traj_uid, steps in g_calc_trajs.items():
        # ======================= ✅ [ Bug 1 修复 ] =======================
        n_success = n_success_map[traj_uid] # 使用预计算的值
        # ======================= 修复结束 =======================
        
        current_successful_step_t = 0
        for step in steps:
            if step.get('action_success', False):
                current_successful_step_t += 1
                k_norm = current_successful_step_t / n_success if n_success > 0 else 0
                
                stage = 'Late'
                if k_norm <= 0.33: stage = 'Early'
                elif k_norm <= 0.66: stage = 'Mid'
                step['b_stage'] = stage
                
                ccapo_file_logger.debug(f"  [b_stage] Traj {traj_uid} Step {step.get('step_index')}: "
                                        f"k_t={current_successful_step_t}, n_success={n_success}, k_norm={k_norm:.2f} -> Stage='{stage}'")
                
                # 收集统计
                action_type = step.get('action_type')
                if action_type:
                    key = (action_type, stage)
                    stats[key]["total_steps"] += 1
                    stats[key]["total_trajs"].add(traj_uid) # 用于 S_utility

                    # ======================= ✅ [ V5 效率加权修复 ] =======================
                    current_traj_weight = traj_q_step_weights.get(traj_uid, 0.0)
                    if current_traj_weight > 0: # 这意味着 R_core == 1.0
                        stats[key]["success_trajs"].add(traj_uid) # 用于 S_utility

                        # 确保每个 (action, stage) 组合在同一轨迹中只被加权一次
                        if traj_uid not in stats[key]["seen_in_success_trajs"]:
                            stats[key]["weighted_success_sum"] += current_traj_weight
                            stats[key]["seen_in_success_trajs"].add(traj_uid)
                    # ======================= V5 修复结束 =======================
            else:
                step['b_stage'] = 'N/A'

    # 2. 计算 I_action
    I_action_cache = {}
    ccapo_file_logger.info("  [I_action] 计算全局 I_action 缓存:")
    for key, data in stats.items():
        action_type, stage = key
        
        # ======================= ✅ [ V5 效率加权修复 ] =======================
        # S_necessity (Sec 5.1.2.2) - 使用 Q_step 加权
        S_necessity = data["weighted_success_sum"] / total_q_step_weight_sum
        # ======================= V5 修复结束 =======================
        
        # S_utility (Sec 5.1.2.3) (保持不变)
        P_success_given_action = len(data["success_trajs"]) / (len(data["total_trajs"]) + 1e-6)
        S_utility = P_success_given_action / (P_success_global + 1e-6)
        
        I_action_cache[key] = S_necessity * S_utility
        
        # ======================= ✅ [ V5 效率加权修复 ] =======================
        # 更新日志以反映加权
        ccapo_file_logger.info(f"      - Key=({action_type}, {stage}): "
                               f"S_nec={S_necessity:.4f} (WeightedSum={data['weighted_success_sum']:.2f}/{total_q_step_weight_sum:.2f}), "
                               f"S_util={S_utility:.4f} (P_cond={P_success_given_action:.4f}), "
                               f"==> I_action={I_action_cache[key]:.4f}")
        # ======================= V5 修复结束 =======================

    # --- Sec 5.1.3 & 5.1: 仅计算 R_core_raw ---
    ccapo_file_logger.info("  [R_core_raw] 为 R_core=1.0 的步骤计算 R_core_raw:")
    for step in g_calc_steps:
        if step.get('R_core') != 1.0:
            continue 
        
        log_ctx = f"[Traj {step.get('traj_uid')}, t={step.get('step_index')}]"
        
        if not step.get('action_success', False):
            step['R_core_raw'] = 0.0 # 动作失败，核心奖励为0
            ccapo_file_logger.debug(f"    {log_ctx}: Action failed, R_core_raw = 0.0")
            continue
        
        # Q_step (Sec 5.1.3.1)
        # ======================= ✅ [ Bug 1 修复 ] & [ V5 修复 ] =======================
        # 确保使用 n_success_map 中正确的 traj_uid (用于日志)
        n_success = n_success_map.get(step.get('traj_uid'), 0)
        # [V5] 直接从预计算的权重中获取 q_step
        q_step = traj_q_step_weights.get(step.get('traj_uid'), 0.0) 
        # ======================= 修复结束 =======================
        
        q_economy = step.get('Q_economy', 1.0) # 已在上面计算
        q_efficiency = q_step * q_economy
        
        key = (step.get('action_type'), step.get('b_stage'))
        i_action = I_action_cache.get(key, 0.0) 
        
        step['R_core_raw'] = i_action * q_efficiency
        
        ccapo_file_logger.debug(f"    {log_ctx}: Action='{key[0]}', Stage='{key[1]}' "
                                f"==> I_action={i_action:.4f}, "
                                f"Q_step={q_step:.4f} (n_succ={n_success}), "
                                f"Q_econ={q_economy:.4f} "
                                f"==> R_core_raw={step['R_core_raw']:.4f}")

# --- Sec 5.2: 微观步骤奖励 (失败轨迹) ---
def _calculate_R_step_fail(g_calc_steps: List[Dict[str, Any]], g_buffer_steps: List[Dict[str, Any]], embedding_model, config):
    """
    计算 R_step 的“匹配”原始分量 (R_match_raw)
    (✅ [CCAPO V2] 修改：使用 R_step (S(a_j*)) 作为匹配源，不使用 w_match, 仅存储 R_match_raw)
    (✅ [CCAPO V4] 升级日志)
    """
    ccapo_file_logger.info(f"--- [CCAPO] Phase 3.2: 计算 R_match_raw (失败轨迹) ---")
    
    # --- 1. 预计算 STDB (G_buffer) 步骤的分数和嵌入 ---
    stdb_step_scores = collections.defaultdict(list)
    stdb_thoughts_to_embed = []
    stdb_thought_map = {} # map id -> (action, index_in_scores_list)

    idx_counter = 0
    ccapo_file_logger.info(f"  [R_fail] 正在索引 {len(g_buffer_steps)} 个 STDB 步骤...")
    for step in g_buffer_steps:
        # G_buffer 都是成功轨迹 (R_core == 1.0)
        # R_step 此时已由 R_step_success (Z_core + w_N*Z_novelty) 填充
        score = step.get('R_step', 0.0)
        action = step.get('parsed_action')
        thought = step.get('thought')
        
        if not action or not thought or step.get('R_core') != 1.0:
            continue
            
        stdb_step_scores[action].append({'score': score})
        stdb_thoughts_to_embed.append(thought)
        stdb_thought_map[idx_counter] = (action, len(stdb_step_scores[action]) - 1)
        idx_counter += 1

    if not stdb_thoughts_to_embed:
        ccapo_file_logger.warning("  [R_fail] STDB (G_buffer) 为空或无有效 (thought, action) 对。跳过 R_match_raw 计算。")
        return
        
    ccapo_file_logger.info(f"  [R_fail] 正在为 {len(stdb_thoughts_to_embed)} 个 STDB thoughts 编码...")
    stdb_embeddings = embedding_model.encode(stdb_thoughts_to_embed, convert_to_tensor=True)
    
    # 将嵌入存回字典
    for i, emb in enumerate(stdb_embeddings):
        action, list_idx = stdb_thought_map[i]
        stdb_step_scores[action][list_idx]['embedding'] = emb

    # --- 2. 匹配失败步骤 ---
    fail_steps_to_embed = []
    fail_step_map = {} # map id -> step_dict
    idx_counter = 0
    
    fail_steps_count = 0
    for step in g_calc_steps:
        # --- ✅ [CCAPO V2] 修改: 只处理 R_core == -1.0 的步骤 ---
        if step.get('R_core') != -1.0:
            continue
        
        fail_steps_count += 1
        action = step.get('parsed_action')
        thought = step.get('thought')
        
        if not action or not thought:
            ccapo_file_logger.debug(f"    [Traj {step.get('traj_uid')}, t={step.get('step_index')}]: 跳过 (缺少 action/thought)。")
            continue
        if action not in stdb_step_scores:
            ccapo_file_logger.debug(f"    [Traj {step.get('traj_uid')}, t={step.get('step_index')}]: 跳过 (Action '{action}' 在 STDB 中无匹配)。")
            continue
        
        fail_steps_to_embed.append(thought)
        fail_step_map[idx_counter] = step
        idx_counter += 1
        
    if not fail_steps_to_embed:
        ccapo_file_logger.info(f"  [R_fail] G_calc 中有 {fail_steps_count} 个失败步骤，但没有可用于嵌入匹配的步骤。")
        return
        
    ccapo_file_logger.info(f"  [R_fail] 正在为 {len(fail_steps_to_embed)} 个 G_calc 失败 thoughts 编码...")
    fail_embeddings = embedding_model.encode(fail_steps_to_embed, convert_to_tensor=True)

    # --- 3. 计算相似度并赋分 ---
    ccapo_file_logger.info(f"  [R_fail] 正在计算 R_match_raw (相似度匹配)...")
    for i, emb_t in enumerate(fail_embeddings):
        step = fail_step_map[i]
        action = step['parsed_action']
        log_ctx = f"[Traj {step.get('traj_uid')}, t={step.get('step_index')}]"
        
        action_matches = stdb_step_scores[action]
        if not any('embedding' in m for m in action_matches):
            ccapo_file_logger.debug(f"    {log_ctx}: Action '{action}' 在 STDB 中没有嵌入向量 (???)")
            continue
            
        compare_embeddings = torch.stack([m['embedding'] for m in action_matches if 'embedding' in m]).to(emb_t.device)
        scores = torch.tensor([m['score'] for m in action_matches if 'embedding' in m], device=emb_t.device)
        
        cos_sims = util.cos_sim(emb_t, compare_embeddings)[0]
        
        max_score = 0.0
        matches = torch.where(cos_sims > config.similarity_threshold)[0]
        
        if len(matches) > 0:
            best_sim_idx = torch.argmax(cos_sims[matches]) # 找到相似度最高的索引
            best_match_idx_in_scores = matches[best_sim_idx] # 映射回 scores 张量
            
            # ✅ [CCAPO V2] 修改: R_match_raw = S(a_j*)
            # 我们不使用相似度加权，只取匹配上的 R_step 最高分
            max_score = torch.max(scores[matches]).item()
            
            best_sim_score = cos_sims[best_match_idx_in_scores].item()
            score_from_best_sim = scores[best_match_idx_in_scores].item()
            
            ccapo_file_logger.debug(f"    {log_ctx}: Action '{action}'. 找到 {len(matches)} 个匹配 (>{config.similarity_threshold:.2f})。")
            ccapo_file_logger.debug(f"      -> 最高相似度: {best_sim_score:.4f} (其 R_step={score_from_best_sim:.4f})")
            ccapo_file_logger.debug(f"      -> 最高 R_step: {max_score:.4f} (使用此分数)")
        else:
             ccapo_file_logger.debug(f"    {log_ctx}: Action '{action}'. 未找到相似匹配。 R_match_raw=0.0")

        step['R_match_raw'] = max_score
        # (移除 R_step 的计算)

# --- Sec 5: 微观步骤优势 A_step ---
def _calculate_A_step(g_calc_steps: List[Dict[str, Any]]):
    """计算 A_step 并将其写回每个步骤"""
    # (此函数无需修改, 它正确地标准化了最终的 R_step)
    all_R_step = [step['R_step'] for step in g_calc_steps if 'R_step' in step]
    if not all_R_step:
        for step in g_calc_steps: step['A_step'] = 0.0
        return

    mean_R_step = np.mean(all_R_step)
    std_R_step = np.std(all_R_step) + 1e-6
    
    # ✅ [日志] 记录 A_step 的标准化
    ccapo_file_logger.info(f"--- [CCAPO] Phase 5: 标准化 R_step (计算 A_step) ---")
    ccapo_file_logger.info(f"  [A_step] R_step (G_calc) 分布: mean={mean_R_step:.4f}, std={std_R_step:.4f} (来自 {len(all_R_step)} 个步骤)")
    
    for step in g_calc_steps:
        r_step = step.get('R_step', mean_R_step)
        a_step = (r_step - mean_R_step) / std_R_step
        step['A_step'] = a_step
        ccapo_file_logger.debug(f"  [A_step] [Traj {step.get('traj_uid')}, t={step.get('step_index')}]: R_step={r_step:.4f} -> A_step={a_step:.4f}")

# --- [新函数] Sec 5 & Sec 1: 局部分离的优势计算 ---
def _calculate_final_advantages_for_subset(steps_subset: List[Dict[str, Any]], omega: float, subset_name: str):
    """
    [GEM FIX V2/V3]
    计算 A_traj, A_step, 和最终标准化的 Advantages *局部*
    为给定的步骤子集 (例如 G_online 或 G_buffer)。
    这修复了“漏洞二 (信号扭曲)”和“漏洞三 (双重标准化)”。
    """
    ccapo_file_logger.info(f"--- [CCAPO] Phase 9 ({subset_name}): 为 {len(steps_subset)} 个步骤计算局部优势 ---")
    if not steps_subset:
        ccapo_file_logger.info(f"  [{subset_name}_Adv] 子集为空，跳过。")
        return

    # 1. 局部计算 A_traj (基于此子集的 R_tau 统计)
    #    (日志记录在 _calculate_A_traj 内部)
    _calculate_A_traj(steps_subset)
    
    # 2. 局部计算 A_step (基于此子集的 R_step 统计)
    #    (日志记录在 _calculate_A_step 内部)
    _calculate_A_step(steps_subset)
    
    # 3. [修复漏洞三] 计算原始 A_final = A_traj + omega * A_step
    all_A_final_raw = []
    for step in steps_subset:
        a_traj = step.get('A_traj', 0.0)
        a_step = step.get('A_step', 0.0)
        a_final_raw = a_traj + omega * a_step
        step['A_final_raw'] = a_final_raw
        all_A_final_raw.append(a_final_raw)
        
    # 4. [修复漏洞三] 局部标准化最终优势
    mean_A_final = np.mean(all_A_final_raw)
    std_A_final = np.std(all_A_final_raw) + 1e-6
    ccapo_file_logger.info(f"  [{subset_name}_Adv] 原始 A_final 分布: mean={mean_A_final:.4f}, std={std_A_final:.4f}")

    for step in steps_subset:
        # 将最终的、标准化的优势写入 'advantages' 键
        step['advantages'] = (step['A_final_raw'] - mean_A_final) / std_A_final
        ccapo_file_logger.debug(f"  [{subset_name}_Adv] [Traj {step.get('traj_uid')}, t={step.get('step_index')}]: "
                                f"A_final = A_traj({step.get('A_traj', 0.0):.4f}) + o({omega:.2f}) * A_step({step.get('A_step', 0.0):.4f}) "
                                f"==> Raw={step['A_final_raw']:.4f} "
                                f"==> Norm_Adv={step['advantages']:.4f}")

# --- 主函数 ---
def compute_ccapo_advantages(g_calc_steps: List[Dict[str, Any]], g_online_steps: List[Dict[str, Any]], embedding_model, config):
    """
    计算 CCAPO 奖励和优势的主函数。 (✅ [CCAPO V3] 修改: 解耦格式惩罚)
    修改 g_calc_steps 列表，为其添加 'R_tau', 'A_traj', 'R_step', 'A_step', 'advantages'。
    (✅ [CCAPO V4] 升级日志)
    (✅ [Gem] 修复漏洞 2 和 3：分离优势计算)
    """
    ccapo_file_logger.info(f"================ [CCAPO V4] 开始计算优势 ================")
    ccapo_file_logger.info(f"G_calc 步骤数: {len(g_calc_steps)}, G_online 步骤数: {len(g_online_steps)}")
    
    g_calc_trajs = _group_steps_by_traj(g_calc_steps)
    # [GEM 修复] g_buffer_steps 必须从 g_calc_steps 过滤出来，
    # 因为 g_online_steps 可能不包含 g_buffer_steps (例如在 collate_fn 之后)
    g_buffer_steps = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    ccapo_file_logger.info(f"G_calc 轨迹数: {len(g_calc_trajs)}, G_buffer 步骤数: {len(g_buffer_steps)}")
    
    # --- 1. Sec 4: R_tau, R_core (统一计算) ---
    ccapo_file_logger.info("--- [CCAPO] Phase 1: 统一计算 R_tau, R_core ---")
    # [GEM 修复] R_tau 和 R_core 在 G_calc 上统一计算
    _calculate_R_tau(g_calc_trajs, config)
    # [GEM 修复] _calculate_A_traj(g_calc_steps) 已删除 (必须分离)
    
    # --- 2. Sec 7: 计算 SR (lambda_SR) 和 w_N ---
    ccapo_file_logger.info("--- [CCAPO] Phase 2: 计算 SR (lambda_SR) 和 w_N (来自 G_online) ---")
    online_trajs = _group_steps_by_traj(g_online_steps)
    online_trajs_count = len(online_trajs)
    online_success_count = 0
    for traj_uid, steps in online_trajs.items():
        if steps and steps[0].get('R_core') == 1.0:
            online_success_count += 1
            
    success_rate = online_success_count / (online_trajs_count + 1e-6)
    
    # --- [Gem 修复] 保留您在上一步中添加的动态 w_N ---
    max_w_N = config.get("max_w_N", 0.8) # 从 config 读取
    min_w_N = config.get("min_w_N", 0.2) # 从 config 读取
    w_N = min_w_N + (max_w_N - min_w_N) * (1.0 - success_rate)
    # --- 修复结束 ---
    
    ccapo_file_logger.info(f"  [w_N] Online 有效成功率 (lambda_SR): {online_success_count} / {online_trajs_count} = {success_rate:.4f}")
    ccapo_file_logger.info(f"  [w_N] 动态新颖度权重 w_N = {min_w_N:.1f} + ({max_w_N-min_w_N:.1f})*(1-{success_rate:.2f}) = {w_N:.4f}")

    # --- 3. Sec 5: R_step (原始组件 - 统一计算) ---
    # (这部分保持不变, 它们都在 G_calc 级别上运行，这符合您的 GRPO 理念)
    ccapo_file_logger.info("--- [CCAPO] Phase 3-8: 统一计算 R_step (所有组件) ---")
    
    # --- ✅ [CCAPO V3] 关键修正：初始化原始分量键 ---
    for step in g_calc_steps:
        step['R_core_raw'] = 0.0
        step['R_match_raw'] = 0.0
    
    # 3.1 R_format_penalty (G_calc)
    _calculate_R_format_penalty(g_calc_steps, config)
    
    # 3.2 R_novelty_bonus (G_calc)
    _calculate_R_novelty_bonus(g_calc_steps, config)
    
    # 3.3 R_core_raw (G_calc)
    _calculate_R_step_success(g_calc_steps, g_calc_trajs, config)

    # --- 4. Sec 5: Z-Score (G_calc) ---
    # (这部分保持不变, 它们都在 G_calc 级别上运行)
    raw_novelty_all = [s.get('R_novelty_bonus', 0.0) for s in g_calc_steps]
    z_novelty_all = _standardize(raw_novelty_all)
    ccapo_file_logger.info(f"  [Z_novelty] R_novelty (G_calc) 分布: mean={np.mean(raw_novelty_all):.4f}, std={np.std(raw_novelty_all):.4f}")
    
    raw_core_success = [s.get('R_core_raw', 0.0) for s in g_calc_steps if s.get('R_core') == 1.0]
    z_core_success = _standardize(raw_core_success)
    ccapo_file_logger.info(f"  [Z_core] R_core_raw (R_core=1.0) 分布: mean={np.mean(raw_core_success):.4f}, std={np.std(raw_core_success):.4f} (来自 {len(raw_core_success)} 个步骤)")
    
    # --- 5. Sec 5.1: 组合 R_step (G_calc) ---
    core_idx = 0
    ccapo_file_logger.info("  [R_step] 组合 R_step (R_core=1.0 和 R_core=0.0):")
    for i, step in enumerate(g_calc_steps):
        step['Z_novelty'] = z_novelty_all[i]
        R_format_penalty = step.get('R_format_penalty', 0.0)

        if step.get('R_core') == 1.0:
            z_core = z_core_success[core_idx] if core_idx < len(z_core_success) else 0.0
            step['R_step'] = z_core + w_N * step['Z_novelty'] + R_format_penalty
            ccapo_file_logger.debug(f"    [Traj {step.get('traj_uid')}, t={step.get('step_index')}] (R_core=1.0): "
                                    f"R_step = Z_core({z_core:.4f}) + w_N({w_N:.4f})*Z_nov({step['Z_novelty']:.4f}) + R_format({R_format_penalty:.4f}) "
                                    f"==> {step['R_step']:.4f}")
            core_idx += 1
        elif step.get('R_core') == 0.0:
            step['R_step'] = 0.0 + R_format_penalty
            ccapo_file_logger.debug(f"    [Traj {step.get('traj_uid')}, t={step.get('step_index')}] (R_core=0.0): "
                                    f"R_step = 0.0 + R_format({R_format_penalty:.4f}) "
                                    f"==> {step['R_step']:.4f}")
    
    # --- 6. Sec 5.2: R_match_raw (G_calc) ---
    _calculate_R_step_fail(g_calc_steps, g_buffer_steps, embedding_model, config)
    
    # --- 7. Sec 5: Z-Score (G_calc) ---
    raw_match_fail = [s.get('R_match_raw', 0.0) for s in g_calc_steps if s.get('R_core') == -1.0]
    z_match_fail = _standardize(raw_match_fail)
    ccapo_file_logger.info(f"  [Z_match] R_match_raw (R_core=-1.0) 分布: mean={np.mean(raw_match_fail):.4f}, std={np.std(raw_match_fail):.4f} (来自 {len(raw_match_fail)} 个步骤)")

    # --- 8. Sec 5.2: 组合 R_step (G_calc) ---
    match_idx = 0
    ccapo_file_logger.info("  [R_step] 组合 R_step (R_core=-1.0):")
    for step in g_calc_steps:
        if step.get('R_core') == -1.0:
            z_match = z_match_fail[match_idx] if match_idx < len(z_match_fail) else 0.0
            R_format_penalty = step.get('R_format_penalty', 0.0)
            step['R_step'] = z_match + w_N * step.get('Z_novelty', 0.0) + R_format_penalty
            ccapo_file_logger.debug(f"    [Traj {step.get('traj_uid')}, t={step.get('step_index')}] (R_core=-1.0): "
                                    f"R_step = Z_match({z_match:.4f}) + w_N({w_N:.4f})*Z_nov({step.get('Z_novelty', 0.0):.4f}) + R_format({R_format_penalty:.4f}) "
                                    f"==> {step['R_step']:.4f}")
            match_idx += 1
    
    # --- [GEM 修复] 阶段 9 & 10：分离优势计算 ---
    # [GEM 修复] _calculate_A_step(g_calc_steps) 已删除
    # [GEM 修复] 旧的 Phase 10 (A_final 循环) 已删除

    # --- 9. Sec 5 & Sec 1: 局部分离的优势计算 ---
    # 我们现在在 G_online 和 G_buffer 上分别调用新的辅助函数
    
    omega = config.omega
    
    # 9.1 为 G_online 计算局部优势
    _calculate_final_advantages_for_subset(g_online_steps, omega, "Online")
    
    # 9.2 为 G_buffer 计算局部优势
    _calculate_final_advantages_for_subset(g_buffer_steps, omega, "Buffer")
    
    # --- [GEM 修复] 结束 ---
    
    ccapo_file_logger.info(f"================ [CCAPO V4] 优势计算完成 ================")
    
    # g_calc_steps 列表现在已被原地修改
    # G_online 和 G_buffer 中的步骤（它们是 g_calc_steps 的子集）
    # 现在都有了它们各自的、局部标准化的 'advantages' 键
    return g_calc_steps, success_rate