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
    """
    # --- Sec 5.1.3.2: Q_economy ---
    all_token_costs = []
    for step in g_calc_steps:
        usage = step.get('step_token_usage', {})
        cost = (config.w_out * usage.get('completion_tokens', 0) +
                config.w_in * usage.get('prompt_tokens', 0))
        step['TokenCost'] = cost
        all_token_costs.append(cost)
    
    avg_token_cost = np.mean(all_token_costs) + 1e-6
    for step in g_calc_steps:
        step['Q_economy'] = 1.0 - np.tanh((step['TokenCost'] - avg_token_cost) / avg_token_cost)

    # --- Sec 5.1.2: I_action (S_necessity, S_utility) ---
    stats = collections.defaultdict(lambda: {"total_steps": 0, "success_trajs": set(), "total_trajs": set()})
    total_trajs_in_calc = len(g_calc_trajs)
    # --- ✅ [CCAPO V2] 修改: 根据 R_core == 1.0 判断有效成功 ---
    successful_trajs_in_calc = set(uid for uid, s in g_calc_trajs.items() if s[0].get('R_core') == 1.0)
    P_success_global = len(successful_trajs_in_calc) / (total_trajs_in_calc + 1e-6)

    # 1. 计算 b_stage 并收集统计数据
    for traj_uid, steps in g_calc_trajs.items():
        n_success = steps[0]['traj_n_success_steps']
        current_successful_step_t = 0
        for step in steps:
            if step.get('action_success', False):
                current_successful_step_t += 1
                k_norm = current_successful_step_t / n_success if n_success > 0 else 0
                ccapo_file_logger.info(f"[R_success] Traj {traj_uid} Step {step.get('step_index', 0)}: k_norm={k_norm}, n_success={n_success}, current_successful_step_t={current_successful_step_t}") # <-- ✅ [日志] 替换
                
                stage = 'Late'
                if k_norm <= 0.33: stage = 'Early'
                elif k_norm <= 0.66: stage = 'Mid'
                step['b_stage'] = stage
                
                # 收集统计
                action_type = step.get('action_type')
                if action_type:
                    key = (action_type, stage)
                    stats[key]["total_steps"] += 1
                    stats[key]["total_trajs"].add(traj_uid)
                    # --- ✅ [CCAPO V2] 修改: 根据 R_core == 1.0 判断有效成功 ---
                    if step.get('R_core') == 1.0:
                        stats[key]["success_trajs"].add(traj_uid)
            else:
                step['b_stage'] = 'N/A'

    # 2. 计算 I_action
    I_action_cache = {}
    for key, data in stats.items():
        # S_necessity (Sec 5.1.2.2)
        count_success_trajs_with_action = len(data["success_trajs"])
        S_necessity = count_success_trajs_with_action / (len(successful_trajs_in_calc) + 1e-6)
        
        # S_utility (Sec 5.1.2.3)
        P_success_given_action = len(data["success_trajs"]) / (len(data["total_trajs"]) + 1e-6)
        S_utility = P_success_given_action / (P_success_global + 1e-6)
        
        I_action_cache[key] = S_necessity * S_utility
        ccapo_file_logger.info(f"[R_success] Action '{key[0]}' Stage '{key[1]}': S_necessity={S_necessity}, S_utility={S_utility}, I_action={I_action_cache[key]}") # <-- ✅ [日志] 替换

    # --- Sec 5.1.3 & 5.1: 仅计算 R_core_raw ---
    for step in g_calc_steps:
        # --- ✅ [CCAPO V2] 修改: 只处理 R_core == 1.0 的步骤 ---
        if step.get('R_core') != 1.0:
            continue 
        
        if not step.get('action_success', False):
            step['R_core_raw'] = 0.0 # 动作失败，核心奖励为0
            continue
        
        # Q_step (Sec 5.1.3.1)
        n_success = step['traj_n_success_steps']
        q_step = max(0, 1.0 - config.alpha_step * (n_success / config.max_steps))
        
        q_efficiency = q_step * step['Q_economy']
        
        key = (step.get('action_type'), step.get('b_stage'))
        i_action = I_action_cache.get(key, 0.0) 
        
        # --- ✅ [CCAPO V2] 修改: 只存储 R_core_raw ---
        step['R_core_raw'] = i_action * q_efficiency
        ccapo_file_logger.info(f"[R_success] Step {step.get('step_index', 0)}: Q_step={q_step}, Q_economy={step['Q_economy']}, I_action={i_action}, R_core_raw={step['R_core_raw']}") # <-- ✅ [日志] 替换
        # (移除 R_step 的计算)

# --- Sec 5.2: 微观步骤奖励 (失败轨迹) ---
def _calculate_R_step_fail(g_calc_steps: List[Dict[str, Any]], g_buffer_steps: List[Dict[str, Any]], embedding_model, config):
    """
    计算 R_step 的“匹配”原始分量 (R_match_raw)
    (✅ [CCAPO V2] 修改：使用 R_step (S(a_j*)) 作为匹配源，不使用 w_match, 仅存储 R_match_raw)
    """
    
    # --- 1. 预计算 STDB (G_buffer) 步骤的分数和嵌入 ---
    stdb_step_scores = collections.defaultdict(list)
    stdb_thoughts_to_embed = []
    stdb_thought_map = {} # map id -> (action, index_in_scores_list)

    idx_counter = 0
    for step in g_buffer_steps:
        # G_buffer 都是成功轨迹 (R_core == 1.0)
        # --- ✅ [CCAPO V2] 修改: S(a_j*) = R_step_success(a_j*)
        # 此时 R_step 已由 R_step_success (Z_core + w_N*Z_novelty) 填充
        score = step.get('R_step', 0.0)
        action = step.get('parsed_action')
        thought = step.get('thought')
        
        if not action or not thought or step.get('R_core') != 1.0:
            # R_step_fail 只应匹配有效的成功步骤
            # (R_step 在 R_core != 1.0 时可能未定义或为 0)
            continue
            
        stdb_step_scores[action].append({'score': score})
        stdb_thoughts_to_embed.append(thought)
        stdb_thought_map[idx_counter] = (action, len(stdb_step_scores[action]) - 1)
        idx_counter += 1

    if not stdb_thoughts_to_embed:
        ccapo_file_logger.warning("[R_fail] STDB (G_buffer) 为空或无有效 (thought, action) 对。跳过 R_match_raw 计算。") # <-- ✅ [日志] 替换
        return
        
    stdb_embeddings = embedding_model.encode(stdb_thoughts_to_embed, convert_to_tensor=True)
    
    # 将嵌入存回字典
    for i, emb in enumerate(stdb_embeddings):
        action, list_idx = stdb_thought_map[i]
        stdb_step_scores[action][list_idx]['embedding'] = emb

    # --- 2. 匹配失败步骤 ---
    fail_steps_to_embed = []
    fail_step_map = {} # map id -> step_dict
    idx_counter = 0
    
    for step in g_calc_steps:
        # --- ✅ [CCAPO V2] 修改: 只处理 R_core == -1.0 的步骤 ---
        if step.get('R_core') != -1.0:
            continue
        
        action = step.get('parsed_action')
        thought = step.get('thought')
        
        if not action or not thought or action not in stdb_step_scores:
            continue
        
        fail_steps_to_embed.append(thought)
        fail_step_map[idx_counter] = step
        idx_counter += 1
        
    if not fail_steps_to_embed:
        ccapo_file_logger.info("[R_fail] 没有需要匹配的失败步骤。") # <-- ✅ [日志] 替换
        return
        
    fail_embeddings = embedding_model.encode(fail_steps_to_embed, convert_to_tensor=True)

    # --- 3. 计算相似度并赋分 ---
    for i, emb_t in enumerate(fail_embeddings):
        step = fail_step_map[i]
        action = step['parsed_action']
        
        action_matches = stdb_step_scores[action]
        if not any('embedding' in m for m in action_matches):
            continue
            
        compare_embeddings = torch.stack([m['embedding'] for m in action_matches if 'embedding' in m]).to(emb_t.device)
        scores = torch.tensor([m['score'] for m in action_matches if 'embedding' in m], device=emb_t.device)
        
        cos_sims = util.cos_sim(emb_t, compare_embeddings)[0]
        
        max_score = 0.0
        matches = torch.where(cos_sims > config.similarity_threshold)[0]
        if len(matches) > 0:
            max_score = torch.max(scores[matches]).item()
            
        # --- ✅ [CCAPO V2] 修改: 存储 R_match_raw, 移除 w_match ---
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
    
    for step in g_calc_steps:
        step['A_step'] = (step.get('R_step', mean_R_step) - mean_R_step) / std_R_step

# --- 主函数 ---
def compute_ccapo_advantages(g_calc_steps: List[Dict[str, Any]], g_online_steps: List[Dict[str, Any]], embedding_model, config):
    """
    计算 CCAPO 奖励和优势的主函数。 (✅ [CCAPO V3] 修改: 解耦格式惩罚)
    修改 g_calc_steps 列表，为其添加 'R_tau', 'A_traj', 'R_step', 'A_step', 'advantages'。
    """
    ccapo_file_logger.info(f"--- [CCAPO V3] 开始计算优势 ---") # <-- ✅ [日志] 替换
    ccapo_file_logger.info(f"G_calc 步骤数: {len(g_calc_steps)}, G_online 步骤数: {len(g_online_steps)}")
    
    g_calc_trajs = _group_steps_by_traj(g_calc_steps)
    g_buffer_steps = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    
    # --- 1. Sec 4: R_tau, R_core 和 A_traj ---
    _calculate_R_tau(g_calc_trajs, config)
    _calculate_A_traj(g_calc_steps)
    
    # --- 2. Sec 7: 计算 SR (lambda_SR) 和 w_N ---
    online_trajs = _group_steps_by_traj(g_online_steps)
    online_trajs_count = len(online_trajs)
    online_success_count = 0
    for traj_uid, steps in online_trajs.items():
        # --- ✅ [CCAPO V2] 修改: 根据 R_core == 1.0 判断有效成功 ---
        if steps and steps[0].get('R_core') == 1.0:
            online_success_count += 1
            
    success_rate = online_success_count / (online_trajs_count + 1e-6)
    w_N = 1.0 - success_rate # 动态 w_N
    
    ccapo_file_logger.info(f"[CCAPO V3] lambda_SR (有效成功率): {success_rate:.4f}, w_N: {w_N:.4f}") # <-- ✅ [日志] 替换

    # --- ✅ [CCAPO V3] 关键修正：初始化原始分量键 ---
    # 确保 R_core_raw 和 R_match_raw 在 *所有* 步骤中都存在，以防止 collate_fn 错误
    for step in g_calc_steps:
        step['R_core_raw'] = 0.0
        step['R_match_raw'] = 0.0
    # --- 修正结束 ---

    # --- 3. Sec 5: R_step (原始组件) ---
    
    # 3.1 R_format_penalty (所有步骤) - ✅ [CCAPO V3] 变更
    _calculate_R_format_penalty(g_calc_steps, config)
    
    # 3.2 R_novelty_bonus (仅成功动作) - ✅ [CCAPO V3] 变更
    _calculate_R_novelty_bonus(g_calc_steps, config)
    
    # 3.3 R_core_raw (仅 R_core == 1.0 步骤)
    # (此函数现在将 *覆盖* 上面设置的 0.0)
    _calculate_R_step_success(g_calc_steps, g_calc_trajs, config)

    # --- 4. Sec 5: Z-Score 标准化 (除 R_format_penalty 和 R_match_raw) ---
    
    # 4.1 Z_novelty (所有步骤) - ✅ [CCAPO V3] 变更
    raw_novelty_all = [s.get('R_novelty_bonus', 0.0) for s in g_calc_steps]
    z_novelty_all = _standardize(raw_novelty_all)
    
    # 4.2 Z_core (仅 R_core == 1.0 步骤)
    raw_core_success = [s.get('R_core_raw', 0.0) for s in g_calc_steps if s.get('R_core') == 1.0]
    z_core_success = _standardize(raw_core_success)
    
    # --- 5. Sec 5.1: 组合 R_step (成功和捷径轨迹) ---
    # (必须在 R_step_fail 之前计算, 因为 R_step_fail 依赖 G_buffer 中的 R_step)
    core_idx = 0
    for i, step in enumerate(g_calc_steps):
        step['Z_novelty'] = z_novelty_all[i] # 存下 Z_novelty 供 R_step_fail 使用
        
        # ✅ [CCAPO V3] 提取 R_format_penalty (硬惩罚，不标准化)
        R_format_penalty = step.get('R_format_penalty', 0.0)

        if step.get('R_core') == 1.0:
            # R_step_success = Z_core + w_N * Z_novelty + R_format_penalty
            z_core = z_core_success[core_idx]
            step['R_step'] = z_core + w_N * step['Z_novelty'] + R_format_penalty
            core_idx += 1
        elif step.get('R_core') == 0.0:
            # R_step (shortcut) = 0 + R_format_penalty
            step['R_step'] = 0.0 + R_format_penalty
    
    # --- 6. Sec 5.2: R_match_raw (仅 R_core == -1.0 步骤) ---
    # (这必须在 G_buffer 步骤的 R_step 填充后运行)
    # (G_buffer 中的 R_step 此时已包含 R_format_penalty, 这是OK的,
    # 因为惩罚总是负的, 会使匹配到的分数更低, 这符合逻辑。)
    # (此函数现在将 *覆盖* 上面设置的 0.0)
    _calculate_R_step_fail(g_calc_steps, g_buffer_steps, embedding_model, config)
    
    # --- 7. Sec 5: Z-Score 标准化 (R_match_raw) ---
    raw_match_fail = [s.get('R_match_raw', 0.0) for s in g_calc_steps if s.get('R_core') == -1.0]
    z_match_fail = _standardize(raw_match_fail)

    # --- 8. Sec 5.2: 组合 R_step (失败轨迹) ---
    match_idx = 0
    for step in g_calc_steps:
        if step.get('R_core') == -1.0:
            # R_step_fail = Z_match + w_N * Z_novelty + R_format_penalty
            z_match = z_match_fail[match_idx]
            # ✅ [CCAPO V3] 提取 R_format_penalty (硬惩罚，不标准化)
            R_format_penalty = step.get('R_format_penalty', 0.0)
            step['R_step'] = z_match + w_N * step.get('Z_novelty', 0.0) + R_format_penalty
            match_idx += 1
    
    # --- 9. Sec 5: A_step (标准化最终的 R_step) ---
    _calculate_A_step(g_calc_steps)
    
    # --- 10. Sec 1: A_final ---
    for step in g_calc_steps:
        step['advantages'] = step.get('A_traj', 0.0) + config.omega * step.get('A_step', 0.0)
        # 移除了调试 print 语句
    
    ccapo_file_logger.info(f"[CCAPO V3] 优势计算完成 (已解耦格式惩罚)。") # <-- ✅ [日志] 替换
    
    return g_calc_steps, success_rate