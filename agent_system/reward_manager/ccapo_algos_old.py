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

# --- [新增] 辅助函数：Softmax ---
def _compute_softmax_weights(scores: List[float], temperature: float = 0.2) -> List[float]:
    """
    计算 Softmax 权重
    temperature: 温度系数。越小(0.1)越强调头部高效轨迹，越大(1.0)越趋向平均。
    """
    if not scores: return []
    scores_arr = np.array(scores)
    # 数值稳定性处理：减去最大值
    exps = np.exp((scores_arr - np.max(scores_arr)) / temperature)
    sum_exps = np.sum(exps)
    if sum_exps < 1e-9: return [1.0/len(scores)] * len(scores) # 兜底
    weights = exps / sum_exps
    return weights.tolist()

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
    (✅ [Fix] 新增: 计算并记录 TokenCost)
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
            R_core = -1.0   # 任务失败
        elif not is_shortcut:
            R_core = 1.0    # 有效成功
        # else: 作弊成功 (IsShortcut=True), R_core = 0.0 (默认)
        
        R_tau = R_core # 默认值 (失败或作弊)
        
        # [Fix] 计算 TokenCost (用于分析和日志)
        # 定义 TokenCost 为消耗的 Token 比例，或者简单的总 Token 数
        m_token_ratio = total_tokens / config.max_tokens
        token_cost = m_token_ratio  # 这里记录比例，也可以记录 total_tokens
        
        if R_core == 1.0:
            # 只有有效成功才考虑效率
            m_steps_ratio = total_steps / config.max_steps
            # m_token_ratio 已在上面计算

            M_steps = (max(0.0, 1.0 - m_steps_ratio))**0.5
            ccapo_file_logger.info(f"[R_tau] Traj {traj_uid}: total_steps={total_steps}, m_steps_ratio={m_steps_ratio:.2f}, M_steps={M_steps:.4f} (sqrt scaled)")
            
            M_token = (max(0.0, 1.0 - m_token_ratio))**0.5
            ccapo_file_logger.info(f"[R_tau] Traj {traj_uid}: total_tokens={total_tokens}, m_token_ratio={m_token_ratio:.2f}, M_token={M_token:.4f} (sqrt scaled)")
        
            R_tau = R_core * M_steps * M_token
        # --- 新版 CCAPO 逻辑结束 ---
        
        for step in steps:
            step['R_tau'] = R_tau
            step['R_core'] = R_core # 存储 R_core 供 R_step 使用
            step['TokenCost'] = token_cost # [Fix] 显式记录 TokenCost，供 dp_actor 保存
            ccapo_file_logger.info(f"[R_tau] Traj {traj_uid} Step {step.get('step_index', 0)}: R_tau={R_tau}, R_core={R_core}")

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
    [CCAPO V3.5] 
    1. S_necessity: 使用 Softmax 加权的动作密度 (Weighted Action Density)。
    2. Q_step: 解耦，作为 explicit feedback 乘数。
    3. Q_economy: 移除。
    """
    ccapo_file_logger.info("--- [CCAPO V3.5] Phase 3.1: 计算 R_core_raw (Softmax Density) ---")
    
    # 1. 准备 Softmax 权重
    traj_raw_q_steps = []
    traj_uid_list = []
    n_success_map = {} # 缓存 n_success

    for traj_uid, steps in g_calc_trajs.items():
        # 统计成功步数 (有效步数)
        n_success = sum(1 for s in steps if s.get('action_success', False))
        n_success_map[traj_uid] = n_success
        
        if steps and steps[0].get('R_core') == 1.0:
            raw_q = max(0, 1.0 - config.alpha_step * (n_success / config.max_steps))
            traj_raw_q_steps.append(raw_q)
            traj_uid_list.append(traj_uid)
    
    temperature = getattr(config, 'softmax_temperature', 0.2)
    softmax_weights = _compute_softmax_weights(traj_raw_q_steps, temperature)
    
    traj_weight_map = {uid: w for uid, w in zip(traj_uid_list, softmax_weights)}
    
    if softmax_weights:
        ccapo_file_logger.info(f"  [Weights] Softmax(t={temperature}) 分布: Max={max(softmax_weights):.4f}, Min={min(softmax_weights):.4f}")

    # 2. 统计加权数据
    stage_stats = collections.defaultdict(lambda: {
        "total_weighted_steps": 0.0, 
        "actions": collections.defaultdict(float),
        "trajs_total": set(),
        "trajs_success": set() 
    })
    
    # 3. 遍历所有轨迹进行统计
    for traj_uid, steps in g_calc_trajs.items():
        n_success = n_success_map.get(traj_uid, 0)
        weight = traj_weight_map.get(traj_uid, 0.0) 
        
        current_successful_step_t = 0
        for step in steps:
            if step.get('action_success', False):
                current_successful_step_t += 1
                
                # 计算 Stage
                k_norm = current_successful_step_t / n_success if n_success > 0 else 0
                stage = 'Late'
                if k_norm <= 0.33: stage = 'Early'
                elif k_norm <= 0.66: stage = 'Mid'
                step['b_stage'] = stage
                
                action_type = step.get('action_type')
                if action_type:
                    if weight > 0:
                        stage_stats[stage]["actions"][action_type] += weight
                        stage_stats[stage]["total_weighted_steps"] += weight
                        
                    stage_stats[stage]["trajs_total"].add(traj_uid) 
                    if steps[0].get('R_core') == 1.0:
                        stage_stats[stage]["trajs_success"].add(traj_uid)
            else:
                step['b_stage'] = 'N/A'

    # 4. 计算 I_action
    total_trajs_count = len(g_calc_trajs)
    success_trajs_count = len(traj_uid_list)
    P_success_global = success_trajs_count / (total_trajs_count + 1e-6)
    
    # 4.1 计算 I_action 并缓存参数
    I_action_cache = {}
    Debug_params_cache = {} # [新增] 用于缓存详细参数以便回写
    
    all_keys = set()
    utility_stats = collections.defaultdict(lambda: {"success": set(), "total": set()})
    
    for traj_uid, steps in g_calc_trajs.items():
        n_success = n_success_map.get(traj_uid, 0)
        current_t = 0
        for step in steps:
            if step.get('action_success', False):
                current_t += 1
                k_norm = current_t / n_success if n_success > 0 else 0
                stage = 'Late'
                if k_norm <= 0.33: stage = 'Early'
                elif k_norm <= 0.66: stage = 'Mid'
                
                action_type = step.get('action_type')
                if action_type:
                    key = (action_type, stage)
                    all_keys.add(key)
                    utility_stats[key]["total"].add(traj_uid)
                    if steps[0].get('R_core') == 1.0:
                        utility_stats[key]["success"].add(traj_uid)

    for key in all_keys:
        action_type, stage = key
        
        # --- A. S_necessity ---
        w_count = stage_stats[stage]["actions"][action_type]
        w_total = stage_stats[stage]["total_weighted_steps"]
        S_necessity = (w_count / w_total) if w_total > 1e-9 else 0.0
        
        # --- B. S_utility ---
        n_success_with_action = len(utility_stats[key]["success"])
        n_total_with_action = len(utility_stats[key]["total"])
        
        P_succ_cond = n_success_with_action / (n_total_with_action + 1e-6)
        S_utility = P_succ_cond / (P_success_global + 1e-6)
        S_utility = min(S_utility, 5.0) # 封顶防止过大
        
        I_action_cache[key] = S_necessity * S_utility
        # [新增] 缓存参数
        Debug_params_cache[key] = (S_necessity, S_utility)

    # 5. 写回 R_core_raw 和详细参数
    for step in g_calc_steps:
        if step.get('R_core') != 1.0: continue
        
        if not step.get('action_success', False):
            step['R_core_raw'] = 0.0
            # 初始化为0，防止 key error
            step['S_necessity'] = 0.0
            step['S_utility'] = 0.0
            step['I_action'] = 0.0
            step['Q_step'] = 0.0
            continue
        
        traj_uid = step['traj_uid']
        try:
            idx = traj_uid_list.index(traj_uid)
            q_step_local = traj_raw_q_steps[idx]
        except ValueError:
            q_step_local = 0.0
            
        key = (step.get('action_type'), step.get('b_stage'))
        i_action = I_action_cache.get(key, 0.0)
        s_nec, s_util = Debug_params_cache.get(key, (0.0, 0.0))
        
        step['R_core_raw'] = i_action * q_step_local
        
        # [新增] 显式写入详细参数，供 dp_actor 回写到 json
        step['S_necessity'] = s_nec
        step['S_utility'] = s_util
        step['I_action'] = i_action
        step['Q_step'] = q_step_local

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
    (✅ [CCAPO V5] 方案A：统一优势归一化 + 禁用新颖度)
    (✅ [Fix] 修复 Viewer 数据缺失：全局初始化关键键值)
    """
    ccapo_file_logger.info(f"================ [CCAPO V5] 开始计算优势 (Unified Norm) ================")
    ccapo_file_logger.info(f"G_calc 步骤数: {len(g_calc_steps)}, G_online 步骤数: {len(g_online_steps)}")
    
    # --- ✅ [Fix] 关键修复：全局初始化所有可能的奖励组件 ---
    # 确保即使步骤失败、被跳过或 R_core!=1.0，这些键也存在（值为 0.0 或 N/A）
    # 这样 dp_actor.py 就能将其保存到 JSON，Viewer 就不会显示 "-"
    for step in g_calc_steps:
        # 核心组件
        step.setdefault('R_core_raw', 0.0)
        step.setdefault('R_match_raw', 0.0)
        step.setdefault('R_novelty_bonus', 0.0)
        step.setdefault('R_format_penalty', 0.0)
        
        # 详细参数 (Viewer 抱怨缺失的部分)
        step.setdefault('S_necessity', 0.0)
        step.setdefault('S_utility', 0.0)
        step.setdefault('I_action', 0.0)
        step.setdefault('Q_step', 0.0)
        
        # 其他
        step.setdefault('b_stage', 'N/A')
        step.setdefault('Z_novelty', 0.0)
        step.setdefault('TokenCost', 0.0)
    # --- 初始化结束 ---

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
    # [注意] 已在函数开头统一处理，此处可保留作为双重保险，或删除
    for step in g_calc_steps:
        step['R_core_raw'] = 0.0
        step['R_match_raw'] = 0.0
    
    # 3.1 R_format_penalty (G_calc)
    _calculate_R_format_penalty(g_calc_steps, config)
    
    # 3.2 R_novelty_bonus (G_calc)
    # --- ❌ [功能开关] 暂时禁用新颖度评分 (按要求注释) ---
    ccapo_file_logger.info("  [Feature Toggle] R_novelty_bonus 计算已禁用 (Value=0.0)")
    # _calculate_R_novelty_bonus(g_calc_steps, config)
    
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
    
    # --- [GEM 修复] 阶段 9: 统一优势计算 (方案A) ---
    # 废除之前的 _calculate_final_advantages_for_subset
    
    ccapo_file_logger.info(f"--- [CCAPO] Phase 9 (Unified): 全局计算优势 (G_calc = Online + Buffer) ---")

    # 9.1 计算全局 A_traj (基于 G_calc 全体)
    _calculate_A_traj(g_calc_steps)
    
    # 9.2 计算全局 A_step (基于 G_calc 全体)
    _calculate_A_step(g_calc_steps)
    
    # 9.3 组合 A_final 并计算全局统计量
    all_A_final_raw = []
    for step in g_calc_steps:
        a_traj = step.get('A_traj', 0.0)
        a_step = step.get('A_step', 0.0)
        a_final_raw = a_traj + config.omega * a_step
        step['A_final_raw'] = a_final_raw
        all_A_final_raw.append(a_final_raw)
        
    # 9.4 计算全局均值和标准差 (统一的尺子)
    if all_A_final_raw:
        mean_final = np.mean(all_A_final_raw)
        std_final = np.std(all_A_final_raw) + 1e-6
        ccapo_file_logger.info(f"  [Advantage] 全局统一归一化: Mean={mean_final:.4f}, Std={std_final:.4f} (N={len(all_A_final_raw)})")
    else:
        mean_final, std_final = 0.0, 1.0
        
    # 9.5 应用标准化并写入 advantages
    for step in g_calc_steps:
        # 这就是统一尺度下的优势！
        step['advantages'] = (step['A_final_raw'] - mean_final) / std_final

    ccapo_file_logger.info(f"================ [CCAPO V5] 优势计算完成 ================")
    
    return g_calc_steps, success_rate