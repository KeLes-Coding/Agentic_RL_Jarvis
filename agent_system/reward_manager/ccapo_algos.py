# agent_system/reward_manager/ccapo_algos.py

import torch
import numpy as np
import collections
from typing import List, Dict, Any
from sentence_transformers import util
import logging

logger = logging.getLogger(__name__)

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

# --- Sec 4: 宏观轨迹奖励 R_tau ---
def _calculate_R_tau(g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    """计算 R_tau 并将其写回每个步骤"""
    for traj_uid, steps in g_calc_trajs.items():
        if not steps: continue
        first_step = steps[0]
        
        R_success = 1.0 if first_step['traj_task_completed'] else -1.0
        
        total_steps = first_step['traj_total_steps']
        total_tokens = first_step['traj_total_tokens']
        
        P_steps = config.alpha * (total_steps / config.max_steps)
        P_token = config.gamma * (total_tokens / config.max_tokens)
        
        P_shortcut = 0.0
        if R_success > 0 and total_steps < config.min_reasonable_steps:
            P_shortcut = config.lambda_shortcut
        
        # R_format_novelty 暂时忽略，如需添加在此处
        R_tau = R_success - P_steps - P_token - P_shortcut
        
        for step in steps:
            step['R_tau'] = R_tau

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

# --- Sec 5.1: 微观步骤奖励 (成功轨迹) ---
def _calculate_R_step_success(g_calc_steps: List[Dict[str, Any]], g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    
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
    successful_trajs_in_calc = set(uid for uid, s in g_calc_trajs.items() if s[0]['traj_task_completed'])
    P_success_global = len(successful_trajs_in_calc) / (total_trajs_in_calc + 1e-6)

    # 1. 计算 b_stage 并收集统计数据
    for traj_uid, steps in g_calc_trajs.items():
        n_success = steps[0]['traj_n_success_steps']
        current_successful_step_t = 0
        for step in steps:
            if step.get('action_success', False):
                current_successful_step_t += 1
                k_norm = current_successful_step_t / n_success if n_success > 0 else 0
                
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
                    if step['traj_task_completed']:
                        stats[key]["success_trajs"].add(traj_uid)

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

    # --- Sec 5.1.3 & 5.1: 组合 R_step_success ---
    for step in g_calc_steps:
        if not step['traj_task_completed']:
            continue # 失败轨迹在 R_step_fail 中处理
        
        if not step.get('action_success', False):
            step['R_step'] = 0.0 # R_format_novelty 暂为 0
            continue
        
        # Q_step (Sec 5.1.3.1)
        n_success = step['traj_n_success_steps']
        q_step = max(0, 1.0 - config.alpha_step * (n_success / config.max_steps))
        
        q_efficiency = q_step * step['Q_economy']
        
        key = (step.get('action_type'), step.get('b_stage'))
        i_action = I_action_cache.get(key, 0.0) # 默认为 0
        
        step['R_step'] = i_action * q_efficiency # R_format_novelty 暂为 0

# --- Sec 5.2: 微观步骤奖励 (失败轨迹) ---
def _calculate_R_step_fail(g_calc_steps: List[Dict[str, Any]], g_buffer_steps: List[Dict[str, Any]], embedding_model, config):
    
    # --- 1. 预计算 STDB (G_buffer) 步骤的分数和嵌入 ---
    stdb_step_scores = collections.defaultdict(list)
    stdb_thoughts_to_embed = []
    stdb_thought_map = {} # map id -> (action, index_in_scores_list)

    idx_counter = 0
    for step in g_buffer_steps:
        # G_buffer 都是成功轨迹，且 R_step (S(a_j*)) 已经计算
        score = step.get('R_step', 0.0)
        action = step.get('parsed_action')
        thought = step.get('thought')
        
        if not action or not thought:
            continue
            
        stdb_step_scores[action].append({'score': score})
        stdb_thoughts_to_embed.append(thought)
        stdb_thought_map[idx_counter] = (action, len(stdb_step_scores[action]) - 1)
        idx_counter += 1

    if not stdb_thoughts_to_embed:
        # STDB 为空，无法匹配
        logger.warning("[CCAPO] STDB 为空或无有效 (thought, action) 对。跳过 R_step_fail 计算。")
        for step in g_calc_steps:
            if not step['traj_task_completed']:
                step['R_step'] = 0.0
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
        if step.get('is_buffer_data', False) or step['traj_task_completed']:
            continue # 已在 success 中处理,或本身是buffer数据
        
        action = step.get('parsed_action')
        thought = step.get('thought')
        
        if not action or not thought or action not in stdb_step_scores:
            step['R_step'] = 0.0 # 无法匹配
            continue
        
        fail_steps_to_embed.append(thought)
        fail_step_map[idx_counter] = step
        idx_counter += 1
        
    if not fail_steps_to_embed:
        logger.debug("[CCAPO] 没有需要匹配的失败步骤。")
        return # 没有需要匹配的失败步骤
        
    fail_embeddings = embedding_model.encode(fail_steps_to_embed, convert_to_tensor=True)

    # --- 3. 计算相似度并赋分 ---
    for i, emb_t in enumerate(fail_embeddings):
        step = fail_step_map[i]
        action = step['parsed_action']
        
        action_matches = stdb_step_scores[action]
        if not any('embedding' in m for m in action_matches):
            step['R_step'] = 0.0
            continue
            
        compare_embeddings = torch.stack([m['embedding'] for m in action_matches if 'embedding' in m]).to(emb_t.device)
        scores = torch.tensor([m['score'] for m in action_matches if 'embedding' in m], device=emb_t.device)
        
        # 计算余弦相似度
        cos_sims = util.cos_sim(emb_t, compare_embeddings)[0] # shape: (num_matches,)
        
        max_score = 0.0
        # 找到所有 > threshold 的匹配
        matches = torch.where(cos_sims > config.similarity_threshold)[0]
        if len(matches) > 0:
            # 取匹配项中的最高分
            max_score = torch.max(scores[matches]).item()
            
        step['R_step'] = config.w_match * max_score

# --- Sec 5: 微观步骤优势 A_step ---
def _calculate_A_step(g_calc_steps: List[Dict[str, Any]]):
    """计算 A_step 并将其写回每个步骤"""
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
    计算 CCAPO 奖励和优势的主函数。
    修改 g_calc_steps 列表，为其添加 'R_tau', 'A_traj', 'R_step', 'A_step', 'advantages'。
    
    Args:
        g_calc_steps (List[Dict]): G_online U G_buffer 的所有步骤
        g_online_steps (List[Dict]): G_online 的所有步骤 (用于计算 SR)
        embedding_model: 用于计算相似度的 SentenceTransformer
        config: 包含所有超参数的配置对象 (e.g., config.algorithm.ccapo)
        
    Returns:
        Tuple[List[Dict], float]:
            - g_calc_steps_with_adv: 带有 'advantages' 键的 G_calc 步骤
            - success_rate: G_online 的成功率 (lambda_SR)
    """
    logger.info(f"[CCAPO] 开始计算优势。G_calc 步骤数: {len(g_calc_steps)}, G_online 步骤数: {len(g_online_steps)}")
    
    g_calc_trajs = _group_steps_by_traj(g_calc_steps)
    g_buffer_steps = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    
    # 1. Sec 4: R_tau 和 A_traj
    _calculate_R_tau(g_calc_trajs, config)
    _calculate_A_traj(g_calc_steps)
    
    # 2. Sec 5: R_step 和 A_step
    _calculate_R_step_success(g_calc_steps, g_calc_trajs, config)
    _calculate_R_step_fail(g_calc_steps, g_buffer_steps, embedding_model, config)
    _calculate_A_step(g_calc_steps)
    
    # 3. Sec 1: A_final
    for step in g_calc_steps:
        step['advantages'] = step.get('A_traj', 0.0) + config.omega * step.get('A_step', 0.0)
    
    # 4. 计算 SR (用于 L_CCAPO)
    online_trajs = _group_steps_by_traj(g_online_steps)
    online_trajs_count = len(online_trajs)
    online_success_count = 0
    for traj_uid, steps in online_trajs.items():
        if steps and steps[0]['traj_task_completed']:
            online_success_count += 1
                
    success_rate = online_success_count / (online_trajs_count + 1e-6)
    
    logger.info(f"[CCAPO] 优势计算完成。lambda_SR (成功率): {success_rate:.4f}")
    
    return g_calc_steps, success_rate