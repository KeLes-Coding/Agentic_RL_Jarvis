# agent_system/reward_manager/ccapo_algos.py

import torch
import numpy as np
import collections
from typing import List, Dict, Any
from sentence_transformers import util
import logging
import os 

# --- 1. 标准日志器 (用于 STDOUT / 主日志) ---
logger = logging.getLogger(__name__)

# --- 2. 专用文件日志器 (用于 logger/CCAPO/ccapo_operations.log) ---
ccapo_file_logger = logging.getLogger("CCAPO_FILE")
ccapo_file_logger.setLevel(logging.INFO) 
ccapo_file_logger.propagate = False      

if not ccapo_file_logger.handlers:
    try:
        log_dir = "logger/CCAPO"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "ccapo_operations.log")
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - [CCAPO_FILE] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        ccapo_file_logger.addHandler(file_handler)
        ccapo_file_logger.info("--- CCAPO 专用文件日志器已初始化 ---")
    except Exception as e:
        logger.error(f"[CCAPO] 无法创建专用文件日志器: {e}")

# =============================================================================
# 基础辅助函数
# =============================================================================

def _compute_softmax_weights(scores: List[float], temperature: float = 0.2) -> List[float]:
    """计算 Softmax 权重"""
    if not scores: return []
    scores_arr = np.array(scores)
    exps = np.exp((scores_arr - np.max(scores_arr)) / temperature)
    sum_exps = np.sum(exps)
    if sum_exps < 1e-9: return [1.0/len(scores)] * len(scores)
    weights = exps / sum_exps
    return weights.tolist()

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

# =============================================================================
# 核心逻辑组件
# =============================================================================

def _calculate_R_tau(g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    """
    [Sec 4] 计算 R_tau, R_core, TokenCost。
    [优化]: 保持原有逻辑，但在后续 Advantage 计算中会进行保护。
    """
    for traj_uid, steps in g_calc_trajs.items():
        if not steps: continue
        first_step = steps[0]
        
        R_success = 1.0 if first_step['traj_task_completed'] else -1.0
        total_steps = first_step['traj_total_steps']
        total_tokens = first_step['traj_total_tokens']
        
        is_shortcut = (R_success > 0 and total_steps < config.min_reasonable_steps)
        
        R_core = 0.0
        if R_success <= 0:
            R_core = -1.0
        elif not is_shortcut:
            R_core = 1.0
        
        # 原始 R_tau 计算逻辑保持不变，依靠 A_traj 进行修正
        R_tau = R_core 
        
        m_token_ratio = total_tokens / config.max_tokens
        token_cost = m_token_ratio 
        
        if R_core == 1.0:
            m_steps_ratio = total_steps / config.max_steps
            M_steps = (max(0.0, 1.0 - m_steps_ratio))**0.5
            M_token = (max(0.0, 1.0 - m_token_ratio))**0.5
            R_tau = R_core * M_steps * M_token
            
            ccapo_file_logger.debug(f"[R_tau] Traj {traj_uid}: R_tau={R_tau:.4f} (Steps={M_steps:.2f}, Tokens={M_token:.2f})")
        
        for step in steps:
            step['R_tau'] = R_tau
            step['R_core'] = R_core
            step['TokenCost'] = token_cost

def _calculate_A_traj(g_calc_steps: List[Dict[str, Any]]):
    """
    [Sec 4.2 Modified] 计算 A_traj (Standardize R_tau)
    [关键修复]: 引入符号保护 (Sign Protection) 和 鲁棒标准化，
    防止 Group_size 较小或全对/全错时的数值崩溃。
    """
    if not g_calc_steps: return
    all_R_tau = [step.get('R_tau', 0.0) for step in g_calc_steps]
    
    mean_R = np.mean(all_R_tau)
    std_R = np.std(all_R_tau)
    
    # [Fix 1] 增大 epsilon，防止 Group_size=4 时微小差异被过度放大
    # 对于 -1/1 的 reward 分布，std 通常在 1.0 左右，给 0.05 的缓冲是安全的
    safe_std = std_R + 0.05 
    
    # 预计算标准化的 Advantage
    raw_advantages = [(v - mean_R) / safe_std for v in all_R_tau]
    
    final_advantages = []
    for r_val, adv_val in zip(all_R_tau, raw_advantages):
        # [Fix 2] 符号保护 (Sign Protection)
        # 如果原 Reward 是正的(成功)，Advantage 绝不能是负的 (防止冤杀好样本)
        if r_val > 0.0 and adv_val < 0.0:
            adv_val = 0.05 # 给予一个微小的正向激励，而不是惩罚
        
        # 如果原 Reward 是负的(失败)，Advantage 绝不能是正的 (防止奖励幻觉)
        elif r_val < 0.0 and adv_val > 0.0:
            adv_val = -0.05 # 给予一个微小的负向惩罚
            
        final_advantages.append(adv_val)

    # 回写
    for step, adv in zip(g_calc_steps, final_advantages):
        step['A_traj'] = adv

def _calculate_R_format_penalty(g_calc_steps: List[Dict[str, Any]], config):
    """计算格式惩罚"""
    for step in g_calc_steps:
        action_status = step.get('action_status', '')
        penalty = 0.0
        if action_status.startswith('FORMAT_ERROR'):
            penalty = config.penalty_format_error
        elif action_status.startswith('FAILURE'):
            penalty = config.penalty_failure
        step['R_format_penalty'] = penalty

def _calculate_R_step_success(g_calc_steps: List[Dict[str, Any]], g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    """
    [Sec 5.1 Revised] 计算 R_core_raw
    [Feature]: 支持 use_fine_grained_action 开关，切换动作粒度。
    [Fix]: 使用 Trajectory Coverage 公式计算 S_necessity，以适应细粒度带来的稀疏性。
    """
    # 0. 获取粒度开关
    use_fine_grained = getattr(config, 'use_fine_grained_action', False)

    # 1. 准备 Softmax 权重
    traj_raw_q_steps = []
    traj_uid_list = []
    n_success_map = {}

    for traj_uid, steps in g_calc_trajs.items():
        n_success = sum(1 for s in steps if s.get('action_success', False))
        n_success_map[traj_uid] = n_success
        
        if steps and steps[0].get('R_core') == 1.0:
            raw_q = max(0, 1.0 - config.alpha_step * (n_success / config.max_steps))
            traj_raw_q_steps.append(raw_q)
            traj_uid_list.append(traj_uid)
    
    temperature = getattr(config, 'softmax_temperature', 0.2)
    softmax_weights = _compute_softmax_weights(traj_raw_q_steps, temperature)
    traj_weight_map = {uid: w for uid, w in zip(traj_uid_list, softmax_weights)}

    # =========================================================================
    # [New Logic] S_necessity Calculation (Trajectory Coverage + Granularity)
    # =========================================================================
    
    # 2.1 统计成功轨迹的总权重 (分母)
    total_success_weight_sum = sum(softmax_weights) + 1e-9
    
    # 2.2 统计分子 & 辅助结构
    necessity_numerator = collections.defaultdict(float)
    utility_stats = collections.defaultdict(lambda: {"success": set(), "total": set()})
    all_keys = set()

    for traj_uid, steps in g_calc_trajs.items():
        weight = traj_weight_map.get(traj_uid, 0.0)
        is_success_traj = (steps[0].get('R_core') == 1.0)
        
        n_success = n_success_map.get(traj_uid, 0)
        
        # 记录当前轨迹在这个 Stage 是否已经贡献过该 Action (防止同一轨迹内重复统计)
        seen_in_this_traj = set() 
        
        current_t = 0
        for step in steps:
            if step.get('action_success', False):
                current_t += 1
                k_norm = current_t / n_success if n_success > 0 else 0
                stage = 'Late' if k_norm > 0.66 else ('Mid' if k_norm > 0.33 else 'Early')
                step['b_stage'] = stage
                
                action_type = step.get('action_type')
                if action_type:
                    # --- [Granularity Logic] 核心切换逻辑 ---
                    if use_fine_grained:
                        # 使用 parsed_action (包含参数) 构造 Key
                        # 注意：需要转为字符串以确保可哈希
                        raw_act = step.get('parsed_action')
                        if raw_act is not None:
                            # 格式示例: "click::{'uid': '123'}"
                            act_identifier = f"{action_type}::{str(raw_act)}"
                        else:
                            act_identifier = action_type
                    else:
                        # 仅使用 action_type (模糊匹配)
                        act_identifier = action_type
                    
                    key = (act_identifier, stage)
                    all_keys.add(key)
                    
                    # Utility 统计
                    utility_stats[key]["total"].add(traj_uid)
                    if is_success_traj:
                        utility_stats[key]["success"].add(traj_uid)
                        
                        # Necessity 统计 (Trajectory Coverage)
                        if weight > 0 and key not in seen_in_this_traj:
                            necessity_numerator[key] += weight
                            seen_in_this_traj.add(key)
            else:
                step['b_stage'] = 'N/A'

    # 3. 计算最终指标
    total_trajs = len(g_calc_trajs)
    success_trajs = len(traj_uid_list)
    P_succ_global = max(success_trajs / (total_trajs + 1e-6), 0.01)
    
    I_action_cache = {}
    Debug_params_cache = {}
    
    for key in all_keys:
        # S_necessity
        w_numerator = necessity_numerator.get(key, 0.0)
        S_necessity = w_numerator / total_success_weight_sum
        
        # S_utility
        n_succ_act = len(utility_stats[key]["success"])
        n_total_act = len(utility_stats[key]["total"])
        P_succ_cond = n_succ_act / (n_total_act + 1e-6)
        
        S_utility = min(P_succ_cond / P_succ_global, 3.0)
        
        I_action_cache[key] = S_necessity * S_utility
        Debug_params_cache[key] = (S_necessity, S_utility)

    # 4. 写回
    for step in g_calc_steps:
        if step.get('R_core') != 1.0: continue
        if not step.get('action_success', False): continue
        
        traj_uid = step['traj_uid']
        q_step_local = 0.0
        if traj_uid in traj_uid_list:
             idx = traj_uid_list.index(traj_uid)
             q_step_local = traj_raw_q_steps[idx]
        
        # 重复 Granularity Key 构建逻辑以匹配 Cache
        action_type = step.get('action_type')
        stage = step.get('b_stage')
        
        if use_fine_grained:
            raw_act = step.get('parsed_action')
            if raw_act is not None:
                act_identifier = f"{action_type}::{str(raw_act)}"
            else:
                act_identifier = action_type
        else:
            act_identifier = action_type

        key = (act_identifier, stage)
        
        i_action = I_action_cache.get(key, 0.0)
        s_nec, s_util = Debug_params_cache.get(key, (0.0, 0.0))
        
        step['R_core_raw'] = i_action * q_step_local
        step['S_necessity'] = s_nec
        step['S_utility'] = s_util
        step['I_action'] = i_action
        step['Q_step'] = q_step_local

def _calculate_R_step_fail(g_calc_steps: List[Dict[str, Any]], g_buffer_steps: List[Dict[str, Any]], embedding_model, config):
    """
    [Sec 5.2] 计算 R_match_raw (Embedding Matching)
    """
    # 1. 索引 STDB
    stdb_step_scores = collections.defaultdict(list)
    stdb_thoughts = []
    stdb_map = {} 
    idx = 0
    for step in g_buffer_steps:
        if step.get('R_core') != 1.0: continue
        action = step.get('parsed_action')
        if not action: continue
        stdb_step_scores[action].append({'score': step.get('R_step', 0.0)})
        stdb_thoughts.append(step.get('thought', ''))
        stdb_map[idx] = (action, len(stdb_step_scores[action]) - 1)
        idx += 1

    if not stdb_thoughts: return
    
    # 批量编码 STDB
    stdb_embeddings = embedding_model.encode(stdb_thoughts, convert_to_tensor=True)
    for i, emb in enumerate(stdb_embeddings):
        action, list_idx = stdb_map[i]
        stdb_step_scores[action][list_idx]['embedding'] = emb

    # 2. 准备失败步骤
    fail_steps = []
    fail_indices = [] # step index in g_calc_steps
    
    for i, step in enumerate(g_calc_steps):
        if step.get('R_core') != -1.0: continue
        action = step.get('parsed_action')
        if not action or action not in stdb_step_scores: continue
        fail_steps.append(step.get('thought', ''))
        fail_indices.append(i)
        
    if not fail_steps: return
    
    # 批量编码失败步骤
    fail_embeddings = embedding_model.encode(fail_steps, convert_to_tensor=True)

    # 3. 匹配
    for i, emb_t in enumerate(fail_embeddings):
        step_idx = fail_indices[i]
        step = g_calc_steps[step_idx]
        action = step['parsed_action']
        
        matches = stdb_step_scores[action]
        valid_m = [m for m in matches if 'embedding' in m]
        if not valid_m: continue
            
        comp_embs = torch.stack([m['embedding'] for m in valid_m]).to(emb_t.device)
        comp_scores = torch.tensor([m['score'] for m in valid_m], device=emb_t.device)
        
        cos_sims = util.cos_sim(emb_t, comp_embs)[0]
        found = torch.where(cos_sims > config.similarity_threshold)[0]
        
        if len(found) > 0:
            # 取匹配中最高的 R_step
            max_score = torch.max(comp_scores[found]).item()
            step['R_match_raw'] = max_score

def _calculate_A_step(g_calc_steps: List[Dict[str, Any]]):
    """计算 A_step (Standardize R_step)"""
    if not g_calc_steps: return
    all_R_step = [step.get('R_step', 0.0) for step in g_calc_steps]
    mean_s = np.mean(all_R_step)
    std_s = np.std(all_R_step) + 1e-6
    for step in g_calc_steps:
        step['A_step'] = (step.get('R_step', 0.0) - mean_s) / std_s

def _calculate_separated_advantages(steps: List[Dict[str, Any]], omega: float):
    """
    [关键] 分离计算 Advantage
    """
    if not steps: return
    
    # 1. Local A_traj
    _calculate_A_traj(steps)
    
    # 2. Local A_step
    _calculate_A_step(steps)
    
    # 3. Combine & Normalize Advantage Locally
    raw_advs = []
    for s in steps:
        raw = s['A_traj'] + omega * s['A_step']
        s['A_final_raw'] = raw
        raw_advs.append(raw)
        
    mean_adv = np.mean(raw_advs)
    std_adv = np.std(raw_advs) + 1e-6
    
    for s in steps:
        s['advantages'] = (s['A_final_raw'] - mean_adv) / std_adv

# =============================================================================
# 主入口
# =============================================================================

def compute_ccapo_advantages(g_calc_steps: List[Dict[str, Any]], g_online_steps: List[Dict[str, Any]], embedding_model, config):
    """
    CCAPO V6 (Separated Norm + Robust Import)
    """
    ccapo_file_logger.info("=== [CCAPO V6] Start Calculation ===")
    
    # 0. 初始化
    keys = ['R_core_raw', 'R_match_raw', 'R_format_penalty', 'S_necessity', 'S_utility', 'I_action', 'Q_step', 'Z_novelty', 'Z_core', 'Z_match', 'TokenCost', 'b_stage', 'R_novelty_bonus']
    for step in g_calc_steps:
        for k in keys: step.setdefault(k, 0.0 if k != 'b_stage' else 'N/A')

    g_calc_trajs = _group_steps_by_traj(g_calc_steps)
    g_buffer_steps = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    
    # 1. R_tau & R_core
    _calculate_R_tau(g_calc_trajs, config)
    
    # 2. w_N (Dynamic Weight)
    online_trajs = _group_steps_by_traj(g_online_steps)
    online_succ = sum(1 for steps in online_trajs.values() if steps and steps[0].get('R_core') == 1.0)
    sr = online_succ / (len(online_trajs) + 1e-6)
    
    max_w = config.get("max_w_N", 0.8)
    min_w = config.get("min_w_N", 0.2)
    w_N = min_w + (max_w - min_w) * (1.0 - sr)
    
    ccapo_file_logger.info(f"SR: {sr:.4f}, w_N: {w_N:.4f}")

    # 3. R_step Components
    _calculate_R_format_penalty(g_calc_steps, config)
    # _calculate_R_novelty_bonus(g_calc_steps, config) # Disabled
    _calculate_R_step_success(g_calc_steps, g_calc_trajs, config)
    _calculate_R_step_fail(g_calc_steps, g_buffer_steps, embedding_model, config)
    
    # 4. Combine R_step (Global Z-Score for components)
    # Note: Components are Z-scored globally across G_calc to align scales
    raw_core = [s['R_core_raw'] for s in g_calc_steps if s.get('R_core') == 1.0]
    z_core = _standardize(raw_core)
    success_steps = [s for s in g_calc_steps if s.get('R_core') == 1.0]
    for s, z in zip(success_steps, z_core): s['Z_core'] = z
    
    raw_match = [s['R_match_raw'] for s in g_calc_steps if s.get('R_core') == -1.0]
    z_match = _standardize(raw_match)
    fail_steps = [s for s in g_calc_steps if s.get('R_core') == -1.0]
    for s, z in zip(fail_steps, z_match): s['Z_match'] = z
    
    for step in g_calc_steps:
        rc = step.get('R_core')
        pen = step['R_format_penalty']
        if rc == 1.0:
            step['R_step'] = step.get('Z_core', 0.0) + w_N * step.get('Z_novelty', 0.0) + pen
        elif rc == -1.0:
            step['R_step'] = step.get('Z_match', 0.0) + w_N * step.get('Z_novelty', 0.0) + pen
        else:
            step['R_step'] = pen

    # 5. Separated Advantage Calculation
    online_subset = [s for s in g_calc_steps if not s.get('is_buffer_data', False)]
    buffer_subset = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    
    _calculate_separated_advantages(online_subset, config.omega)
    _calculate_separated_advantages(buffer_subset, config.omega)
    
    ccapo_file_logger.info("=== [CCAPO V6] Done ===")
    return g_calc_steps, sr