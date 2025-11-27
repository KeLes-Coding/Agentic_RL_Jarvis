# agent_system/reward_manager/ccapo_algos.py

import torch
import numpy as np
import collections
from typing import List, Dict, Any
from sentence_transformers import util
import logging
import os 

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

# =============================================================================
# 核心逻辑组件
# =============================================================================

def _calculate_R_tau(g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    """
    [Sec 4] 计算 R_tau, R_core
    """
    for traj_uid, steps in g_calc_trajs.items():
        if not steps: continue
        first_step = steps[0]
        
        # 兼容处理：Buffer 数据可能没有 task_completed 字段，默认为 True
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

def _calculate_A_traj(g_calc_steps: List[Dict[str, Any]]):
    """
    [Sec 4.2] 计算 A_traj (恢复原始逻辑，移除 SR 参数)
    """
    if not g_calc_steps: return
    all_R_tau = [step.get('R_tau', 0.0) for step in g_calc_steps]
    
    mean_R = np.mean(all_R_tau)
    std_R = np.std(all_R_tau)
    safe_std = std_R + 0.05 
    
    raw_advantages = [(v - mean_R) / safe_std for v in all_R_tau]
    
    final_advantages = []
    for r_val, adv_val in zip(all_R_tau, raw_advantages):
        # 符号保护: 成功但这步很差 -> 软着陆
        if r_val > 0.0 and adv_val < 0.0:
            adv_val = adv_val * 0.2 
        # 符号保护: 失败但这步相对好 -> 拦截
        elif r_val < 0.0 and adv_val > 0.0:
            adv_val = -0.05
        final_advantages.append(adv_val)

    for step, adv in zip(g_calc_steps, final_advantages):
        step['A_traj'] = adv

def _calculate_R_format_penalty(g_calc_steps: List[Dict[str, Any]], config):
    """计算格式惩罚"""
    for step in g_calc_steps:
        action_status = step.get('action_status', '')
        is_success = step.get('action_success', False) 
        penalty = 0.0
        
        if is_success is True or action_status == 'true' or action_status == '':
            penalty = 0.0
        elif (action_status.startswith('FORMAT_ERROR') or 
              action_status.startswith('UNKNOWN_ACTION') or 
              action_status.startswith('ARGUMENT_ERROR')):
            penalty = config.penalty_format_error
        elif (action_status.startswith('FAILURE') or 
              action_status.startswith('EXECUTION_ERROR')):
            penalty = config.penalty_failure
        else:
            penalty = config.penalty_failure
        step['R_format_penalty'] = penalty

def _calculate_R_step_success(g_calc_steps: List[Dict[str, Any]], g_calc_trajs: Dict[str, List[Dict[str, Any]]], config):
    """
    [Sec 5.1 Revised] LCS 序列对齐逻辑
    """
    use_fine_grained = getattr(config, 'use_fine_grained_action', False)
    beta = getattr(config, 'redundancy_penalty', 0.5)

    # 1. 准备数据
    success_traj_info = {}
    for traj_uid, steps in g_calc_trajs.items():
        # 只处理 R_core=1.0 的轨迹
        if not steps or steps[0].get('R_core') != 1.0: continue
        
        valid_steps = []
        act_identifiers = []
        for step in steps:
            if not step.get('action_success', False): continue
            
            valid_steps.append(step)
            action_type = step.get('action_type')
            
            # 构造粒度标识
            if use_fine_grained:
                if action_type == 'finish' or action_type == 'input_text':
                    act_identifier = action_type
                else:
                    raw_act = step.get('parsed_action')
                    act_identifier = f"{action_type}::{str(raw_act)}" if raw_act else action_type
            else:
                act_identifier = action_type
            act_identifiers.append(act_identifier)
            
        if valid_steps:
            success_traj_info[traj_uid] = {'steps': valid_steps, 'act_ids': act_identifiers, 'n_success': len(valid_steps)}

    if not success_traj_info: return

    # 2. 锚点选择 (最短的成功轨迹)
    sorted_trajs = sorted(success_traj_info.items(), key=lambda x: x[1]['n_success'])
    anchor_uid, anchor_data = sorted_trajs[0]
    anchor_seq = anchor_data['act_ids']
    
    # 3. 序列对齐与赋值
    for traj_uid, info in success_traj_info.items():
        current_seq = info['act_ids']
        current_steps = info['steps']
        n_success = info['n_success']
        
        # 效率乘数
        m_steps_ratio = n_success / config.max_steps
        q_step = max(0.0, 1.0 - config.alpha_step * m_steps_ratio)
        
        matched_indices = set()
        if traj_uid == anchor_uid:
            matched_indices = set(range(len(current_seq)))
        else:
            # LCS DP
            m, n = len(anchor_seq), len(current_seq)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if anchor_seq[i-1] == current_seq[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            # Backtrack
            i, j = m, n
            while i > 0 and j > 0:
                if anchor_seq[i-1] == current_seq[j-1]:
                    matched_indices.add(j-1)
                    i -= 1; j -= 1
                elif dp[i-1][j] > dp[i][j-1]: i -= 1
                else: j -= 1

        # 回写
        for idx, step in enumerate(current_steps):
            # 强制人工标记为 Match
            is_match = (idx in matched_indices) or step.get('is_human_marked', False)
            
            i_action = 1.0 if is_match else -beta
            # Match 乘效率，Mismatch 直接罚
            r_core_raw = i_action * q_step if is_match else i_action
            
            step['R_core_raw'] = r_core_raw
            step['I_action'] = i_action
            step['Q_step'] = q_step
            
            # 兼容字段
            step['S_necessity'] = 1.0 if is_match else 0.0
            step['S_utility'] = 1.0 if is_match else 0.0

def _calculate_R_step_fail(g_calc_steps: List[Dict[str, Any]], g_buffer_steps: List[Dict[str, Any]], embedding_model, config):
    """
    [Sec 5.2] 失败挽救
    """
    # 1. 索引 STDB
    stdb_step_scores = collections.defaultdict(list)
    stdb_thoughts = []
    stdb_map = {} 
    idx = 0
    
    for step in g_buffer_steps:
        # 必须是成功且执行无误的步骤
        if step.get('R_core') != 1.0: continue
        if not step.get('action_success', False): continue
        
        action = step.get('parsed_action')
        if not action: continue
        action_key = str(action)
        
        # 关键：此时这里能读到非零的 R_step，因为我们在主函数里调整了顺序
        stdb_step_scores[action_key].append({'score': step.get('R_step', 0.0)})
        stdb_thoughts.append(step.get('thought', ''))
        stdb_map[idx] = (action_key, len(stdb_step_scores[action_key]) - 1)
        idx += 1

    if not stdb_thoughts: return
    stdb_embeddings = embedding_model.encode(stdb_thoughts, convert_to_tensor=True)
    for i, emb in enumerate(stdb_embeddings):
        action_key, list_idx = stdb_map[i]
        stdb_step_scores[action_key][list_idx]['embedding'] = emb

    # 2. 匹配失败步骤
    fail_steps, fail_indices = [], []
    for i, step in enumerate(g_calc_steps):
        if step.get('R_core') != -1.0: continue
        if not step.get('action_success', False): continue
        action = step.get('parsed_action')
        if not action: continue
        action_key = str(action)
        if action_key not in stdb_step_scores: continue
        
        fail_steps.append(step.get('thought', ''))
        fail_indices.append(i)
        
    if not fail_steps: return
    fail_embeddings = embedding_model.encode(fail_steps, convert_to_tensor=True)
    
    # 3. 计算相似度并赋值
    for i, emb_t in enumerate(fail_embeddings):
        step_idx = fail_indices[i]
        step = g_calc_steps[step_idx]
        action_key = str(step.get('parsed_action'))
        
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

def _calculate_separated_advantages(steps: List[Dict[str, Any]], omega: float):
    """
    [关键] 分离计算 Advantage (恢复原始逻辑，移除 SR 参数)
    """
    if not steps: return
    
    _calculate_A_traj(steps) # 恢复无参数调用
    _calculate_A_step(steps)
    
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
    [CCAPO V8 Final]
    修复数据传递错误 (Buffer Update Bug) 和 函数签名错误 (TypeError)。
    """
    ccapo_file_logger.info("=== [CCAPO V8] Start Calculation ===")
    
    # 0. 初始化
    keys = ['R_core_raw', 'R_match_raw', 'R_format_penalty', 'S_necessity', 'S_utility', 'I_action', 'Q_step', 'Z_novelty', 'Z_core', 'Z_match', 'TokenCost', 'b_stage', 'R_novelty_bonus']
    for step in g_calc_steps:
        for k in keys: step.setdefault(k, 0.0 if k != 'b_stage' else 'N/A')

    g_calc_trajs = _group_steps_by_traj(g_calc_steps)
    g_buffer_steps = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    
    # 1. R_tau & R_core
    _calculate_R_tau(g_calc_trajs, config)
    
    # 2. w_N 计算
    online_trajs = _group_steps_by_traj(g_online_steps)
    online_succ = sum(1 for steps in online_trajs.values() if steps and steps[0].get('R_core') == 1.0)
    sr = online_succ / (len(online_trajs) + 1e-6)
    
    max_w = config.get("max_w_N", 0.8)
    min_w = config.get("min_w_N", 0.2)
    w_N = min_w + (max_w - min_w) * (1.0 - sr)
    ccapo_file_logger.info(f"SR: {sr:.4f}, w_N: {w_N:.4f}")

    # 3. 计算 R_step (采用分段计算+即时写入，修复数据传递错误)
    _calculate_R_format_penalty(g_calc_steps, config)

    # --- Phase A: 优先处理成功轨迹 ---
    _calculate_R_step_success(g_calc_steps, g_calc_trajs, config)
    
    # [Critical Fix] 立即计算并写入成功步骤的 R_step
    # 确保 Buffer 中的成功数据有了 R_step，后续的失败挽救才能读取到
    success_steps = [s for s in g_calc_steps if s.get('R_core') == 1.0]
    if success_steps:
        raw_core = [s['R_core_raw'] for s in success_steps]
        z_core = _standardize(raw_core)
        for s, z in zip(success_steps, z_core):
            s['Z_core'] = z
            s['R_step'] = z + w_N * s.get('Z_novelty', 0.0) + s['R_format_penalty']

    # --- Phase B: 处理失败轨迹 ---
    # 此时 Buffer 数据已经就绪，可以进行匹配
    _calculate_R_step_fail(g_calc_steps, g_buffer_steps, embedding_model, config)
    
    fail_steps = [s for s in g_calc_steps if s.get('R_core') == -1.0]
    if fail_steps:
        raw_match = [s['R_match_raw'] for s in fail_steps]
        z_match = _standardize(raw_match)
        for s, z in zip(fail_steps, z_match):
            s['Z_match'] = z
            s['R_step'] = z + w_N * s.get('Z_novelty', 0.0) + s['R_format_penalty']

    # --- Phase C: 兜底 ---
    for step in g_calc_steps:
        if step.get('R_core') not in [1.0, -1.0]:
            step['R_step'] = step.get('R_format_penalty', 0.0)

    # 5. 计算 Advantage
    online_subset = [s for s in g_calc_steps if not s.get('is_buffer_data', False)]
    buffer_subset = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    
    # ✅ 恢复无 SR 参数调用
    _calculate_separated_advantages(online_subset, config.omega)
    _calculate_separated_advantages(buffer_subset, config.omega)
    
    ccapo_file_logger.info("=== [CCAPO V8] Done ===")
    return g_calc_steps, sr