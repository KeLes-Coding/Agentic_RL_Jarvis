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
    """[Sec 4.2 Modified] 计算 A_traj"""
    if not g_calc_steps: return
    all_R_tau = [step.get('R_tau', 0.0) for step in g_calc_steps]
    
    mean_R = np.mean(all_R_tau)
    std_R = np.std(all_R_tau)
    
    safe_std = std_R + 0.05 
    
    raw_advantages = [(v - mean_R) / safe_std for v in all_R_tau]
    protection_factor = min(max(current_sr, 0.2), 1.0)

    final_advantages = []
    for r_val, adv_val in zip(all_R_tau, raw_advantages):
        if r_val > 0.0 and adv_val < 0.0:
            adv_val = adv_val * protection_factor
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
        
        if is_success is True or action_status == 'true' or action_status == '' or action_status == 'SUCCESS':
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
    [Sec 5.1 Revised] LCS 序列对齐逻辑 - 全量日志 + 轨迹重建
    """
    use_fine_grained = getattr(config, 'use_fine_grained_action', True)
    beta = getattr(config, 'redundancy_penalty', 0.5)

    batch_id = datetime.datetime.now().strftime("%H:%M:%S.%f")
    match_debug_logger.info(f"\n\n{'='*20} [Batch {batch_id}] Start LCS Calculation (V8.6) {'='*20}")

    # 1. 准备数据 & 重建轨迹
    success_traj_info = {}
    
    for traj_uid, raw_steps in g_calc_trajs.items():
        if not raw_steps: continue
        
        if raw_steps[0].get('R_core') != 1.0: continue
        
        # 强制重建
        steps = _reconstruct_trajectory_from_disk(raw_steps)

        is_buffer = steps[0].get('is_buffer_data', False)
        
        # Log Path
        raw_path = steps[0].get('log_dir_path', 'Unknown_Path')
        if raw_path != 'Unknown_Path':
            clickable_path = f"file://{os.path.abspath(raw_path)}"
        else:
            clickable_path = "Path Not Available"

        # --- [DEBUG] Extract Log ---
        match_debug_logger.info(f"\n{'-'*10} Extracting: {traj_uid} {'-'*10}")
        match_debug_logger.info(f"  Source: {'[BUFFER]' if is_buffer else '[ONLINE]'}")
        match_debug_logger.info(f"  Path:   {clickable_path}")

        valid_steps = []
        act_identifiers = []
        
        for step in steps:
            s_idx = step.get('step_index')
            s_type = step.get('action_type', '')
            s_parsed = step.get('parsed_action')
            s_success = step.get('action_success', False)
            is_healed = step.get('is_healed_data', False)
            
            # 过滤: Buffer/Healed 宽容，Online 严格
            keep = False
            if is_buffer or is_healed:
                status = step.get('action_status', '')
                if not (status.startswith('FAILURE') or status.startswith('ERROR') or status.startswith('FORMAT')):
                    keep = True
            else:
                if s_success:
                    keep = True
            
            if not keep:
                match_debug_logger.info(f"  Step {s_idx}: [SKIPPED] Success={s_success}")
                continue
            
            valid_steps.append(step)
            
            # 构造 ID
            if use_fine_grained:
                if s_type in ['finish', 'input_text']:
                     act_identifier = f"{s_type}::{str(s_parsed).strip()}" if s_parsed else s_type
                else:
                     act_identifier = f"{s_type}::{str(s_parsed).strip()}" if s_parsed else s_type
            else:
                act_identifier = s_type
            
            act_identifiers.append(act_identifier)
            match_debug_logger.info(f"  Step {s_idx}: [KEEP] ID='{act_identifier}' {'(REC)' if is_healed else ''}")
            
        if valid_steps:
            success_traj_info[traj_uid] = {
                'steps': valid_steps, 
                'act_ids': act_identifiers, 
                'n_success': len(valid_steps),
                'is_buffer': is_buffer,
                'path': clickable_path,
                'r_tau': steps[0].get('R_tau', 0.0)
            }
            match_debug_logger.info(f"  -> Final Len: {len(valid_steps)}")
        else:
            match_debug_logger.info(f"  -> WARNING: No valid steps!")

    if not success_traj_info: 
        match_debug_logger.info("No successful trajectories found.")
        return

    # 2. 锚点选择
    def calculate_anchor_score(item):
        traj_uid, info = item
        steps = info['steps']
        n_succ = info['n_success']
        r_tau = info['r_tau']
        is_buffer = info['is_buffer']
        
        bad_taps = 0
        for s in steps:
            if s.get('action_type') == 'tap':
                parsed = str(s.get('parsed_action', ''))
                if '1' in parsed or '(0,' in parsed:
                    bad_taps += 1
        
        buffer_priority_bonus = -1000.0 if is_buffer else 1000.0
        
        score = buffer_priority_bonus + (1.0 * n_succ) + (0.5 * bad_taps) - (0.1 * r_tau)
        info['debug_score'] = score
        return score

    sorted_trajs = sorted(success_traj_info.items(), key=calculate_anchor_score)
    
    # --- [DEBUG] Anchor Table ---
    match_debug_logger.info(f"\n--- Anchor Candidates ---")
    for rank, (uid, info) in enumerate(sorted_trajs[:5]):
        src = "BUFFER" if info['is_buffer'] else "ONLINE"
        match_debug_logger.info(f"{rank+1}. {src} | Score: {info['debug_score']:.2f} | Len: {info['n_success']} | UID: {uid}")

    anchor_uid, anchor_data = sorted_trajs[0]
    anchor_seq = anchor_data['act_ids']

    match_debug_logger.info(f"\n>>> SELECTED ANCHOR: {anchor_uid}")
    match_debug_logger.info(f">>> Seq: {anchor_seq}")

    # 3. 序列对齐与赋值
    for traj_uid, info in success_traj_info.items():
        current_seq = info['act_ids']
        current_steps = info['steps']
        
        m_steps_ratio = info['n_success'] / config.max_steps
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

        # --- [DEBUG] 打印所有匹配详情 ---
        match_rate = len(matched_indices) / (len(current_seq) + 1e-6)
        match_debug_logger.info(f"\n[Match Result] Traj: {traj_uid}")
        match_debug_logger.info(f"  Source: {'BUFFER' if info['is_buffer'] else 'ONLINE'}")
        match_debug_logger.info(f"  Match Rate: {match_rate:.2f} ({len(matched_indices)}/{len(current_seq)})")
        
        visual_match = []
        for idx, act in enumerate(current_seq):
            mark = "✅" if idx in matched_indices else "❌"
            visual_match.append(f"{mark} {act}")
        match_debug_logger.info(f"  Alignment: {visual_match}")

        # 回写 (Write Back)
        for idx, step in enumerate(current_steps):
            if step.get('is_healed_data'): continue 

            is_match = (idx in matched_indices) or step.get('is_human_marked', False)
            
            act_type = step.get('action_type', '')
            parsed = str(step.get('parsed_action', ''))
            is_suspicious_tap = (act_type == 'tap' and ('1' in parsed or '(0,' in parsed))

            if is_match:
                if is_suspicious_tap and not step.get('is_human_marked', False):
                    i_action = 0.1
                else:
                    i_action = 1.0
            else:
                i_action = -beta

            r_core_raw = i_action * q_step if i_action > 0 else i_action
            
            step['R_core_raw'] = r_core_raw
            step['I_action'] = i_action
            step['Q_step'] = q_step
            step['S_necessity'] = 1.0 if is_match else 0.0
            step['S_utility'] = 1.0 if is_match else 0.0

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
    [关键] 分离计算 Advantage
    """
    if not steps: return
    
    _calculate_A_traj(steps, current_sr)
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
    [CCAPO V8.6] (With Reconstruction & Full Logs)
    """
    ccapo_file_logger.info("=== [CCAPO V8.6] Start Calculation ===")
    
    keys = ['R_core_raw', 'R_match_raw', 'R_format_penalty', 'S_necessity', 'S_utility', 'I_action', 'Q_step', 'Z_novelty', 'Z_core', 'Z_match', 'TokenCost', 'b_stage', 'R_novelty_bonus']
    for step in g_calc_steps:
        for k in keys: step.setdefault(k, 0.0 if k != 'b_stage' else 'N/A')

    for step in g_calc_steps:
        if step.get('is_buffer_data', False):
            if 'traj_task_completed' not in step: step['traj_task_completed'] = True
            if 'traj_total_steps' not in step: step['traj_total_steps'] = step.get('step_index', 0)
            if 'traj_total_tokens' not in step: step['traj_total_tokens'] = 0

    g_calc_trajs = _group_steps_by_traj(g_calc_steps)
    g_buffer_steps = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    
    _calculate_R_tau(g_calc_trajs, config)
    
    online_trajs = _group_steps_by_traj(g_online_steps)
    online_succ = sum(1 for steps in online_trajs.values() if steps and steps[0].get('R_core') == 1.0)
    sr = online_succ / (len(online_trajs) + 1e-6)
    
    max_w = config.get("max_w_N", 0.8)
    min_w = config.get("min_w_N", 0.2)
    w_N = min_w + (max_w - min_w) * (1.0 - sr)
    
    ccapo_file_logger.info(f"SR: {sr:.4f}, w_N: {w_N:.4f}")

    _calculate_R_format_penalty(g_calc_steps, config)

    _calculate_R_step_success(g_calc_steps, g_calc_trajs, config)
    
    success_steps = [s for s in g_calc_steps if s.get('R_core') == 1.0]
    if success_steps:
        raw_core = [s['R_core_raw'] for s in success_steps]
        z_core = _standardize(raw_core)
        for s, z in zip(success_steps, z_core):
            s['Z_core'] = z
            s['R_step'] = z + w_N * s.get('Z_novelty', 0.0) + s['R_format_penalty']

    _calculate_R_step_fail(g_calc_steps, g_buffer_steps, embedding_model, config)
    
    fail_steps = [s for s in g_calc_steps if s.get('R_core') == -1.0]
    if fail_steps:
        raw_match = [s['R_match_raw'] for s in fail_steps]
        z_match = _standardize(raw_match)
        for s, z in zip(fail_steps, z_match):
            s['Z_match'] = z
            s['R_step'] = z + w_N * s.get('Z_novelty', 0.0) + s['R_format_penalty']

    for step in g_calc_steps:
        if step.get('R_core') not in [1.0, -1.0]:
            step['R_step'] = step.get('R_format_penalty', 0.0)

    online_subset = [s for s in g_calc_steps if not s.get('is_buffer_data', False)]
    buffer_subset = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    
    _calculate_separated_advantages(online_subset, config.omega, sr)
    _calculate_separated_advantages(buffer_subset, config.omega, sr)
    
    ccapo_file_logger.info("=== [CCAPO V8.6] Done ===")
    return g_calc_steps, sr