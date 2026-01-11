# gigpo/core_ccapo.py

import torch
import numpy as np
import collections
from typing import List, Dict, Any
import logging
import os
import re

# ======================= ✅ 日志设置 ✅ =======================
logger = logging.getLogger(__name__)

# 匹配详情日志
match_logger = logging.getLogger("CCAPO_MATCH")
match_logger.setLevel(logging.INFO)
match_logger.propagate = False
if not match_logger.handlers:
    try:
        log_dir = "logger/CCAPO"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "ccapo_match_debug.log"), mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(message)s'))
        match_logger.addHandler(fh)
    except Exception:
        pass

# ======================= 🛠️ 基础工具函数 🛠️ =======================

def is_success_step(s: Dict) -> bool:
    """鲁棒的成功判定"""
    # 1. 检查 won
    if 'won' in s:
        val = s['won']
        if isinstance(val, (bool, np.bool_)):
            if val: return True
        elif isinstance(val, (str,)):
            if val.lower() == 'true': return True
        elif isinstance(val, (torch.Tensor, np.ndarray)):
            try:
                if val.item(): return True
            except: pass
            
    # 2. 检查 traj_task_completed
    if 'traj_task_completed' in s:
        val = s['traj_task_completed']
        if isinstance(val, (bool, np.bool_)) and val: return True
        if str(val).lower() == 'true': return True
    
    # 3. 检查 R_core
    # 注意：R_core 可能是 tensor，安全获取
    r_core = s.get('R_core', 0)
    if hasattr(r_core, 'item'): r_core = r_core.item()
    if r_core == 1.0:
        return True
        
    return False

def extract_actions_from_memory(steps: List[Dict]) -> List[str]:
    """🔥 [核心] 从内存提取动作，不再读文件"""
    actions = []
    for s in steps:
        act = s.get('parsed_action')
        if not act:
            act = s.get('executed_action_str')
        
        if act:
            clean_act = str(act).strip().lower()
            actions.append(clean_act)
    return actions

def compute_lcs_indices(seq1, seq2):
    """标准的 LCS 算法"""
    if not seq1 or not seq2: return set()
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    indices = set()
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            indices.add(i - 1)
            i -= 1; j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return indices

def _group_steps_by_traj(steps_list):
    grouped = collections.defaultdict(list)
    for s in steps_list:
        uid = s.get('traj_uid', 'unknown')
        if hasattr(uid, 'item'): uid = uid.item() # tensor -> scalar
        grouped[uid].append(s)
    # 确保按 step_index 排序
    for uid in grouped:
        grouped[uid].sort(key=lambda x: x.get('step_index', 0))
    return grouped

# ======================= 🧠 核心奖励逻辑 (复刻自旧版) 🧠 =======================

def _calculate_R_tau(g_calc_trajs, config):
    """计算 R_tau (轨迹级基础分)"""
    for traj_uid, steps in g_calc_trajs.items():
        if not steps: continue
        
        # 判定成功
        is_success = False
        for s in steps:
            if is_success_step(s):
                is_success = True
                break
        
        # 计算基础 R_core
        R_core = 1.0 if is_success else -1.0
        
        # 写入步骤 (确保所有步骤都有 R_core)
        for step in steps:
            step['R_tau'] = R_core 
            step['R_core'] = R_core

def _calculate_A_traj(g_calc_steps, current_sr):
    """计算 A_traj (轨迹优势)"""
    traj_map = {}
    for step in g_calc_steps:
        uid = step.get('traj_uid')
        if hasattr(uid, 'item'): uid = uid.item()
        if uid not in traj_map:
            # 优先使用 R_tau，如果没有则用 R_core
            traj_map[uid] = step.get('R_tau', step.get('R_core', 0.0))
    
    unique_r_taus = list(traj_map.values())
    if not unique_r_taus: return

    mean_R = np.mean(unique_r_taus)
    std_R = np.std(unique_r_taus) + 1e-6
    
    # 动态底线 (Dynamic Floor)
    success_floor = 0.0 if current_sr < 0.5 else -0.2

    traj_adv_map = {}
    for uid, r_val in traj_map.items():
        adv = (r_val - mean_R) / std_R
        
        # 保护逻辑：成功样本不给过低的负分
        if r_val > 0.0 and adv < success_floor:
            adv = success_floor
        elif r_val <= 0.0 and adv > -0.05:
            adv = -0.05 # 失败样本不给正分
            
        traj_adv_map[uid] = adv

    # 回写
    for step in g_calc_steps:
        uid = step.get('traj_uid')
        if hasattr(uid, 'item'): uid = uid.item()
        adv = traj_adv_map.get(uid, 0.0)
        step['A_traj'] = float(np.clip(adv, -3.0, 3.0))

def _calculate_A_step(g_calc_steps):
    """计算 A_step (步骤优势)"""
    if not g_calc_steps: return
    all_R_step = [step.get('R_step', 0.0) for step in g_calc_steps]
    mean_s = np.mean(all_R_step)
    std_s = np.std(all_R_step) + 1e-6
    
    for step in g_calc_steps:
        step['A_step'] = (step.get('R_step', 0.0) - mean_s) / std_s

def _calculate_separated_advantages(steps, omega, current_sr):
    """🔥 融合 A_traj 和 A_step，生成最终 advantages"""
    if not steps: return
    
    _calculate_A_traj(steps, current_sr)
    _calculate_A_step(steps)
    
    raw_advs = []
    for s in steps:
        # A_final = A_traj + omega * A_step
        raw = s.get('A_traj', 0.0) + omega * s.get('A_step', 0.0)
        s['A_final_raw'] = raw
        raw_advs.append(raw)
    
    # Batch Normalization (Whitening)
    mean_adv = np.mean(raw_advs)
    std_adv = np.std(raw_advs) + 1e-8
    
    for s in steps:
        # 最终写入 advantages 键，解决 KeyError
        s['advantages'] = (s['A_final_raw'] - mean_adv) / std_adv

# ======================= 🚀 主入口函数 🚀 =======================

def compute_ccapo_advantages(
    g_calc_steps: List[Dict],
    g_online_steps: List[Dict],
    g_buffer_steps: List[Dict],
    embedding_model: Any,
    config: Any
):
    """CCAPO 优势计算 (ALFWORLD In-Memory 完整版)"""
    match_logger.info("\n=== [CCAPO Algo Start (GigPO Core + Pipeline)] ===")
    
    # 1. 初始化默认值 (防止 Key Error)
    default_keys = {
        'R_core_raw': 0.0, 'R_match_raw': 0.0, 'R_format_penalty': 0.0,
        'Z_novelty': 0.0, 'R_repetition': 0.0, 'R_step': 0.0,
        'A_traj': 0.0, 'A_step': 0.0, 'advantages': 0.0 # 默认有值
    }
    for step in g_calc_steps:
        for k, v in default_keys.items():
            if k not in step: step[k] = v

    # 2. 轨迹分组
    g_calc_trajs = _group_steps_by_traj(g_calc_steps)
    online_trajs = _group_steps_by_traj(g_online_steps) # 用于计算 SR

    # 3. 计算基础 R_tau (Success/Fail)
    _calculate_R_tau(g_calc_trajs, config)

    # 4. 计算 LCS 匹配 (内存版)
    # Debug: 找出成功的 UID
    success_uids = []
    for uid, steps in g_calc_trajs.items():
        # R_core 已经被赋给所有步骤，检查第一个即可
        r_core = steps[0].get('R_core', 0)
        if hasattr(r_core, 'item'): r_core = r_core.item()
        if r_core == 1.0:
            success_uids.append(uid)
            
    match_logger.info(f"-> Total Trajs: {len(g_calc_trajs)}")
    match_logger.info(f"-> Success Trajs Count: {len(success_uids)}")
    if len(success_uids) > 0:
        match_logger.info(f"-> Success UIDs (Sample): {success_uids[:5]}") # 打印前5个ID，方便追踪

    success_trajs = {uid: g_calc_trajs[uid] for uid in success_uids}

    # 按 Prompt 分组
    trajs_by_prompt = collections.defaultdict(list)
    for uid, steps in g_calc_trajs.items():
        # 尝试多种 key 获取 prompt
        p_key = steps[0].get('prompt_index')
        if not p_key:
            p_key = steps[0].get('raw_prompt', 'default')
        p_key = str(p_key)
        
        trajs_by_prompt[p_key].append((uid, steps))

    # LCS 计算
    for p_key, traj_list in trajs_by_prompt.items():
        best_anchor_uid = None
        best_anchor_len = float('inf')
        best_anchor_actions = []

        # 选举 Anchor
        for uid, steps in traj_list:
            if uid in success_trajs:
                actions = extract_actions_from_memory(steps)
                if 0 < len(actions) < best_anchor_len:
                    best_anchor_len = len(actions)
                    best_anchor_uid = uid
                    best_anchor_actions = actions
        
        if best_anchor_uid:
            match_logger.info(f"   [Anchor] PromptID: {p_key[:10]} | AnchorUID: {str(best_anchor_uid)}")
        
        # 奖励分配
        if best_anchor_actions:
            for uid, steps in traj_list:
                curr_actions = extract_actions_from_memory(steps)
                matched_indices = compute_lcs_indices(curr_actions, best_anchor_actions)
                
                action_ptr = 0
                for s in steps:
                    act_str = s.get('parsed_action') or s.get('executed_action_str')
                    if act_str:
                        if action_ptr in matched_indices:
                            s['R_match_raw'] = 1.0 
                            # R_core 已经在 _calculate_R_tau 中根据 Success 设置过了
                            # 这里不再重复覆盖 R_core，除非是为了强调
                        action_ptr += 1
                
                # Anchor 自身
                if uid == best_anchor_uid:
                    for s in steps:
                        s['is_anchor'] = True
                        # Anchor 必须是 1.0，虽然本来就是
                        s['R_core'] = 1.0

    # 5. 聚合 R_step
    for s in g_calc_steps:
        # 获取 scalar R_core
        r_core_val = s.get('R_core', 0.0)
        if hasattr(r_core_val, 'item'): r_core_val = r_core_val.item()
        
        base = 1.0 if r_core_val == 1.0 else 0.0
        match = s.get('R_match_raw', 0.0)
        s['R_step'] = base + match 

    # 6. 🔥 [关键] 计算 Advantage
    online_succ_count = len(success_uids) # 直接使用上面统计的
    sr = online_succ_count / (len(online_trajs) + 1e-6)
    
    online_subset = [s for s in g_calc_steps if not s.get('is_buffer_data', False)]
    buffer_subset = [s for s in g_calc_steps if s.get('is_buffer_data', False)]
    
    omega = getattr(config, 'omega', 0.5) 
    
    _calculate_separated_advantages(online_subset, omega, sr)
    _calculate_separated_advantages(buffer_subset, omega, sr)

    match_logger.info("=== [CCAPO Algo Done] ===\n")
    return g_calc_steps, sr