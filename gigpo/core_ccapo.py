# gigpo/core_ccapo.py

import numpy as np
import torch
from collections import defaultdict
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verl import DataProto

logger = logging.getLogger("CCAPO_CORE")

def get_abstract_id(action_str):
    """辅助函数：获取抽象指纹"""
    if not action_str: return ""
    return re.sub(r'\s\d+', '', str(action_str)).strip().lower()

def compute_lcs_match_indices(seq_a, seq_b):
    """
    标准最长公共子序列 (LCS) 匹配。
    返回 seq_a 中属于 LCS 一部分的索引集合。
    """
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i-1] == seq_b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    matched_indices = set()
    i, j = m, n
    while i > 0 and j > 0:
        if seq_a[i-1] == seq_b[j-1]:
            matched_indices.add(i-1)
            i -= 1; j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    return matched_indices

def check_milestone(observation, task_type):
    """
    里程碑检测 (Milestone Detection)
    基于 ALFWorld 的 Observation 文本给予稠密奖励。
    """
    obs_lower = str(observation).lower()
    task_type = str(task_type).lower()
    
    # 定义里程碑关键词 (根据文档配置表)
    milestones = {
        'pick': ['holding'], # Pick & Place / Pick Two
        'cool': ['cool', 'chilled'],
        'heat': ['hot', 'heated'],
        'clean': ['clean'],
        'slice': ['slice']
    }
    
    # 简单的包含匹配
    reward = 0.0
    for key, keywords in milestones.items():
        if key in task_type:
            for kw in keywords:
                if kw in obs_lower:
                    reward = 0.3 # 显著奖励
                    break
        if reward > 0: break
    return reward

def compute_ccapo_outcome_advantage(
    batch: Any, 
    stdb_manager,
    config=None
):
    """
    CCAPO v2.0 核心优势计算:
    Stream A: Concrete LCS (Exact Memory)
    Stream B: Abstract Logic (Generalization)
    Milestone: State change detection
    """
    logger.info(">>> [CCAPO v2.0] Dual-Stream Advantage Calculation")
    
    # 1. 提取元数据 (确保 env_manager.py 中已注入这些字段)
    traj_uids = batch.non_tensor_batch['traj_uid']
    actions_raw = batch.non_tensor_batch['executed_action_str']
    # 关键字段：task_type 用于 Logic Stream
    task_types = batch.non_tensor_batch.get('task_type', ['unknown'] * len(traj_uids))
    # 关键字段：observation_text 用于 Milestone
    observations = batch.non_tensor_batch.get('observation_text', [''] * len(traj_uids))
    
    # 提示词 ID，用于识别同一个 Group (Task Instance)
    # 假设 prompt_index 或 raw_prompt 可以唯一标识一个 Seed Group
    group_keys = batch.non_tensor_batch.get('prompt_index', [])
    if len(group_keys) == 0:
        group_keys = [str(p) for p in batch.non_tensor_batch.get('raw_prompt', [])]

    rewards = batch.batch['token_level_rewards']
    response_mask = batch.batch['response_mask']
    
    # 初始化
    final_advantage = torch.zeros_like(rewards)
    
    # 2. 构建 Group 映射: Group_Key -> List[Traj_UID]
    # 以及 Traj_UID -> List[Batch_Indices]
    traj_to_indices = defaultdict(list)
    for i, uid in enumerate(traj_uids):
        traj_to_indices[uid].append(i)
        
    group_to_uids = defaultdict(set)
    uid_to_group_key = {}
    for i, uid in enumerate(traj_uids):
        g_key = str(group_keys[i])
        group_to_uids[g_key].add(uid)
        uid_to_group_key[uid] = g_key

    # 3. 遍历每个 Group 进行计算
    for g_key, uids in group_to_uids.items():
        sorted_uids = sorted(list(uids))
        
        # --- Stream A: Local Anchor 选举 ---
        # 优先从 Exact STDB 获取 (Memory)
        anchor_seq_concrete = stdb_manager.get_best_sequence(g_key)
        source = "STDB"
        
        # 如果 STDB 没有，尝试在当前 Group 内选举 (RL Exploration)
        # 简单逻辑：假设该 Group 内没有成功轨迹，则 anchor 为空
        # 复杂逻辑：如果有成功轨迹，取最短的作为 anchor (此处省略，以 STDB 为主)
        
        if anchor_seq_concrete:
            # 清洗 Anchor
            anchor_seq_concrete = [s.strip().lower() for s in anchor_seq_concrete]
        
        # --- 遍历 Group 内的每条轨迹 ---
        for t_uid in sorted_uids:
            step_indices = sorted(traj_to_indices[t_uid])
            if not step_indices: continue
            
            # 获取该轨迹的元数据
            first_idx = step_indices[0]
            task_type = str(task_types[first_idx])
            
            # 准备当前轨迹的动作序列
            curr_concrete = [str(actions_raw[i]).strip().lower() for i in step_indices]
            curr_abstract = [get_abstract_id(str(actions_raw[i])) for i in step_indices]
            
            # 计算 LCS 匹配索引 (如果存在 Anchor)
            matched_indices = set()
            if anchor_seq_concrete:
                matched_indices = compute_lcs_match_indices(curr_concrete, anchor_seq_concrete)
            
            # 逐步计算奖励
            for k, idx in enumerate(step_indices):
                step_reward = 0.0
                matched_rule = "None"
                
                # --- 规则 1: Stream A (Exact LCS) ---
                if k in matched_indices:
                    step_reward += 1.0 # Hit Anchor
                    matched_rule = "Exact_LCS"
                elif anchor_seq_concrete:
                    step_reward -= 0.1 # Off Anchor Penalty
                    
                # --- 规则 2: Stream B (Abstract Logic) ---
                # 仅当 Stream A 未匹配时启用 (或作为额外的 Shaping)
                if matched_rule == "None" and k > 0:
                    # 检查 Bigram: prev -> curr
                    bigram = f"{curr_abstract[k-1]}->{curr_abstract[k]}"
                    consensus = stdb_manager.get_abstract_consensus(task_type)
                    freq = consensus.get(bigram, 0)
                    
                    # 简单阈值判断 (可根据实际数据调整)
                    if freq > 5: 
                        step_reward += 0.05
                        matched_rule = "Abstract_Logic"
                
                # --- 规则 3: Milestone (Observation) ---
                obs = str(observations[idx])
                milestone_val = check_milestone(obs, task_type)
                if milestone_val > 0:
                    step_reward += milestone_val
                    matched_rule += "+Milestone"
                
                # 写入 Tensor
                final_advantage[idx] = step_reward
                # logger.debug(f"Step {k} [{curr_concrete[k]}]: {step_reward} ({matched_rule})")

    # 应用 Mask
    final_advantage = final_advantage * response_mask
    return final_advantage, final_advantage