# gigpo/core_ccapo.py

import numpy as np
import torch
from collections import defaultdict
import logging
from typing import TYPE_CHECKING, Any

# 解决 DataProto 未定义报错，仅在类型检查时导入
if TYPE_CHECKING:
    from verl import DataProto

logger = logging.getLogger("CCAPO_CORE")

def compute_lcs_match_indices(seq_a, seq_b):
    """标准 LCS 算法，返回匹配的索引集合"""
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
            matched_indices.add(i-1) # 记录 seq_a 的索引
            i -= 1; j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    return matched_indices

def compute_ccapo_outcome_advantage(
    batch: Any, # 使用 Any 避免运行时 import 循环
    stdb_manager,
    config=None
):
    """
    CCAPO 核心优势计算函数 (带详细 Debug 日志)
    """
    logger.info(">>> [CCAPO] Start Advantage Calculation")
    
    # 1. 提取元数据
    try:
        traj_uids = batch.non_tensor_batch['traj_uid']
        actions_raw = batch.non_tensor_batch['executed_action_str']
        prompts_raw = batch.non_tensor_batch.get('raw_prompt', [])
        # 注意：PPO 的 batch 通常是 token 级别的，这里 rewards 是 (BS, Seq_Len)
        rewards = batch.batch['token_level_rewards']
        response_mask = batch.batch['response_mask']
        
        bsz = len(traj_uids)
        logger.info(f"    Batch Size: {bsz}, Device: {rewards.device}")
        
    except KeyError as e:
        logger.error(f"!!! [CCAPO] Missing Key in Batch: {e}")
        logger.error(f"    Available Keys: {batch.non_tensor_batch.keys()}")
        raise e

    # 初始化优势 Tensor
    ccapo_advantages = torch.zeros_like(rewards)
    
    # 2. 按轨迹聚合
    traj_groups = defaultdict(list)
    for i, t_uid in enumerate(traj_uids):
        traj_groups[t_uid].append(i)
        
    logger.info(f"    Found {len(traj_groups)} unique trajectories.")

    # 3. 遍历评估
    for t_uid, step_indices in traj_groups.items():
        step_indices.sort()
        
        # 获取 Prompt (取第一步的 prompt)
        first_idx = step_indices[0]
        prompt_key = str(prompts_raw[first_idx]).strip()
        
        # 清洗动作序列 (关键：去空格、转小写，防止不匹配)
        current_actions = [str(actions_raw[i]).strip().lower() for i in step_indices]
        
        logger.info(f"    --- Evaluating Traj: {t_uid} ---")
        logger.info(f"    Prompt Key: {prompt_key}")
        logger.info(f"    Agent Actions: {current_actions}")

        # 查询 STDB
        anchor_seq = stdb_manager.get_best_sequence(prompt_key)
        # 如果 STDB 存的是由 string 组成的 list，确保也做同样的清洗
        anchor_seq = [str(a).strip().lower() for a in anchor_seq]
        
        logger.info(f"    Anchor Actions (STDB): {anchor_seq}")

        step_scores = []
        
        if anchor_seq:
            # === 有锚点：LCS 匹配 ===
            matched_indices = compute_lcs_match_indices(current_actions, anchor_seq)
            logger.info(f"    Matched Indices (Agent View): {matched_indices}")
            
            for k, idx_in_batch in enumerate(step_indices):
                if k in matched_indices:
                    score = 1.0
                    status = "MATCH (+1.0)"
                else:
                    score = -0.1
                    status = "MISS (-0.1)"
                
                step_scores.append(score)
                # 写入 Tensor (广播到整行，稍后由 mask 过滤)
                ccapo_advantages[idx_in_batch] = score
                
                logger.debug(f"      Step {k}: {current_actions[k]} -> {status}")
        else:
            # === 冷启动 ===
            logger.warning(f"    [Cold Start] No anchor found for prompt '{prompt_key}'. Assigning 0.0.")
            for idx_in_batch in step_indices:
                ccapo_advantages[idx_in_batch] = 0.0

    # 4. 应用 Mask
    final_adv = ccapo_advantages * response_mask
    
    # 统计非零分数的数量，用于调试
    non_zero_count = torch.count_nonzero(final_adv).item()
    logger.info(f"<<< [CCAPO] Calculation Done. Non-zero elements: {non_zero_count}")
    
    return final_adv, final_adv