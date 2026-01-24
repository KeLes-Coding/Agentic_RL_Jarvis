# CCAPO/reward_manager.py

import torch
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Set
import uuid

# [CCAPO 依赖]
from CCAPO.stdb import STDB
from CCAPO.utils import detect_loop
from CCAPO.adapter import ALFWorldAdapter
from CCAPO.logger import get_logger

logger = logging.getLogger(__name__)

class CCAPORewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, config=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.config = config or {}
        
        # [Component Init]
        self.adapter = ALFWorldAdapter()
        self.logger = get_logger()
        self.global_batch_cnt = 0
        
        # [STDB Init]
        ccapo_cfg = config.get('algorithm', {}).get('ccapo', {}) if config else {}
        self.stdb = STDB(ccapo_cfg)
        
        # 验证连接
        actor_name = getattr(self.stdb, 'actor_name', 'Local')
        print(f">>> [CCAPO] Reward Manager Connected to Global STDB. Actor Name: {actor_name}")
        
        # [Hyperparameters]
        # [Phase 2 Removed] LCS Execution Rewards (r_exec_on, r_exec_off)
        # [Phase 2 Removed] Milestone Rewards (r_milestone)
        self.logic_scale = ccapo_cfg.get('r_logic_scale', 0.05)
        self.loop_penalty = ccapo_cfg.get('r_loop_penalty', -0.5)

    def _update_stdb(self, infos) -> Set[str]:
        """
        更新 STDB 并识别先锋轨迹 (Pioneers)。
        Returns:
            pioneer_groups: 本次更新中打破纪录的 Group ID 集合。
        """
        pioneer_groups = set()
        success_count = 0
        
        for info in infos:
            traj = info.get('ccapo_trajectory')
            # 只用成功轨迹更新 STDB
            if traj and traj['metrics'].get('is_success', False):
                success_count += 1
                group_id = traj['meta']['group_id']
                
                # 1. Update Execution Stream (Check for Pioneer)
                # 如果 update_execution_anchor 返回 True，说明这是该环境下的新最优解
                if self.stdb.update_execution_anchor(group_id, traj):
                    pioneer_groups.add(group_id)
                    print(f">>> [STDB] 🔥 Pioneer Found! Group: {group_id[:6]}, Steps: {traj['metrics']['total_steps']}")
                
                # 2. Update Logic Stream (Skip-Gram Mining)
                abs_acts = [s['action_abstract'] for s in traj['steps']]
                # 注意：Client 端 API 已升级为直接接收 list
                self.stdb.update_logic_consensus(traj['meta']['task_type'], abs_acts, is_success=True)
        
        # 异步触发保存
        if success_count > 0:
            self.stdb.save_checkpoint()
            
        return pioneer_groups

    def __call__(self, data: Any, return_dict: bool = False) -> Any:
        self.global_batch_cnt += 1
        
        # 1. Init Buffer
        responses = data.batch['responses']
        rewards_tensor = torch.zeros_like(responses, dtype=torch.float32, device=responses.device)
        
        infos = data.non_tensor_batch.get('infos', [])
        if len(infos) == 0:
            self.logger.log_event("missing_infos", {"reason": "no infos", "batch": self.global_batch_cnt})
            return rewards_tensor
        
        # 2. Update STDB & Identify Pioneers
        pioneer_groups = self._update_stdb(infos)
        
        # 3. Compute Rewards with Routing Logic
        self._compute_and_log(infos, rewards_tensor, pioneer_groups)
        
        if return_dict:
            return {"reward_tensor": rewards_tensor}
        return rewards_tensor

    def _compute_and_log(self, infos: List[Dict], rewards_tensor: torch.Tensor, pioneer_groups: Set[str]):
        """
        核心计算逻辑：Phase 2 Simplified (Removal of LCS & Milestones)
        """
        # Batch 级统计容器 (移除 exec_hits, milestones_triggered)
        batch_stats = {
            "success_count": 0, "fail_count": 0, "pioneer_count": len(pioneer_groups),
            "logic_score_sum": 0.0, "loops_detected": 0
        }

        traj_found = 0
        for b_idx, info in enumerate(infos):
            traj = info.get('ccapo_trajectory')
            if not traj: continue
            traj_found += 1
                
            steps = traj['steps']
            meta = traj['meta']
            metrics = traj['metrics']
            
            # --- 状态判定 ---
            is_success = metrics.get('is_success', False)
            group_id = meta['group_id']
            is_pioneer = group_id in pioneer_groups
            
            # 统计
            if is_success: batch_stats["success_count"] += 1
            else: batch_stats["fail_count"] += 1

            # --- [Phase 2] 简化路由逻辑 ---
            # 暂时只依据是否开启 Logic Reward
            # 在 Phase 3 我们将引入完整的 Update-then-Evaluate 路由
            enable_logic = True # 默认开启图谱引导
            
            # --- Step 循环计算 ---
            current_token_offset = 0
            history_actions = []
            final_env_reward = metrics.get('final_env_reward', 0.0)
            
            task_type = meta['task_type']
            episode_uuid = str(uuid.uuid4())[:8]

            for t, step in enumerate(steps):
                n_tokens = step['llm_stats'].get('completion_tokens', 0)
                if n_tokens <= 0: continue
                
                # 定位 Reward 在 Tensor 中的位置
                reward_idx = current_token_offset + n_tokens - 1
                current_token_offset += n_tokens
                if reward_idx >= rewards_tensor.shape[1]: break
                
                # === 组件计算 ===
                r_logic = 0.0
                r_loop = 0.0
                r_env = 0.0
                
                # 1. Logic Reward (概率图谱)
                # 仅在非第一步计算转移分
                if enable_logic and t > 0:
                    prev_act = steps[t-1]['action_abstract']
                    curr_act = step['action_abstract']
                    # 从 Graph 获取 (Importance * Criticality * Utility)
                    score = self.stdb.get_transition_score(task_type, prev_act, curr_act)
                    r_logic = self.logic_scale * score
                    batch_stats["logic_score_sum"] += score
                
                # 2. Loop Penalty (全局生效，一票否决)
                history_actions.append(step['action_raw'])
                if detect_loop(history_actions):
                    r_loop = self.loop_penalty
                    batch_stats["loops_detected"] += 1
                    # 发现 Loop 时，剥夺正向奖励
                    r_logic = 0.0
                    
                # 3. Environment Outcome
                if t == len(steps) - 1:
                    r_env = final_env_reward

                # 汇总 (r_exec 和 r_milestone 已移除)
                total_val = r_logic + r_loop + r_env
                rewards_tensor[b_idx, reward_idx] = total_val
                
                # [Logger] 记录归因
                self.logger.log_reward_composition(
                    step=self.global_batch_cnt,
                    trace_id=group_id,
                    step_idx=t,
                    total_reward=total_val,
                    components={
                        "logic": r_logic,
                        "loop": r_loop,
                        "env": r_env
                    },
                    meta={
                        "uuid": episode_uuid,
                        "act": step['action_abstract'], 
                        "valid": step['is_valid'],
                        "status": "PIONEER" if is_pioneer else ("SUCCESS" if is_success else "FAIL")
                    }
                )

        # 写入 Metric Log
        self.logger.log_batch_metrics(global_step=self.global_batch_cnt, metrics=batch_stats)