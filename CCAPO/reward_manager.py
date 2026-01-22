# CCAPO/reward_manager.py

import torch
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# [CCAPO 修改] 引入分布式 STDB 客户端、适配器和日志系统
from CCAPO.stdb import STDB
from CCAPO.utils import compute_lcs_mask, detect_loop
from CCAPO.adapter import ALFWorldAdapter  # 新增：适配器
from CCAPO.logger import get_logger        # 新增：日志系统

logger = logging.getLogger(__name__)

class CCAPORewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, config=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.config = config or {}
        
        # [CCAPO 修改] 初始化 Adapter 和 Logger
        self.adapter = ALFWorldAdapter()
        self.logger = get_logger()
        
        # 初始化 STDB (会自动连接到 Global Ray Actor)
        ccapo_cfg = config.get('algorithm', {}).get('ccapo', {}) if config else {}
        self.stdb = STDB(ccapo_cfg)
        
        # 验证连接 (打印 Actor 名称以确认是 Client 模式)
        actor_name = getattr(self.stdb, 'actor_name', 'Local')
        print(f">>> [CCAPO] Reward Manager Connected to Global STDB. Actor Name: {actor_name}")
        # self.stdb.save_checkpoint() # 可选：测试保存
        
        self.exec_reward_on = ccapo_cfg.get('r_exec_on', 0.2)
        self.exec_reward_off = ccapo_cfg.get('r_exec_off', -0.01)
        self.logic_scale = ccapo_cfg.get('r_logic_scale', 0.05)
        self.milestone_reward = ccapo_cfg.get('r_milestone', 0.3)
        self.loop_penalty = ccapo_cfg.get('r_loop_penalty', -0.5)

    def _update_stdb(self, infos):
        updated = False
        success_count = 0
        
        for info in infos:
            traj = info.get('ccapo_trajectory')
            if traj and traj['metrics'].get('is_success', False):
                success_count += 1
                
                # 获取 Group ID (兼容旧逻辑)
                group_id = traj['meta']['group_id']
                
                # Update Exec
                if self.stdb.update_execution_anchor(group_id, traj):
                    updated = True
                    print(f">>> [STDB] New Anchor Found! Group: {group_id}, Steps: {traj['metrics']['total_steps']}")
                
                # Update Logic
                abs_acts = [s['action_abstract'] for s in traj['steps']]
                self.stdb.update_logic_consensus(traj['meta']['task_type'], abs_acts)
        
        # 只要有成功样本就触发保存建议 (Server 端决定是否真正落盘)
        if success_count > 0:
            # print(f">>> [CCAPO] Processed {success_count} success trajectories.")
            self.stdb.save_checkpoint()

    def __call__(self, data: Any, return_dict: bool = False) -> Any:
        """
        计算奖励的核心入口。
        """
        # 1. 准备空 Reward Tensor
        responses = data.batch['responses']
        batch_size, seq_len = responses.shape
        rewards_tensor = torch.zeros_like(responses, dtype=torch.float32, device=responses.device)
        
        # 获取 infos (包含 ccapo_trajectory)
        infos = data.non_tensor_batch.get('infos', [])
        if len(infos) == 0:
            self.logger.log_event("missing_infos", {"reason": "non_tensor_batch missing infos", "batch_size": int(batch_size)})
        
        # 2. 更新 STDB
        self._update_stdb(infos)
        
        # 3. 计算 Dense Rewards & 记录详细日志
        self._compute_and_log(infos, rewards_tensor)
        
        # 4. 返回结果
        if return_dict:
            return {"reward_tensor": rewards_tensor}
        return rewards_tensor

    def _compute_and_log(self, infos: List[Dict], rewards_tensor: torch.Tensor):
        """
        遍历 Batch，计算奖励填入 tensor，并保存轨迹日志。
        """
        # [CCAPO 新增] 初始化本 Batch 的统计计数器
        batch_stats = {
            "success_count": 0,
            "fail_count": 0,
            "exec_hits": 0,          # 命中骨架的步数
            "logic_score_sum": 0.0,  # 获得的逻辑奖励总和
            "milestones_triggered": 0,
            "loops_detected": 0
        }

        traj_found = 0
        for b_idx, info in enumerate(infos):
            traj = info.get('ccapo_trajectory')
            if not traj: 
                continue
            traj_found += 1
                
            steps = traj['steps']
            meta = traj['meta']
            metrics = traj['metrics']
            
            # 统计成败
            if metrics.get('is_success', False):
                batch_stats["success_count"] += 1
            else:
                batch_stats["fail_count"] += 1
            
            # --- 保存轨迹到本地 (Log) ---
            # self._save_trajectory_to_disk(traj)
            
            # --- 准备计算数据 ---
            raw_actions = [s['action_raw'] for s in steps]
            group_id = meta['group_id']
            task_type = meta['task_type']
            
            # Exec Stream Logic
            anchor_actions = self.stdb.get_execution_anchor(group_id)
            lcs_mask = compute_lcs_mask(raw_actions, anchor_actions) if anchor_actions else [False]*len(steps)
            
            current_token_offset = 0
            history_actions = []
            
            # 累积环境本身的 Outcome Reward
            final_env_reward = metrics.get('final_env_reward', 0.0)
            
            for t, step in enumerate(steps):
                n_tokens = step['llm_stats'].get('completion_tokens', 0)
                if n_tokens <= 0: continue # 保护
                
                # Reward 施加在当前 Action 结束的那个 Token 上
                reward_idx = current_token_offset + n_tokens - 1
                current_token_offset += n_tokens
                
                if reward_idx >= rewards_tensor.shape[1]:
                    break
                
                # === 计算各种 Dense Rewards ===
                r_val = 0.0
                
                # 1. Exec
                if lcs_mask[t]:
                    r_val += self.exec_reward_on
                    batch_stats["exec_hits"] += 1  # [统计]
                else:
                    r_val += self.exec_reward_off
                
                # 2. Logic (Skip first step)
                if t > 0:
                    prev_act = steps[t-1]['action_abstract']
                    curr_act = step['action_abstract']
                    score = self.stdb.get_transition_score(task_type, prev_act, curr_act)
                    r_val += self.logic_scale * score
                    batch_stats["logic_score_sum"] += score # [统计]
                
                # 3. Milestone
                if step.get('milestones'):
                    ms_count = len(step['milestones'])
                    r_val += self.milestone_reward * ms_count
                    batch_stats["milestones_triggered"] += ms_count # [统计]
                
                # 4. Loop Penalty
                history_actions.append(step['action_raw'])
                if detect_loop(history_actions):
                    r_val += self.loop_penalty
                    batch_stats["loops_detected"] += 1 # [统计]
                    
                # 5. Environment Outcome Reward
                if t == len(steps) - 1:
                    r_val += final_env_reward

                # 写入 Tensor (CPU 操作)
                rewards_tensor[b_idx, reward_idx] = r_val

        # [CCAPO 新增] 将本 Batch 的统计指标写入本地日志文件
        # Step 暂时填 0，或者您可以尝试从 infos 中获取 global step
        self.logger.log_step_metrics(step=0, metrics=batch_stats)
        if traj_found == 0:
            self.logger.log_event("missing_trajectory", {"reason": "no ccapo_trajectory in infos", "batch_size": int(len(infos))})

    def _save_trajectory_to_disk(self, traj: Dict):
        """
        将单条轨迹保存为 JSON 文件。
        结构: log_dir / task_type / group_id / timestamp_uuid.json
        """
        try:
            meta = traj['meta']
            task_type = meta.get('task_type', 'unknown')
            group_id = meta.get('group_id', 'unknown')
            
            # 构建目录 (self.log_dir 需要在 init 中设置，或者使用全局 logger 目录)
            # 这里为了不破坏原有逻辑，暂时不做改动，使用默认路径或 logger 路径
            save_dir = os.path.join("experiments/ccapo_logs/trajectories", task_type, group_id)
            os.makedirs(save_dir, exist_ok=True)
            
            # 文件名
            uid = str(hash(str(traj))) 
            timestamp = datetime.now().strftime("%H%M%S_%f")
            filename = f"{timestamp}_{uid}.json"
            
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(traj, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"[CCAPO] Failed to save trajectory: {e}")