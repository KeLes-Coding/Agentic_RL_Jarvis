# CCAPO/reward_manager.py

import torch
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

from CCAPO.stdb import STDB
from CCAPO.utils import compute_lcs_mask, detect_loop

logger = logging.getLogger(__name__)

class CCAPORewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, config=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.config = config or {}
        
        # 初始化 STDB
        ccapo_cfg = config.get('algorithm', {}).get('ccapo', {}) if config else {}
        self.stdb = STDB(ccapo_cfg)
        
        # 强制保存一次 STDB 以验证路径和权限
        print(f">>> [CCAPO] Initializing Reward Manager. STDB Path: {self.stdb.save_path}")
        self.stdb.save_checkpoint() # 测试保存
        
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
                # Update Exec
                if self.stdb.update_execution_anchor(traj['meta']['group_id'], traj):
                    updated = True
                    print(f">>> [STDB] New Anchor Found! Group: {traj['meta']['group_id']}, Steps: {traj['metrics']['total_steps']}")
                
                # Update Logic
                abs_acts = [s['action_abstract'] for s in traj['steps']]
                self.stdb.update_logic_consensus(traj['meta']['task_type'], abs_acts)
        
        # 只要有成功样本就保存 (防止 Anchor 没变但 Logic 变了的情况丢失)
        # 或者至少打印日志
        if success_count > 0:
            print(f">>> [CCAPO] Processed {success_count} success trajectories.")
            self.stdb.save_checkpoint() # 只要有新数据就存，保证数据不丢

    def __call__(self, data: Any, return_dict: bool = False) -> Any:
        """
        计算奖励的核心入口。
        Args:
            data: DataProto
            return_dict: 为兼容 RayTrainer 验证逻辑，如果为 True，返回 dict，否则返回 tensor。
        """
        # 1. 准备空 Reward Tensor (KeyError 修复)
        # 形状: (batch_size, response_length)
        # 注意: data.batch['responses'] 存在于 DataProto 中
        responses = data.batch['responses']
        batch_size, seq_len = responses.shape
        rewards_tensor = torch.zeros_like(responses, dtype=torch.float32, device=responses.device)
        
        # 获取 infos (包含 ccapo_trajectory)
        infos = data.non_tensor_batch.get('infos', [])
        
        # 2. 更新 STDB (并保存)
        self._update_stdb(infos)
        
        # 3. 计算 Dense Rewards & 保存轨迹
        self._compute_and_log(infos, rewards_tensor)
        
        # 4. 返回结果
        if return_dict:
            return {"reward_tensor": rewards_tensor}
        return rewards_tensor

    def _compute_and_log(self, infos: List[Dict], rewards_tensor: torch.Tensor):
        """
        遍历 Batch，计算奖励填入 tensor，并保存轨迹日志。
        """
        for b_idx, info in enumerate(infos):
            traj = info.get('ccapo_trajectory')
            if not traj: 
                continue
                
            steps = traj['steps']
            meta = traj['meta']
            
            # --- 保存轨迹到本地 (Log) ---
            # self._save_trajectory_to_disk(traj)
            
            # --- 准备计算数据 ---
            raw_actions = [s['action_raw'] for s in steps]
            group_id = meta['group_id']
            task_type = meta['task_type']
            
            # Exec Stream Logic
            anchor_actions = self.stdb.get_execution_anchor(group_id)
            lcs_mask = compute_lcs_mask(raw_actions, anchor_actions) if anchor_actions else [False]*len(steps)
            
            # 填坑逻辑 (Token Alignment)
            # 我们需要找到 Response 在 Input+Response 序列中的位置
            # verl 的 rewards_tensor 通常只对应 responses 部分 (shape: [bs, response_len])
            # 所以 index 0 对应 response 的第一个 token
            
            current_token_offset = 0
            history_actions = []
            
            # 累积环境本身的 Outcome Reward (如果有的话，通常在最后一步)
            final_env_reward = traj['metrics'].get('final_env_reward', 0.0)
            
            for t, step in enumerate(steps):
                n_tokens = step['llm_stats'].get('completion_tokens', 0)
                if n_tokens <= 0: continue # 保护
                
                # Reward 施加在当前 Action 结束的那个 Token 上
                reward_idx = current_token_offset + n_tokens - 1
                
                # 更新 offset
                current_token_offset += n_tokens
                
                # 越界检查
                if reward_idx >= rewards_tensor.shape[1]:
                    break
                
                # === 计算各种 Dense Rewards ===
                r_val = 0.0
                
                # 1. Exec
                r_val += self.exec_reward_on if lcs_mask[t] else self.exec_reward_off
                
                # 2. Logic (Skip first step)
                if t > 0:
                    prev_act = steps[t-1]['action_abstract']
                    curr_act = step['action_abstract']
                    score = self.stdb.get_transition_score(task_type, prev_act, curr_act)
                    r_val += self.logic_scale * score
                
                # 3. Milestone
                if step.get('milestones'):
                    r_val += self.milestone_reward * len(step['milestones'])
                
                # 4. Loop Penalty
                history_actions.append(step['action_raw'])
                if detect_loop(history_actions):
                    r_val += self.loop_penalty
                    
                # 5. [新增] 加上 Environment Outcome Reward
                # 策略：如果这是最后一步，加上环境给的稀疏奖励
                if t == len(steps) - 1:
                    r_val += final_env_reward

                # 写入 Tensor (CPU 操作)
                rewards_tensor[b_idx, reward_idx] = r_val

    def _save_trajectory_to_disk(self, traj: Dict):
        """
        将单条轨迹保存为 JSON 文件。
        结构: log_dir / task_type / group_id / timestamp_uuid.json
        """
        try:
            meta = traj['meta']
            task_type = meta.get('task_type', 'unknown')
            group_id = meta.get('group_id', 'unknown')
            
            # 构建目录
            save_dir = os.path.join(self.log_dir, task_type, group_id)
            os.makedirs(save_dir, exist_ok=True)
            
            # 文件名
            uid = str(hash(str(traj))) # 简单 hash 或 uuid
            timestamp = datetime.now().strftime("%H%M%S_%f")
            filename = f"{timestamp}_{uid}.json"
            
            filepath = os.path.join(save_dir, filename)
            
            # 写入
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(traj, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"[CCAPO] Failed to save trajectory: {e}")