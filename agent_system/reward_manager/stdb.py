# agent_system/reward_manager/stdb.py

import os
import json
import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Any, Union

logger = logging.getLogger(__name__)

class SuccessTrajectoryDatabase:
    """
    轻量级 STDB (Lite Version for ALFWorld)
    仅在内存中维护 {Prompt_ID: [Best_Action_Sequence]} 的映射。
    """
    def __init__(self, save_path: str, top_k: int = 1, tokenizer: Any = None):
        self.save_path = save_path
        self.top_k = top_k
        self.tokenizer = tokenizer 
        self.db: Dict[str, List[str]] = {} # Prompt_Str -> List[Action_Str]
        self.load()

    def load(self):
        """从磁盘加载 JSON 数据库"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    self.db = json.load(f)
                logger.info(f"[STDB] Loaded {len(self.db)} prompts from {self.save_path}")
            except Exception as e:
                logger.error(f"[STDB] Failed to load DB: {e}")
        else:
            logger.warning(f"[STDB] No existing DB found at {self.save_path}, starting fresh.")

    def save(self):
        """保存到磁盘"""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(self.db, f, indent=2, ensure_ascii=False)
            logger.info(f"[STDB] Saved database to {self.save_path}")
        except Exception as e:
            logger.error(f"[STDB] Failed to save DB: {e}")

    def get_best_sequence(self, prompt: str) -> List[str]:
        """
        根据 Prompt 获取最佳动作序列。
        """
        # 简单清洗 Key，确保匹配稳定性
        key = str(prompt).strip()
        return self.db.get(key, [])

    def add_online_trajectories(self, trajectories_input: Union[List[Dict], Dict[str, List[Dict]]]):
        """
        接收在线数据并更新数据库。
        支持两种输入格式：
        1. List[SummaryDict]: [{'prompt':..., 'actions':..., 'success':...}] (旧格式)
        2. Dict[UID, List[StepDict]]: {uid: [step1, step2...]} (新格式，来自 dp_actor)
        """
        updated = False
        
        # 1. 统一格式为 List[List[Step]] 或 List[Summary]
        trajectory_iter = []
        if isinstance(trajectories_input, dict):
            trajectory_iter = list(trajectories_input.values())
        elif isinstance(trajectories_input, list):
            trajectory_iter = trajectories_input
        else:
            logger.warning(f"[STDB] Unknown input format: {type(trajectories_input)}")
            return

        for traj_item in trajectory_iter:
            prompt_str = None
            new_actions = []
            is_success = False

            # --- 情况 A: 输入是步骤列表 (来自 dp_actor) ---
            if isinstance(traj_item, list) and len(traj_item) > 0:
                # 假设所有步骤共享轨迹级信息
                first_step = traj_item[0]
                
                # 1. 检查成功状态
                # rollout_loop.py 会在 gather 阶段广播 traj_task_completed
                is_success = first_step.get('traj_task_completed', False)
                # 或者检查最后一个步骤的 info
                # last_info = traj_item[-1]
                # is_success = last_info.get('action_success', False) # 这通常只是最后一步是否成功，不是整个任务

                if not is_success:
                    continue

                # 2. 提取 Prompt
                # 优先使用 raw_prompt
                if 'raw_prompt' in first_step:
                    prompt_str = str(first_step['raw_prompt']).strip()
                # 其次使用 prompt_index (如果是字符串)
                elif 'prompt_index' in first_step:
                    prompt_str = str(first_step['prompt_index']).strip()
                
                # 3. 提取动作序列
                # 依赖于 env_manager.py 中添加的 'executed_action_str'
                new_actions = [s.get('executed_action_str', '') for s in traj_item]
                # 过滤空动作
                new_actions = [a for a in new_actions if a]

            # --- 情况 B: 输入是摘要字典 (测试或旧代码) ---
            elif isinstance(traj_item, dict):
                if not traj_item.get('success', False):
                    continue
                prompt_str = str(traj_item.get('prompt', '')).strip()
                new_actions = traj_item.get('actions', [])

            # --- 执行更新 ---
            if prompt_str and new_actions:
                existing = self.db.get(prompt_str)
                
                # 贪婪策略：如果当前没有记录，或者新轨迹更短，则更新
                # (ALFWorld 通常越短越好)
                if existing is None or len(new_actions) < len(existing):
                    self.db[prompt_str] = new_actions
                    updated = True
                    logger.info(f"[STDB] New Best Traj for prompt '{prompt_str[:30]}...': {len(new_actions)} steps")
        
        if updated:
            self.save()

    def get_buffer_trajectories(self, online_batch_list: List[Dict]) -> List[Dict]:
        """
        根据当前的 Online Batch (Prompt)，从 STDB 中检索对应的成功轨迹。
        并将其构造为 DataProto 兼容的字典格式 (包含 input_ids, attention_mask 等)。
        """
        buffer_list = []
        
        if self.tokenizer is None:
            # 只有在非 None 时警告，避免初始化时的噪音
            # logger.warning("[STDB] Tokenizer is None. Cannot reconstruct buffer trajectories.")
            return buffer_list

        for item in online_batch_list:
            # 1. 获取 Raw Prompt
            raw_prompt = item.get('raw_prompt')
            if not raw_prompt:
                continue
            
            raw_prompt_str = str(raw_prompt).strip()

            # 2. 检索最佳动作序列
            best_actions = self.get_best_sequence(raw_prompt_str)
            
            if best_actions:
                # 3. 构造 Buffer 样本
                # 我们复用 Online 样本的 prompt input_ids，但替换 responses
                
                # ALFWorld 动作之间通常用换行符分隔 (取决于你的 Env/Prompt 模板)
                # 注意：这里简单的 join 可能需要根据你的 Chat Template 调整
                # 更好的方式是像 rollout_loop 那样重新 tokenize 整个对话，但这里做个近似
                response_str = "\n".join(best_actions)
                
                # 编码 Response
                response_ids = self.tokenizer.encode(response_str, add_special_tokens=False)
                
                # 转换为 Tensor 
                response_tensor = torch.tensor(response_ids, dtype=torch.long)
                
                # 获取 Prompt Tensor (复用)
                prompt_tensor = item['input_ids']
                if not isinstance(prompt_tensor, torch.Tensor):
                    prompt_tensor = torch.tensor(prompt_tensor, dtype=torch.long)
                
                # 构造 Attention Mask
                prompt_mask = item['attention_mask']
                if not isinstance(prompt_mask, torch.Tensor):
                    prompt_mask = torch.tensor(prompt_mask, dtype=torch.long)
                
                response_mask = torch.ones_like(response_tensor)
                
                # 构造返回字典
                buffer_item = {
                    'prompts': prompt_tensor,
                    'responses': response_tensor,
                    'input_ids': torch.cat([prompt_tensor, response_tensor], dim=0),
                    'attention_mask': torch.cat([prompt_mask, response_mask], dim=0),
                    'position_ids': torch.arange(len(prompt_tensor) + len(response_tensor)),
                    
                    # 标记这是 Buffer 数据
                    'is_buffer_data': True,
                    'prompt_index': item.get('prompt_index'), 
                    'raw_prompt': raw_prompt
                }
                
                buffer_list.append(buffer_item)
        
        return buffer_list