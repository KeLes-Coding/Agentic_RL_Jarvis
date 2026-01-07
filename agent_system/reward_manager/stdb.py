# agent_system/reward_manager/stdb.py

import os
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class SuccessTrajectoryDatabase:
    """
    轻量级 STDB (Lite Version for ALFWorld)
    仅在内存中维护 {Prompt_ID: [Best_Action_Sequence]} 的映射。
    """
    def __init__(self, save_path: str, top_k: int = 1):
        self.save_path = save_path
        self.top_k = top_k
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
        这里使用简单的精确匹配或 Hash 匹配。
        """
        # 简单清洗 Key
        key = str(prompt).strip()
        return self.db.get(key, [])

    def add_online_trajectories(self, trajectories: List[Dict]):
        """
        接收在线数据并更新数据库。
        trajectories: List of dict, each containing {'prompt': str, 'actions': List[str], 'success': bool, 'steps': int}
        """
        updated = False
        for traj in trajectories:
            if not traj.get('success', False):
                continue
            
            prompt = str(traj['prompt']).strip()
            new_actions = traj['actions']
            
            # 简单的贪婪策略：如果更短，就替换
            existing = self.db.get(prompt)
            
            if existing is None or len(new_actions) < len(existing):
                self.db[prompt] = new_actions
                updated = True
                logger.info(f"[STDB] Update Best Traj for prompt '{prompt[:20]}...': {len(new_actions)} steps")
        
        if updated:
            self.save()