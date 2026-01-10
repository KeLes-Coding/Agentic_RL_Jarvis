# agent_system/reward_manager/stdb.py

import os
import json
import logging
import re
import collections
from typing import List, Dict, Optional, Any, Union

logger = logging.getLogger(__name__)

class SuccessTrajectoryDatabase:
    """
    CCAPO v2.0 STDB for ALFWorld adaptation.
    维护两个层级的数据库：
    1. Exact DB (stdb_exact.json): {Prompt_Hash: Concrete_Action_Sequence}
       - 用于 Stream A (Memory/Warm Start)
       - 存储特定 Seed 下的最优解
    2. Abstract DB (stdb_abstract.json): {Task_Type: {Bigram: Frequency}}
       - 用于 Stream B (Generalization/Logic)
       - 存储特定任务类型的通用逻辑流
    """
    def __init__(self, save_path: str, top_k: int = 1, tokenizer: Any = None):
        # --- 🔥 [修复] 保留 save_path 属性以兼容 ray_trainer.py 的调用 ---
        self.save_path = save_path 
        # -------------------------------------------------------------

        # 拆分保存路径
        self.save_dir = os.path.dirname(save_path)
        self.exact_path = os.path.join(self.save_dir, "stdb_exact.json")
        self.abstract_path = os.path.join(self.save_dir, "stdb_abstract.json")
        
        self.top_k = top_k
        self.tokenizer = tokenizer
        
        # Data Structures
        self.exact_db: Dict[str, List[str]] = {} 
        # Abstract DB: Task_Type -> { "open fridge->put apple": 10, ... }
        self.abstract_db: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        
        self.load()

    def get_abstract_id(self, action_str: str) -> str:
        """
        提取动作的抽象指纹 (Abstract ID)。
        ALFWorld 示例: "put apple 1 in fridge 2" -> "put apple in fridge"
        """
        if not action_str: return ""
        # Regex: 去除数字后缀 (e.g., " 1", " 2")
        s = re.sub(r'\s\d+', '', str(action_str))
        return s.strip().lower()

    def load(self):
        """加载双层数据库"""
        # 1. Load Exact DB
        if os.path.exists(self.exact_path):
            try:
                with open(self.exact_path, 'r', encoding='utf-8') as f:
                    self.exact_db = json.load(f)
                logger.info(f"[STDB] Loaded Exact DB: {len(self.exact_db)} entries")
            except Exception as e:
                logger.error(f"[STDB] Failed to load Exact DB: {e}")
        
        # 2. Load Abstract DB
        if os.path.exists(self.abstract_path):
            try:
                with open(self.abstract_path, 'r', encoding='utf-8') as f:
                    self.abstract_db = json.load(f)
                    # 恢复 defaultdict 行为 (json加载后是普通dict)
                    temp_db = collections.defaultdict(lambda: collections.defaultdict(int))
                    for k, v in self.abstract_db.items():
                        temp_db[k].update(v)
                    self.abstract_db = temp_db
                logger.info(f"[STDB] Loaded Abstract DB: {len(self.abstract_db)} task types")
            except Exception as e:
                logger.error(f"[STDB] Failed to load Abstract DB: {e}")

    def save(self):
        """保存双层数据库"""
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            # Save Exact
            with open(self.exact_path, 'w', encoding='utf-8') as f:
                json.dump(self.exact_db, f, indent=2, ensure_ascii=False)
            # Save Abstract (convert defaultdict to dict)
            with open(self.abstract_path, 'w', encoding='utf-8') as f:
                dumpable = {k: dict(v) for k, v in self.abstract_db.items()}
                json.dump(dumpable, f, indent=2, ensure_ascii=False)
            logger.info("[STDB] Saved databases.")
        except Exception as e:
            logger.error(f"[STDB] Failed to save DB: {e}")

    def get_best_sequence(self, prompt: str) -> List[str]:
        """兼容旧接口，返回 Exact Sequence"""
        key = str(prompt).strip()
        return self.exact_db.get(key, [])

    def get_abstract_consensus(self, task_type: str) -> Dict[str, int]:
        """获取特定任务类型的逻辑共识库"""
        return self.abstract_db.get(task_type, {})

    def add_online_trajectories(self, trajectories_input: Union[List[Dict], Dict[str, List[Dict]]]):
        """
        更新策略 (CCAPO v2.0):
        1. 更新 Exact DB: 如果是特定 Prompt/Seed 的更优解。
        2. 更新 Abstract DB: 只要成功，就将其抽象逻辑链计入统计。
        """
        updated = False
        
        # Flatten input
        trajectory_iter = []
        if isinstance(trajectories_input, dict):
            trajectory_iter = list(trajectories_input.values())
        elif isinstance(trajectories_input, list):
            trajectory_iter = trajectories_input
        else:
            return

        for traj_item in trajectory_iter:
            concrete_actions = []
            prompt_str = None
            task_type = "unknown"
            is_success = False

            # --- 解析数据 ---
            if isinstance(traj_item, list) and len(traj_item) > 0:
                # 来自 dp_actor 的 List[StepDict]
                first_step = traj_item[0]
                is_success = first_step.get('traj_task_completed', False)
                if not is_success: continue
                
                # Key: 使用 raw_prompt 或 prompt_index 作为 Exact DB 的 Key
                prompt_str = str(first_step.get('prompt_index', '')).strip()
                if not prompt_str and 'raw_prompt' in first_step:
                    prompt_str = str(first_step['raw_prompt']).strip()
                
                task_type = first_step.get('task_type', 'unknown')
                
                # Actions
                concrete_actions = [s.get('executed_action_str', '') for s in traj_item]
                concrete_actions = [a for a in concrete_actions if a]
            
            elif isinstance(traj_item, dict):
                # 兼容旧格式
                if not traj_item.get('success', False): continue
                prompt_str = str(traj_item.get('prompt', '')).strip()
                task_type = traj_item.get('task_type', 'unknown')
                concrete_actions = traj_item.get('actions', [])
            
            if not concrete_actions or not prompt_str: continue

            # --- 1. 更新 Exact DB (Memory) ---
            # 贪婪策略：更短的路径更好
            existing = self.exact_db.get(prompt_str)
            if existing is None or len(concrete_actions) < len(existing):
                self.exact_db[prompt_str] = concrete_actions
                updated = True
                logger.info(f"[STDB] New Best Exact for '{prompt_str[:20]}...': {len(concrete_actions)} steps")
            
            # --- 2. 更新 Abstract DB (Generalization) ---
            if task_type and task_type != 'unknown':
                # 转为抽象序列
                abstract_seq = [self.get_abstract_id(a) for a in concrete_actions]
                # 提取 Bi-grams 并更新频率
                for i in range(len(abstract_seq) - 1):
                    # 格式: "prev_action -> curr_action"
                    bigram = f"{abstract_seq[i]}->{abstract_seq[i+1]}"
                    self.abstract_db[task_type][bigram] += 1
                    updated = True

        if updated:
            self.save()

    def get_buffer_trajectories(self, online_batch_list: List[Dict]) -> List[Dict]:
        """
        (保持不变，主要用于 Experience Replay)
        从 Exact DB 中恢复数据用于训练。
        """
        # ... (保留你原有的实现即可) ...
        return []