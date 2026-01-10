# agent_system/reward_manager/stdb.py

import os
import json
import logging
import re
import collections
import numpy as np
import torch
from typing import List, Dict, Optional, Any, Union

# ======================= ✅ STDB 专用日志器 ✅ =======================
logger = logging.getLogger(__name__)

stdb_file_logger = logging.getLogger("STDB_FILE")
stdb_file_logger.setLevel(logging.INFO)
stdb_file_logger.propagate = False

if not stdb_file_logger.handlers:
    try:
        log_dir = "logger/STDB"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "stdb.log"), mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        stdb_file_logger.addHandler(fh)
    except Exception as e:
        logger.error(f"[STDB] Failed to setup file logger: {e}")
# ====================================================================

class SuccessTrajectoryDatabase:
    def __init__(self, save_path: str, top_k: int = 1, tokenizer: Any = None):
        self.save_path = save_path
        self.save_dir = os.path.dirname(save_path)
        self.exact_path = os.path.join(self.save_dir, "stdb_exact.json")
        self.abstract_path = os.path.join(self.save_dir, "stdb_abstract.json")
        
        self.top_k = top_k
        self.tokenizer = tokenizer
        
        self.exact_db: Dict[str, List[str]] = {} 
        self.abstract_db: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        
        stdb_file_logger.info(f"=== STDB Initialized. Save Path: {self.save_dir} ===")
        self.load()

    def get_abstract_id(self, action_str: str) -> str:
        if not action_str: return ""
        s = re.sub(r'\s\d+', '', str(action_str))
        return s.strip().lower()

    def load(self):
        if os.path.exists(self.exact_path):
            try:
                with open(self.exact_path, 'r', encoding='utf-8') as f:
                    self.exact_db = json.load(f)
                stdb_file_logger.info(f"[LOAD] Exact DB loaded: {len(self.exact_db)} entries")
            except Exception as e:
                stdb_file_logger.error(f"[LOAD] Failed to load Exact DB: {e}")
        
        if os.path.exists(self.abstract_path):
            try:
                with open(self.abstract_path, 'r', encoding='utf-8') as f:
                    self.abstract_db = json.load(f)
                    # 恢复 defaultdict
                    temp_db = collections.defaultdict(lambda: collections.defaultdict(int))
                    for k, v in self.abstract_db.items():
                        temp_db[k].update(v)
                    self.abstract_db = temp_db
                stdb_file_logger.info(f"[LOAD] Abstract DB loaded: {len(self.abstract_db)} task types")
            except Exception as e:
                stdb_file_logger.error(f"[LOAD] Failed to load Abstract DB: {e}")

    def save(self):
        try:
            # 确保目录存在
            for p in (self.exact_path, self.abstract_path):
                parent = os.path.dirname(p)
                if parent:
                    os.makedirs(parent, exist_ok=True)

            with open(self.exact_path, 'w', encoding='utf-8') as f:
                json.dump(self.exact_db, f, indent=2, ensure_ascii=False)

            with open(self.abstract_path, 'w', encoding='utf-8') as f:
                dumpable = {k: dict(v) for k, v in self.abstract_db.items()}
                json.dump(dumpable, f, indent=2, ensure_ascii=False)

            stdb_file_logger.info("[SAVE] Databases saved successfully.")
        except Exception as e:
            stdb_file_logger.error(f"[SAVE] Failed to save DB: {e}")

    def is_success_step(self, s: Dict) -> bool:
        """Helper to safely check success in mixed types"""
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
            
        return False

    def get_best_sequence(self, prompt: str) -> List[str]:
        key = str(prompt).strip()
        return self.exact_db.get(key, [])

    def get_abstract_consensus(self, task_type: str) -> Dict[str, int]:
        return self.abstract_db.get(task_type, {})

    def add_online_trajectories(self, trajectories_input: Union[List[Dict], Dict[str, List[Dict]]]):
        stdb_file_logger.info(f"[ADD] Received input batch...")
        
        updated = False
        trajectory_iter = []
        if isinstance(trajectories_input, dict):
            trajectory_iter = list(trajectories_input.values())
        elif isinstance(trajectories_input, list):
            trajectory_iter = trajectories_input
        else:
            return

        success_count = 0
        
        for traj_item in trajectory_iter:
            concrete_actions = []
            prompt_str = None
            task_type = "unknown"
            is_success = False

            # --- 解析 ---
            if isinstance(traj_item, list) and len(traj_item) > 0:
                first_step = traj_item[0]
                
                # 🔥 [重点] 使用类型安全的检查
                if self.is_success_step(first_step):
                    is_success = True
                
                if not is_success: continue
                
                prompt_str = str(first_step.get('prompt_index', '')).strip()
                if not prompt_str and 'raw_prompt' in first_step:
                    prompt_str = str(first_step['raw_prompt']).strip()
                
                task_type = first_step.get('task_type', 'unknown')
                
                # 🔥 [重点] 提取动作逻辑
                concrete_actions = []
                for s in traj_item:
                    act = s.get('parsed_action') or s.get('executed_action_str')
                    if act:
                        concrete_actions.append(str(act).strip().lower())
            
            elif isinstance(traj_item, dict):
                # 兼容旧格式
                if not traj_item.get('success', False): continue
                prompt_str = str(traj_item.get('prompt', '')).strip()
                task_type = traj_item.get('task_type', 'unknown')
                concrete_actions = traj_item.get('actions', [])
            
            if not concrete_actions or not prompt_str: 
                continue

            success_count += 1

            # --- 1. 更新 Exact DB ---
            existing = self.exact_db.get(prompt_str)
            if existing is None or len(concrete_actions) < len(existing):
                self.exact_db[prompt_str] = concrete_actions
                updated = True
                stdb_file_logger.info(f"  -> New Best Exact for '{prompt_str[:30]}...': {len(concrete_actions)} steps")
            
            # --- 2. 更新 Abstract DB ---
            if task_type and task_type != 'unknown':
                abstract_seq = [self.get_abstract_id(a) for a in concrete_actions]
                for i in range(len(abstract_seq) - 1):
                    bigram = f"{abstract_seq[i]}->{abstract_seq[i+1]}"
                    self.abstract_db[task_type][bigram] += 1
                    updated = True

        stdb_file_logger.info(f"[ADD] Processed {len(trajectory_iter)} trajs. Valid Success: {success_count}. Updated: {updated}")

        if updated:
            self.save()

    def get_buffer_trajectories(self, online_batch_list: List[Dict]) -> List[Dict]:
        return []