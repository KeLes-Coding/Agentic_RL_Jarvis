# agent_system/reward_manager/stdb.py

import os
import json
import logging
import re
import collections
import numpy as np
import torch
from typing import List, Dict, Optional, Any, Union

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
                    temp_db = collections.defaultdict(lambda: collections.defaultdict(int))
                    for k, v in self.abstract_db.items():
                        temp_db[k].update(v)
                    self.abstract_db = temp_db
                stdb_file_logger.info(f"[LOAD] Abstract DB loaded: {len(self.abstract_db)} task types")
            except Exception as e:
                stdb_file_logger.error(f"[LOAD] Failed to load Abstract DB: {e}")

    def save(self):
        try:
            for p in (self.exact_path, self.abstract_path):
                parent = os.path.dirname(p)
                if parent: os.makedirs(parent, exist_ok=True)
            with open(self.exact_path, 'w', encoding='utf-8') as f:
                json.dump(self.exact_db, f, indent=2, ensure_ascii=False)
            with open(self.abstract_path, 'w', encoding='utf-8') as f:
                dumpable = {k: dict(v) for k, v in self.abstract_db.items()}
                json.dump(dumpable, f, indent=2, ensure_ascii=False)
            stdb_file_logger.info("[SAVE] Databases saved successfully.")
        except Exception as e:
            stdb_file_logger.error(f"[SAVE] Failed to save DB: {e}")

    def is_success_step(self, s: Dict) -> bool:
        if 'won' in s:
            val = s['won']
            if isinstance(val, (bool, np.bool_)):
                if val: return True
            elif isinstance(val, (str,)):
                if val.lower() == 'true': return True
            elif hasattr(val, 'item'):
                try:
                    if val.item(): return True
                except: pass
        if 'traj_task_completed' in s:
            val = s['traj_task_completed']
            if isinstance(val, (bool, np.bool_)) and val: return True
            if str(val).lower() == 'true': return True
        return False

    def _extract_action_string(self, step: Dict) -> Optional[str]:
        # 优先用文本字段
        for key in ['parsed_action', 'executed_action_str', 'action', 'response_str']:
            val = step.get(key)
            if val and isinstance(val, str):
                return val.strip().lower() # 这里暂时不在此处做 XML 清洗，保持原始记录以便排查
        
        # Token 解码
        if self.tokenizer and 'responses' in step:
            token_ids = step['responses']
            if hasattr(token_ids, 'tolist'): token_ids = token_ids.tolist()
            elif isinstance(token_ids, np.ndarray): token_ids = token_ids.tolist()
            if isinstance(token_ids, list):
                valid_ids = [t for t in token_ids if isinstance(t, int) and t >= 0]
                if valid_ids:
                    try:
                        decoded = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                        if decoded and decoded.strip(): return decoded.strip().lower()
                    except Exception: pass
        return None

    def get_best_sequence(self, prompt: str) -> List[str]:
        key = str(prompt).strip()
        return self.exact_db.get(key, [])

    def get_abstract_consensus(self, task_type: str) -> Dict[str, int]:
        return self.abstract_db.get(task_type, {})

    def add_online_trajectories(self, trajectories_input: Union[List[Dict], Dict[str, List[Dict]]]):
        stdb_file_logger.info(f"[ADD] Received input batch...")
        
        trajectory_iter = []
        if isinstance(trajectories_input, dict):
            trajectory_iter = list(trajectories_input.values())
        elif isinstance(trajectories_input, list):
            trajectory_iter = trajectories_input
        else:
            return

        success_count = 0
        updated = False
        debug_fail_reasons = collections.defaultdict(int)

        for i, traj_item in enumerate(trajectory_iter):
            concrete_actions = []
            prompt_str = None
            task_type = "unknown"
            is_success = False
            
            # --- 解析 ---
            if isinstance(traj_item, list) and len(traj_item) > 0:
                for s in traj_item:
                    if self.is_success_step(s):
                        is_success = True
                        break
                
                if not is_success:
                    debug_fail_reasons['not_marked_success'] += 1
                    continue
                
                # 提取 prompt (优先使用 prompt_index 作为 Key)
                first_step = traj_item[0]
                prompt_str = str(first_step.get('prompt_index', '')).strip()
                if not prompt_str and 'raw_prompt' in first_step:
                    prompt_str = str(first_step['raw_prompt']).strip()
                
                task_type = first_step.get('task_type', 'unknown')
                
                # 提取 actions
                for s in traj_item:
                    act_str = self._extract_action_string(s)
                    if act_str: concrete_actions.append(act_str)
                
            elif isinstance(traj_item, dict):
                # 兼容旧格式
                if not traj_item.get('success', False): 
                    debug_fail_reasons['not_marked_success'] += 1
                    continue
                prompt_str = str(traj_item.get('prompt', '')).strip()
                task_type = traj_item.get('task_type', 'unknown')
                concrete_actions = traj_item.get('actions', [])
            
            if not concrete_actions:
                debug_fail_reasons['no_actions'] += 1
                continue

            success_count += 1

            # 🔥 [DEBUG-TRACE] 打印即将存入的 Key 和部分 Action
            if i < 3: # 只打印前 3 条成功样本，避免刷屏
                stdb_file_logger.info(f"[STDB-DEBUG] Saving Success Traj:")
                stdb_file_logger.info(f"  > Key (Prompt): '{prompt_str}'")
                stdb_file_logger.info(f"  > First Action: '{concrete_actions[0] if concrete_actions else 'N/A'}'")
                stdb_file_logger.info(f"  > Num Actions:  {len(concrete_actions)}")

            # --- 1. 更新 Exact DB ---
            # 只有当 key 有效时才存
            if prompt_str and prompt_str != 'None':
                existing = self.exact_db.get(prompt_str)
                if existing is None or len(concrete_actions) < len(existing):
                    self.exact_db[prompt_str] = concrete_actions
                    updated = True
                    stdb_file_logger.info(f"  -> New Best Exact for Key '{prompt_str[:30]}...': {len(concrete_actions)} steps")
            
            # --- 2. 更新 Abstract DB ---
            if task_type and task_type != 'unknown':
                abstract_seq = [self.get_abstract_id(a) for a in concrete_actions]
                for i in range(len(abstract_seq) - 1):
                    bigram = f"{abstract_seq[i]}->{abstract_seq[i+1]}"
                    self.abstract_db[task_type][bigram] += 1
                    updated = True

        stdb_file_logger.info(f"[ADD] Processed {len(trajectory_iter)} trajs. Valid Success: {success_count}. Updated: {updated}")
        if len(debug_fail_reasons) > 0:
            stdb_file_logger.warning(f"[DEBUG] Rejection Reasons: {dict(debug_fail_reasons)}")

        if updated:
            self.save()

    def get_buffer_trajectories(self, online_batch_list: List[Dict]) -> List[Dict]:
        return []