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

    # ---------------- [修改] 抽象指纹生成 (白名单暴力版) ----------------
    def get_abstract_id(self, action_str: str) -> str:
        """
        Input: "<think>...</think><action>put cd 1 in safe 1</action>"
        Output: "put cd safe"
        Logic: 强制匹配 ALFWorld 动词白名单，忽略其余所有垃圾字符。
        """
        if not action_str: return "noop"
        
        # 1. 初步清洗
        s = str(action_str).lower().strip()
        # 移除 XML
        s = re.sub(r"<[^>]+>", " ", s) 
        # 移除数字
        s = re.sub(r'\s\d+', '', s)
        # 移除标点
        s = re.sub(r'[^\w\s]', '', s)
        
        # 2. 白名单提取 (这是核心，防止 <think> 混入)
        # ALFWorld 核心动词表
        valid_verbs = ["go to", "goto", "take", "pick", "put", "open", "close", "toggle", "clean", "heat", "cool", "examine", "look", "use", "inventory", "slice"]
        
        extracted_action = ""
        
        # 检查是否以动词开头
        for v in valid_verbs:
            if s.startswith(v):
                extracted_action = s
                break
        
        # 如果不以动词开头，尝试在字符串中寻找动词 (应对 "action: go to..." 情况)
        if not extracted_action:
            for v in valid_verbs:
                # 查找动词位置
                idx = s.find(v + " ") # 动词后必须跟空格，避免匹配到单词内部
                if idx != -1:
                    extracted_action = s[idx:]
                    break
                elif s == v: # 仅有一个词的情况 (inventory, look)
                    extracted_action = s
                    break
        
        # 3. 最终防线
        if not extracted_action:
            return "noop"
            
        # 再次清洗多余空格
        extracted_action = " ".join(extracted_action.split())
        
        # 如果提取出的动作依然过长 (超过40字符)，说明还是有垃圾，截断它
        if len(extracted_action) > 40:
            extracted_action = extracted_action[:40]
            
        return extracted_action.strip()

    # ---------------- [修改] add_online_trajectories (保持结构化) ----------------
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
            prompt_str = None
            task_type = "unknown"
            is_success = False
            
            # --- 1. 解析成功状态和元数据 ---
            if isinstance(traj_item, list) and len(traj_item) > 0:
                for s in traj_item:
                    if self.is_success_step(s):
                        is_success = True
                        break
                if not is_success:
                    debug_fail_reasons['not_marked_success'] += 1
                    continue
                
                first_step = traj_item[0]
                prompt_str = str(first_step.get('prompt_index', '')).strip()
                if not prompt_str and 'raw_prompt' in first_step:
                    prompt_str = str(first_step['raw_prompt']).strip()
                task_type = first_step.get('task_type', 'unknown')

            elif isinstance(traj_item, dict):
                if not traj_item.get('success', False): 
                    debug_fail_reasons['not_marked_success'] += 1
                    continue
                prompt_str = str(traj_item.get('prompt', '')).strip()
                task_type = traj_item.get('task_type', 'unknown')
                # 构造假数据以复用逻辑
                if 'actions' in traj_item:
                    traj_item = [{'step_details': {'action': a, 'thought': ''}} for a in traj_item['actions']]
                else:
                    continue

            # --- 2. 提取结构化轨迹 ---
            trajectory_steps = []
            if isinstance(traj_item, list):
                for s in traj_item:
                    # 优先读取结构化数据
                    details = s.get('step_details', {})
                    action_str = details.get('action')
                    
                    # 兼容性 Fallback
                    if not action_str:
                        action_str = self._extract_action_string(s)
                    
                    # 必须再次经过清洗，防止漏网之鱼
                    if action_str:
                        # 临时清洗一下用于 Exact 存储
                        clean_act_for_exact = action_str
                        # 如果 action_str 包含了 <think> (说明 env_manager 还是漏了)，这里切掉
                        if "<think>" in str(clean_act_for_exact):
                             clean_act_for_exact = re.sub(r"<think>.*?</think>", "", str(clean_act_for_exact), flags=re.DOTALL).strip()
                             clean_act_for_exact = re.sub(r"<.*?>", "", clean_act_for_exact).strip()

                        trajectory_steps.append({
                            "action": clean_act_for_exact, 
                            "thought": details.get('thought', ""), 
                            "raw": details.get('raw_output', "")
                        })

            if not trajectory_steps:
                debug_fail_reasons['no_actions'] += 1
                continue

            success_count += 1

            # --- 3. 更新 Exact DB ---
            if prompt_str and prompt_str != 'None':
                if prompt_str not in self.exact_db: self.exact_db[prompt_str] = []
                pool = self.exact_db[prompt_str]
                
                # 查重 (Tuple of actions)
                try:
                    curr_act_tuple = tuple(t['action'] for t in trajectory_steps)
                    existing_act_tuples = []
                    for p in pool:
                        if isinstance(p, list) and len(p) > 0 and isinstance(p[0], dict):
                            existing_act_tuples.append(tuple(step['action'] for step in p))
                        else:
                            existing_act_tuples.append(tuple()) # Bad data skip

                    if curr_act_tuple not in existing_act_tuples:
                        pool.append(trajectory_steps)
                        pool.sort(key=len)
                        self.exact_db[prompt_str] = pool[:self.top_k]
                        updated = True
                        if i < 2:
                            stdb_file_logger.info(f"  -> Updated Exact for '{prompt_str[:15]}...'. Steps: {[t['action'] for t in trajectory_steps]}")
                except Exception as e:
                    stdb_file_logger.error(f"Error updating exact: {e}")

            # --- 4. 更新 Abstract DB ---
            # Fallback task type
            if (not task_type or task_type == 'unknown') and prompt_str:
                lower_p = prompt_str.lower()
                if 'cool' in lower_p: task_type = 'cool_object'
                elif 'heat' in lower_p: task_type = 'heat_object'
                elif 'clean' in lower_p: task_type = 'clean_object'
                elif 'lamp' in lower_p or 'light' in lower_p: task_type = 'look_at_obj_in_light'
                elif 'two' in lower_p: task_type = 'pick_two_obj_and_place'
                elif 'pick' in lower_p or 'put' in lower_p: task_type = 'pick_and_place_simple'

            if task_type and task_type != 'unknown':
                # 生成抽象指纹
                abstract_ids = [self.get_abstract_id(t['action']) for t in trajectory_steps]
                # 过滤 noop
                abstract_ids = [aid for aid in abstract_ids if aid != "noop"]
                
                for k in range(len(abstract_ids) - 1):
                    bigram = f"{abstract_ids[k]}->{abstract_ids[k+1]}"
                    self.abstract_db[task_type][bigram] += 1
                    updated = True
            else:
                pass

        if updated:
            self.save()

    def get_buffer_trajectories(self, online_batch_list: List[Dict]) -> List[Dict]:
        return []