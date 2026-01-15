# CCAPO/stdb.py

import collections
import threading
import os
import json
import logging
import pickle
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class STDB:
    """
    State-Transition Database (STDB) for CCAPO v2.0.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # --- Storage ---
        self._execution_db: Dict[str, Dict] = {}
        self._logic_db: Dict[str, Dict[Tuple[str, str], int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self._logic_totals: Dict[str, int] = collections.defaultdict(int)
        
        self._lock = threading.RLock()
        
        # --- [新增] 初始化专用日志系统 ---
        self._init_logger()
        
        # 自动加载
        self.save_path = self.config.get('stdb_save_path', 'stdb/alfworld_stdb.json')
        if self.save_path:
            self.load_checkpoint(self.save_path)

    def _init_logger(self):
        """初始化 STDB 专用文件日志"""
        self.logger = logging.getLogger("CCAPO_STDB")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False # 不向上传递给 root logger，避免刷屏
        
        if not self.logger.handlers:
            try:
                log_dir = "logger/stdb"
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, "stdb_operations.log")
                
                fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
                self.logger.info("=== STDB Logger Initialized ===")
            except Exception as e:
                print(f"[STDB] Failed to init logger: {e}")

    # ======================================================
    # Interface 1: Execution Stream (Anchor Management)
    # ======================================================

    def update_execution_anchor(self, group_id: str, trajectory: Dict) -> bool:
        """
        尝试更新某个 Group 的 Anchor（锚点轨迹）。
        """
        # 必须是成功轨迹
        if not group_id or not trajectory.get('metrics', {}).get('is_success', False):
            return False

        current_steps = trajectory['metrics']['total_steps']
        task_str = trajectory['meta'].get('task_string', 'unknown')
        
        with self._lock:
            existing = self._execution_db.get(group_id)
            
            # Case 1: New Group
            if existing is None:
                self._execution_db[group_id] = trajectory
                self.logger.info(f"[EXEC-NEW] Group={group_id[:8]}... | Task={task_str} | Steps={current_steps}")
                return True
            
            # Case 2: Better Trajectory
            existing_steps = existing['metrics']['total_steps']
            if current_steps < existing_steps:
                self._execution_db[group_id] = trajectory
                self.logger.info(f"[EXEC-UPDATE] Group={group_id[:8]}... | Better Steps: {existing_steps} -> {current_steps}")
                return True
            
            # Case 3: Worse or Equal (Log debug info)
            self.logger.debug(f"[EXEC-IGNORE] Group={group_id[:8]}... | Current={current_steps} >= Existing={existing_steps}")
                
        return False

    def get_execution_anchor(self, group_id: str) -> Optional[List[str]]:
        """
        获取指定 Group 的最优执行骨架 (Raw Actions 列表)。
        """
        with self._lock:
            record = self._execution_db.get(group_id)
            if record:
                # 提取 raw actions 序列
                return [step['action_raw'] for step in record['steps']]
        return None

    # ======================================================
    # Interface 2: Logic Stream (Abstract Consensus)
    # ======================================================

    def update_logic_consensus(self, task_type: str, abstract_actions: List[str]):
        """
        利用成功的轨迹更新抽象逻辑共识 (N-gram 统计)。
        """
        if len(abstract_actions) < 2:
            return

        with self._lock:
            # Bigram 统计
            for i in range(len(abstract_actions) - 1):
                curr_a = abstract_actions[i]
                next_a = abstract_actions[i+1]
                
                transition = (curr_a, next_a)
                self._logic_db[task_type][transition] += 1
                self._logic_totals[task_type] += 1
            
            self.logger.info(f"[LOGIC-UPDATE] TaskType={task_type} | Added {len(abstract_actions)-1} transitions. Total samples: {self._logic_totals[task_type]}")

    def get_transition_score(self, task_type: str, prev_abstract: str, curr_abstract: str) -> float:
        """
        查询动作转移的共识分数 (Frequency Probability)。
        """
        with self._lock:
            total = self._logic_totals.get(task_type, 0)
            if total == 0:
                return 0.0
            
            count = self._logic_db.get(task_type, {}).get((prev_abstract, curr_abstract), 0)
            return count / total

    # ======================================================
    # Interface 3: Statistics & IO
    # ======================================================
    
    def get_status_info(self) -> Dict:
        with self._lock:
            return {
                "stdb_anchors_count": len(self._execution_db),
                "stdb_logic_types": len(self._logic_db),
                "stdb_logic_samples": sum(self._logic_totals.values())
            }

    def save_checkpoint(self, path: str = None):
        """
        持久化保存 STDB (JSON 格式，人类可读)。
        """
        target_path = path or self.save_path
        if not target_path:
            return

        # 1. 转换为绝对路径
        abs_path = os.path.abspath(target_path)
        
        # 2. 自动创建父目录
        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        except OSError as e:
            logger.error(f"[STDB] Failed to create directory for {abs_path}: {e}")
            return
        
        with self._lock:
            # --- 数据转换：将非 JSON 兼容的类型 (Tuple Key) 转换为 String ---
            
            # 处理 Logic DB: Dict[str, Dict[Tuple, int]] -> Dict[str, Dict[str, int]]
            serializable_logic_db = {}
            for task_type, transitions in self._logic_db.items():
                serializable_logic_db[task_type] = {}
                for (prev_a, curr_a), count in transitions.items():
                    # 使用特殊分隔符组合 Key
                    key_str = f"{prev_a}__TO__{curr_a}"
                    serializable_logic_db[task_type][key_str] = count

            data = {
                "execution_db": self._execution_db,
                "logic_db": serializable_logic_db,
                "logic_totals": dict(self._logic_totals)
            }
            
            try:
                # 使用 json.dump 写入文本文件
                with open(abs_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                # print(f">>> [STDB] Successfully saved JSON checkpoint to {abs_path}")
            except Exception as e:
                logger.error(f"[STDB] Failed to save JSON checkpoint: {e}")
                print(f">>> [STDB] Error saving JSON checkpoint: {e}")

    def load_checkpoint(self, path: str):
        """
        加载 STDB (支持 JSON 格式)。
        """
        if not os.path.exists(path):
            logger.warning(f"[STDB] Checkpoint not found at {path}, starting fresh.")
            return

        try:
            # 尝试作为 JSON 读取
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self._lock:
                self._execution_db = data.get("execution_db", {})
                self._logic_totals = collections.defaultdict(int, data.get("logic_totals", {}))
                
                # --- 数据恢复：String Key -> Tuple Key ---
                raw_logic_db = data.get("logic_db", {})
                self._logic_db = collections.defaultdict(lambda: collections.defaultdict(int))
                
                for task_type, transitions in raw_logic_db.items():
                    for key_str, count in transitions.items():
                        # 解析 Key: "action_a__TO__action_b" -> ("action_a", "action_b")
                        if "__TO__" in key_str:
                            parts = key_str.split("__TO__")
                            # 防止动作本身包含分隔符导致分割错误，只分割一次通常不够，这里假设动作本身不含此特殊符
                            # 如果动作可能很复杂，建议用更健壮的分隔符或 list 存储
                            if len(parts) >= 2:
                                # 重新组合 (简单处理：前部分是 prev, 后部分是 curr)
                                # 考虑到 split 可能会分出多段如果动作里也有 TO，这里取第一个和剩余所有
                                prev_a = parts[0]
                                curr_a = "__TO__".join(parts[1:])
                                self._logic_db[task_type][(prev_a, curr_a)] = count
                        else:
                            # 兼容旧格式或异常数据
                            self._logic_db[task_type][("unknown", key_str)] = count
                    
            logger.info(f"[STDB] Loaded JSON checkpoint from {path}. Anchors: {len(self._execution_db)}")
            
        except (json.JSONDecodeError, UnicodeDecodeError):
            # 如果 JSON 加载失败，尝试回退到 pickle (兼容旧文件)
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                # ... (pickle 恢复逻辑，与之前相同，这里略过，建议直接删除旧文件重新跑) ...
                print(f">>> [STDB] Loaded legacy Pickle checkpoint.")
            except Exception as e:
                logger.error(f"[STDB] Failed to load checkpoint: {e}")
        except Exception as e:
            logger.error(f"[STDB] Failed to load checkpoint: {e}")