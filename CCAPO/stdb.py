# CCAPO/stdb.py

import ray
import collections
import threading
import os
import json
import logging
import pickle
import time
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

def append_log(filepath, data):
    """简单的独立日志写入函数，避免依赖复杂的 Logger 类"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            entry = {"timestamp": time.time(), **data}
            f.write(json.dumps(entry) + "\n")
    except:
        pass

# =========================================================
# 1. Server Side: The Ray Actor (Global Source of Truth)
# =========================================================

@ray.remote(num_cpus=0.1) # 占用极少资源，作为一个后台服务运行
class STDBServer:
    def __init__(self, config: Dict):
        self.config = config or {}
        
        # --- Storage ---
        self._execution_db: Dict[str, Dict] = {}
        # 注意：为了序列化安全，这里不用 defaultdict，用普通 dict
        self._logic_db: Dict[str, Dict[str, int]] = {}
        self._logic_totals: Dict[str, int] = {}
        
        # 日志配置
        self._init_logger()
        
        # 自动加载
        self.save_path = self.config.get('stdb_save_path', 'stdb/alfworld_stdb.json')
        if self.save_path:
            self.load_checkpoint(self.save_path)
        self.log_file = config.get('stdb_log_path', 'experiments/ccapo_logs/stdb_server.jsonl')

    def _init_logger(self):
        # 简单的服务器端日志
        print("[STDB-Server] Service Started.")

    def ping(self):
        return "pong"

    # --- Execution Stream Logic ---

    def update_execution_anchor(self, group_id: str, trajectory: Dict) -> bool:
        """Server端更新逻辑：比较分数和步数"""
        current_steps = trajectory['metrics']['total_steps']
        # 这里假设 metric 中有 final_env_reward，如果没有默认为 1.0 (因为只传入成功轨迹)
        current_score = trajectory['metrics'].get('final_env_reward', 1.0)
        
        existing = self._execution_db.get(group_id)
        
        should_update = False
        if existing is None:
            should_update = True
            print(f"[STDB-Server] New Anchor | Group={group_id[:6]} | Steps={current_steps}")
        else:
            prev_steps = existing['metrics']['total_steps']
            # 简单策略：步数更少则更新
            if current_steps < prev_steps:
                should_update = True
                print(f"[STDB-Server] Better Anchor | Group={group_id[:6]} | Steps {prev_steps}->{current_steps}")
                
        if should_update:
            self._execution_db[group_id] = trajectory
            
            # [埋点] 记录 Anchor 更新
            append_log(self.log_file, {
                "event": "anchor_update",
                "group_id": group_id,
                "steps": current_steps,
                "score": current_score,
                "is_new": existing is None
            })
            return True
        return False

    def get_execution_anchor(self, group_id: str) -> Optional[List[str]]:
        record = self._execution_db.get(group_id)
        if record:
            return [step['action_raw'] for step in record['steps']]
        return None

    # --- Logic Stream Logic ---

    def update_logic_consensus(self, task_type: str, transitions: List[Tuple[str, str]]):
        """批量更新逻辑共识"""
        if task_type not in self._logic_db:
            self._logic_db[task_type] = {}
            self._logic_totals[task_type] = 0
            
        for prev, curr in transitions:
            key = f"{prev}__TO__{curr}"
            self._logic_db[task_type][key] = self._logic_db[task_type].get(key, 0) + 1
            self._logic_totals[task_type] += 1

        if self._logic_totals[task_type] % 10 == 0:
            append_log(self.log_file, {
                "event": "logic_update",
                "task_type": task_type,
                "total_samples": self._logic_totals[task_type]
            })

    def get_transition_score(self, task_type: str, prev_abstract: str, curr_abstract: str) -> float:
        total = self._logic_totals.get(task_type, 0)
        if total == 0: return 0.0
        
        key = f"{prev_abstract}__TO__{curr_abstract}"
        count = self._logic_db.get(task_type, {}).get(key, 0)
        return count / total

    # --- IO Logic ---

    def save_checkpoint(self, path: str = None):
        target_path = path or self.save_path
        if not target_path: return
        
        try:
            abs_path = os.path.abspath(target_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            
            data = {
                "execution_db": self._execution_db,
                "logic_db": self._logic_db,
                "logic_totals": self._logic_totals
            }
            with open(abs_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[STDB-Server] Checkpoint saved to {abs_path}")
        except Exception as e:
            print(f"[STDB-Server] Save failed: {e}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path): return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._execution_db = data.get("execution_db", {})
            self._logic_db = data.get("logic_db", {})
            self._logic_totals = data.get("logic_totals", {})
            print(f"[STDB-Server] Loaded {len(self._execution_db)} anchors.")
        except Exception as e:
            print(f"[STDB-Server] Load failed (trying pickle fallback...): {e}")
            # Pickle fallback support removed for simplicity in Server version to rely on pure JSON

# =========================================================
# 2. Client Side: The Wrapper (Used by RewardManager)
# =========================================================

class STDBClient:
    """
    本地代理，负责与全局唯一的 STDB Server 通信。
    接口保持与原 STDB 类一致，确保 RewardManager 无需大幅修改。
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.save_path = self.config.get('stdb_save_path', 'stdb/alfworld_stdb.json')
        self.actor_name = "Global_CCAPO_STDB_v2"
        
        self._actor = self._get_or_create_actor()
        
    def _get_or_create_actor(self):
        """确保拿到全局唯一的 Actor，如果不存在则创建"""
        try:
            return ray.get_actor(self.actor_name)
        except ValueError:
            print(f"[STDB-Client] Global Actor not found. Creating new one: {self.actor_name}")
            # lifetime="detached" 保证 Actor 在 Driver 退出后（或者 Ray Client 断开后）依然存活，
            # 直到被显式 kill 或者 Ray 集群关闭。这对于多任务/多阶段训练很有用。
            try:
                actor = STDBServer.options(name=self.actor_name, lifetime="detached").remote(self.config)
                # 等待初始化完成
                ray.get(actor.ping.remote())
                return actor
            except Exception as e:
                # 处理并发创建时的 Race Condition
                print(f"[STDB-Client] Creation race detected, trying to get actor again: {e}")
                time.sleep(1)
                return ray.get_actor(self.actor_name)

    # --- 代理方法 (Interface Proxy) ---

    def update_execution_anchor(self, group_id: str, trajectory: Dict) -> bool:
        # 同步调用 (ray.get)，因为我们需要知道是否更新成功来打印日志
        return ray.get(self._actor.update_execution_anchor.remote(group_id, trajectory))

    def get_execution_anchor(self, group_id: str) -> Optional[List[str]]:
        return ray.get(self._actor.get_execution_anchor.remote(group_id))

    def update_logic_consensus(self, task_type: str, abstract_actions: List[str]):
        if len(abstract_actions) < 2: return
        # 预处理数据，减少 Server 负担
        transitions = []
        for i in range(len(abstract_actions) - 1):
            transitions.append((abstract_actions[i], abstract_actions[i+1]))
        
        # 异步调用 (Fire and Forget)，不等待结果，提高训练吞吐量
        self._actor.update_logic_consensus.remote(task_type, transitions)

    def get_transition_score(self, task_type: str, prev_abstract: str, curr_abstract: str) -> float:
        return ray.get(self._actor.get_transition_score.remote(task_type, prev_abstract, curr_abstract))

    def save_checkpoint(self, path: str = None):
        # 异步触发保存
        self._actor.save_checkpoint.remote(path)

# 为了兼容旧代码的 import 方式，将 STDB 指向 Client
STDB = STDBClient