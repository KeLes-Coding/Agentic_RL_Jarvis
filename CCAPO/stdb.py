# CCAPO/stdb.py

import ray
import os
import json
import math
import time
from typing import List, Dict, Optional, Tuple, Any

# =========================================================
# 1. Server Side: The Ray Actor (Global Source of Truth)
# =========================================================

@ray.remote(num_cpus=0.1)
class STDBServer:
    def __init__(self, config: Dict):
        self.config = config or {}
        
        # --- Storage v2.3: Probabilistic Logic Graph ---
        self._execution_db: Dict[str, Dict] = {} # Key: GroupID, Value: Full Trajectory
        
        # [v2.3 Upgrade] 逻辑图谱存储结构
        # Structure: 
        # {
        #   "task_type": {
        #       "pre_action_hash": {
        #           "post_action_hash": {
        #               "count": int,          # 总出现次数 (Importance)
        #               "gap_sum": int,        # 距离总和 (用于计算 Avg Gap -> Utility)
        #               "success_cnt": int,    # 成功轨迹中出现的次数
        #               "fail_cnt": int,       # 失败轨迹中出现的次数 (Criticality)
        #               "last_updated": float  # 时间戳
        #           }
        #       }
        #   }
        # }
        self._logic_graph: Dict[str, Dict[str, Dict[str, Dict]]] = {}
        self._logic_totals: Dict[str, int] = {} # 记录每个任务类型的总样本数
        
        # 配置参数
        self.skip_gram_window = self.config.get('skip_gram_window', 3)  # v2.3 窗口大小 K
        self.metric_lambda = self.config.get('metric_lambda', 1.0)      # Criticality 权重
        self.metric_alpha = self.config.get('metric_alpha', 0.5)        # Utility 衰减指数
        
        # 自动加载
        self.save_path = self.config.get('stdb_save_path', 'stdb/alfworld_stdb.json')
        self.log_file = self.config.get('stdb_log_path', 'experiments/ccapo_logs/stdb_server.jsonl')
        
        if self.save_path:
            self.load_checkpoint(self.save_path)
            
        self._init_logger()

    def _init_logger(self):
        print(f"[STDB-Server] Service Started. Mode: Probabilistic Logic Graph (Window={self.skip_gram_window})")

    def ping(self):
        return "pong"

    # --- Execution Stream Logic (保持 v1 稳定逻辑) ---

    def update_execution_anchor(self, group_id: str, trajectory: Dict) -> bool:
        """
        Server端更新逻辑：比较分数和步数。
        Execution Stream 负责“组内强收敛”。
        """
        current_steps = trajectory['metrics']['total_steps']
        current_score = trajectory['metrics'].get('final_env_reward', 1.0)
        
        existing = self._execution_db.get(group_id)
        should_update = False
        
        if existing is None:
            should_update = True
        else:
            prev_steps = existing['metrics']['total_steps']
            # 简单策略：步数更少则更新 (后续可加入 Score 比较)
            if current_steps < prev_steps:
                should_update = True
                
        if should_update:
            self._execution_db[group_id] = trajectory
            # [埋点]
            self._log_event("anchor_update", {
                "group_id": group_id, 
                "steps": current_steps, 
                "is_new": existing is None
            })
            return True
        return False

    def get_execution_anchor(self, group_id: str) -> Optional[List[str]]:
        record = self._execution_db.get(group_id)
        if record:
            return [step['action_raw'] for step in record['steps']]
        return None

    # --- Logic Stream Logic (v2.3 Core: Skip-Gram & Graph) ---

    def update_logic_skip_gram(self, task_type: str, abstract_actions: List[str], is_success: bool = True):
        """
        [v2.3 Upgrade] 使用 Skip-Gram 方式更新概率图谱。
        不仅记录 (t, t+1)，还记录 (t, t+k) 以捕捉长程因果。
        """
        if len(abstract_actions) < 2: return
        
        if task_type not in self._logic_graph:
            self._logic_graph[task_type] = {}
            self._logic_totals[task_type] = 0
            
        # 仅对成功轨迹增加 Total 计数 (作为 Importance 的分母)
        if is_success:
            self._logic_totals[task_type] += 1
            
        # 遍历轨迹
        L = len(abstract_actions)
        # 限制窗口不超过轨迹长度
        actual_window = min(self.skip_gram_window, L)
        
        for i in range(L - 1):
            pre_node = abstract_actions[i]
            if not pre_node: continue
            
            if pre_node not in self._logic_graph[task_type]:
                self._logic_graph[task_type][pre_node] = {}
                
            # Skip-Gram Window
            for k in range(1, actual_window + 1):
                if i + k >= L: break
                
                post_node = abstract_actions[i + k]
                if not post_node: continue
                
                # 获取或初始化 Edge 数据
                if post_node not in self._logic_graph[task_type][pre_node]:
                    self._logic_graph[task_type][pre_node][post_node] = {
                        "count": 0,
                        "gap_sum": 0,
                        "success_cnt": 0,
                        "fail_cnt": 0,
                        "last_updated": 0
                    }
                
                edge_data = self._logic_graph[task_type][pre_node][post_node]
                
                # 更新统计量
                edge_data["count"] += 1
                edge_data["last_updated"] = time.time()
                
                if is_success:
                    edge_data["success_cnt"] += 1
                    edge_data["gap_sum"] += k  # 记录距离 (Gap=1 表示相邻)
                else:
                    edge_data["fail_cnt"] += 1

    def get_transition_score(self, task_type: str, prev_abstract: str, curr_abstract: str) -> float:
        """
        [v2.3 Upgrade] 乘性门控评分公式
        Score(E) = I(E) * (1 + lambda * C(E)) * U(E)
        """
        if task_type not in self._logic_graph: return 0.0
        
        total_success_samples = self._logic_totals.get(task_type, 1)
        if total_success_samples < 1: total_success_samples = 1
        
        # 查找边
        edges = self._logic_graph[task_type].get(prev_abstract, {})
        edge_data = edges.get(curr_abstract)
        
        if not edge_data:
            return 0.0
            
        # 1. Importance (I): 基础信任度 (频率)
        # 使用 success_cnt 而非 total count，保证逻辑来源于成功经验
        success_cnt = edge_data.get("success_cnt", 0)
        if success_cnt == 0: return 0.0 # 一票否决
        
        I_score = success_cnt / total_success_samples
        
        # 2. Utility (U): 距离衰减
        # 逻辑越紧凑 (gap 越小)，Utility 越高
        # Avg Gap = gap_sum / success_cnt
        avg_gap = edge_data.get("gap_sum", 0) / success_cnt
        if avg_gap < 1.0: avg_gap = 1.0 # 理论最小值为 1
        
        U_score = 1.0 / (avg_gap ** self.metric_alpha)
        
        # 3. Criticality (C): 关键性倍率 (暂且保留接口，简单实现)
        # 如果该边在失败轨迹中很少出现，但在成功轨迹中经常出现，说明它很关键
        # 这里做一个简单的 Heuristic: Fail Ratio 低的加分
        fail_cnt = edge_data.get("fail_cnt", 0)
        
        # C_score ≈ log(Success_Rate_of_Edge / Global_Success_Rate)
        # 这里简化为：(Success / (Success + Fail))
        edge_total = success_cnt + fail_cnt
        edge_success_rate = success_cnt / edge_total
        C_score = math.tanh(self.metric_lambda * edge_success_rate)
        
        # Final Formula
        final_score = I_score * (1 + C_score) * U_score
        
        return float(final_score)

    def _log_event(self, event_type, data):
        try:
            entry = {"timestamp": time.time(), "type": event_type, "data": data}
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except:
            pass

    # --- IO Logic ---

    def save_checkpoint(self, path: str = None):
        target_path = path or self.save_path
        if not target_path: return
        
        try:
            abs_path = os.path.abspath(target_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            
            data = {
                "version": "2.3",
                "execution_db": self._execution_db,
                "logic_graph": self._logic_graph,
                "logic_totals": self._logic_totals
            }
            # 使用临时文件写入防止损坏
            tmp_path = abs_path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.rename(tmp_path, abs_path)
            
            print(f"[STDB-Server] Checkpoint saved to {abs_path}")
        except Exception as e:
            print(f"[STDB-Server] Save failed: {e}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path): return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            version = data.get("version", "1.0")
            if version == "1.0":
                print("[STDB-Server] Detected v1.0 checkpoint. Migrating to v2.3 Graph...")
                self._execution_db = data.get("execution_db", {})
                # v1 logic_db 是扁平的，这里简单丢弃或需要写迁移逻辑
                # 为了安全，这里暂时重置逻辑库，让它重新学习 (因为 v1 没有 gap 信息)
                self._logic_graph = {}
                self._logic_totals = {}
            else:
                self._execution_db = data.get("execution_db", {})
                self._logic_graph = data.get("logic_graph", {})
                self._logic_totals = data.get("logic_totals", {})
                
            print(f"[STDB-Server] Loaded. Anchors: {len(self._execution_db)}, LogicTasks: {len(self._logic_graph)}")
        except Exception as e:
            print(f"[STDB-Server] Load failed: {e}")

# =========================================================
# 2. Client Side: The Wrapper (Updated API)
# =========================================================

class STDBClient:
    """
    本地代理。
    [Change] 
    1. update_logic_consensus 现在接收完整的 list，不再是 pairs。
    2. 增加参数 update_logic_consensus(..., is_success)
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.actor_name = "Global_CCAPO_STDB_v2"
        self._actor = self._get_or_create_actor()
        
    def _get_or_create_actor(self):
        try:
            return ray.get_actor(self.actor_name)
        except ValueError:
            print(f"[STDB-Client] Global Actor not found. Creating new one: {self.actor_name}")
            try:
                actor = STDBServer.options(name=self.actor_name, lifetime="detached").remote(self.config)
                ray.get(actor.ping.remote())
                return actor
            except Exception as e:
                time.sleep(1)
                return ray.get_actor(self.actor_name)

    # --- 代理方法 (Interface Proxy) ---

    def update_execution_anchor(self, group_id: str, trajectory: Dict) -> bool:
        return ray.get(self._actor.update_execution_anchor.remote(group_id, trajectory))

    def get_execution_anchor(self, group_id: str) -> Optional[List[str]]:
        return ray.get(self._actor.get_execution_anchor.remote(group_id))

    def update_logic_consensus(self, task_type: str, abstract_actions: List[str], is_success: bool = True):
        """
        [v2.3 Update] 发送整条动作序列，让 Server 进行 Skip-Gram 挖掘
        """
        # Fire and Forget
        self._actor.update_logic_skip_gram.remote(task_type, abstract_actions, is_success)

    def get_transition_score(self, task_type: str, prev_abstract: str, curr_abstract: str) -> float:
        # 这是一个高频读操作，Ray 的开销可能会成为瓶颈。
        # 在真正的生产环境中，建议 Client 端缓存 (Local Cache with TTL)。
        # 这里先保持直连。
        return ray.get(self._actor.get_transition_score.remote(task_type, prev_abstract, curr_abstract))

    def save_checkpoint(self, path: str = None):
        self._actor.save_checkpoint.remote(path)

# 兼容旧代码 Import
STDB = STDBClient