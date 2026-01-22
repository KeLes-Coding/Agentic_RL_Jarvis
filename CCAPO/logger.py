# CCAPO/logger.py

import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional

class CCAPOLogger:
    _instance = None
    _lock = threading.Lock()
    _write_lock = threading.Lock()

    def __new__(cls, log_dir="experiments/ccapo_logs"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CCAPOLogger, cls).__new__(cls)
                cls._instance._init(log_dir)
            return cls._instance

    def _init(self, log_dir):
        # 创建带时间戳的日志目录，确保每次运行独立
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 定义日志文件路径
        self.event_log_path = os.path.join(self.log_dir, "ccapo_events.jsonl")       # 稀疏高价值事件 (Anchor更新)
        self.metric_log_path = os.path.join(self.log_dir, "ccapo_metrics.jsonl")     # 每个Batch的统计 (Success Rate)
        self.reward_log_path = os.path.join(self.log_dir, "ccapo_rewards.jsonl")     # [v2.3新增] 细粒度奖励归因
        
        print(f">>> [CCAPO Logger] Logging to: {self.log_dir}")

    def log_event(self, event_type: str, data: dict):
        """记录系统级关键事件（如发现新 Anchor、STDB 快照保存等）"""
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        }
        self._append_to_file(self.event_log_path, entry)

    def log_step_metrics(self, step: int, metrics: dict):
        """记录训练步数维度的统计指标"""
        entry = {
            "timestamp": time.time(),
            "step": step,
            "metrics": metrics
        }
        self._append_to_file(self.metric_log_path, entry)

    def log_reward_composition(self, 
                             step: int, 
                             trace_id: str, 
                             step_idx: int, 
                             total_reward: float, 
                             components: Dict[str, float], 
                             meta: Optional[Dict] = None):
        """
        [v2.3 核心] 记录单步奖励的详细构成。
        用于分析 Agent 到底是靠 'Exec流' 还是 'Logic流' 拿分。
        
        Args:
            step: 全局训练步数 (batch index)
            trace_id: 轨迹唯一ID (group_id)
            step_idx: 轨迹内的第几步
            total_reward: 最终给出的总分
            components: 分数构成字典, e.g. {"exec": 0.1, "logic": 0.05, "milestone": 0.0}
            meta: 额外的元数据 (task_type, action_str等)
        """
        entry = {
            "t": time.time(),
            "step": step,
            "trace_id": trace_id,
            "step_idx": step_idx,
            "r_total": round(total_reward, 4),
            "components": components,
            "meta": meta or {}
        }
        self._append_to_file(self.reward_log_path, entry)

    def _append_to_file(self, filepath, data):
        # 使用锁确保多线程写入安全 (主要针对 RewardManager 多线程调用)
        with self._write_lock:
            try:
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Logger Error] Failed to write to {filepath}: {e}")

def get_logger():
    return CCAPOLogger()