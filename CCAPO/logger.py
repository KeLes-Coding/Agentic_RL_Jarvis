# CCAPO/logger.py

import json
import os
import time
import threading
from datetime import datetime

class CCAPOLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_dir="experiments/ccapo_logs"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CCAPOLogger, cls).__new__(cls)
                cls._instance._init(log_dir)
            return cls._instance

    def _init(self, log_dir):
        # 创建带时间戳的日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 定义日志文件路径
        self.event_log_path = os.path.join(self.log_dir, "ccapo_events.jsonl")
        self.metric_log_path = os.path.join(self.log_dir, "ccapo_metrics.jsonl")
        
        print(f">>> [CCAPO Logger] Logging to: {self.log_dir}")

    def log_event(self, event_type: str, data: dict):
        """记录具体的事件（如发现新 Anchor、触发 Milestone）"""
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        }
        self._append_to_file(self.event_log_path, entry)

    def log_step_metrics(self, step: int, metrics: dict):
        """记录每一步的统计指标"""
        entry = {
            "timestamp": time.time(),
            "step": step,
            "metrics": metrics
        }
        self._append_to_file(self.metric_log_path, entry)

    def _append_to_file(self, filepath, data):
        # 简单的追加写入 (Thread-safe for local file usually requires logging module, 
        # but for simplicity in this specific architecture, append is generally fine)
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Logger Error] {e}")

# 全局单例访问点
def get_logger():
    return CCAPOLogger()