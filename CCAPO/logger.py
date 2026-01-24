# CCAPO/logger.py

import os
import json
import time
import threading
import shutil
import uuid
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List

class JsonLineLogger:
    """
    专门用于写入 .jsonl 格式流式日志的辅助类，线程安全。
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock = threading.Lock()
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def log(self, data: Dict[str, Any]):
        """写入单行 JSON"""
        with self._lock:
            try:
                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                # 降级打印，防止日志系统崩溃影响主流程
                print(f"[Logger Error] Failed to write to {self.filepath}: {e}")

class CCAPOMonitor:
    """
    CCAPO v3.0 监控中心 (Singleton)
    负责统一管理 Metric流、Event流 和 Trajectory 归档。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CCAPOMonitor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, exp_name: str = "default_exp", log_root: str = "experiments"):
        """
        初始化监控系统。
        目录结构:
        experiments/
          └── {exp_name}/
              └── runs_{date}_{time}/
                  ├── metrics.jsonl       # 训练曲线 (Step, Reward, SuccessRate)
                  ├── events.jsonl        # 稀疏事件 (Pioneer发现, STDB扩容)
                  ├── rewards.jsonl       # 详细奖励归因 (Exec/Logic/Loop/Env)
                  └── traces/             # 轨迹详情归档
                      ├── batch_0/
                      │   ├── success_s12_groupA.json
                      │   └── fail_s50_groupB.json
                      └── ...
        """
        if getattr(self, "_initialized", False):
            return

        # 1. 建立目录结构
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")
        self.run_id = f"run_{date_str}_{time_str}"
        self.root_dir = os.path.join(log_root, exp_name, self.run_id)
        
        self.trace_dir = os.path.join(self.root_dir, "traces")
        os.makedirs(self.trace_dir, exist_ok=True)

        # 2. 初始化子日志器
        self.metric_logger = JsonLineLogger(os.path.join(self.root_dir, "metrics.jsonl"))
        self.event_logger = JsonLineLogger(os.path.join(self.root_dir, "events.jsonl"))
        self.reward_logger = JsonLineLogger(os.path.join(self.root_dir, "rewards.jsonl"))

        # 3. 配置 Python 标准 Logging (用于控制台输出)
        self._setup_console_logger()

        print(f">>> [CCAPO Monitor] System Online. Logging to: {self.root_dir}")
        self._initialized = True

    def _setup_console_logger(self):
        self.console = logging.getLogger("CCAPO")
        self.console.setLevel(logging.INFO)
        if not self.console.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            ch.setFormatter(formatter)
            self.console.addHandler(ch)

    # ==========================================================
    # 1. High-Frequency Logs (Reward Composition)
    # ==========================================================
    
    def log_reward_composition(self, 
                               step: int, 
                               trace_id: str, 
                               step_idx: int, 
                               total_reward: float, 
                               components: Dict[str, float], 
                               meta: Optional[Dict] = None):
        """
        [RewardManager 调用] 记录单步奖励的详细构成。
        """
        data = {
            "t": time.time(),
            "step": step,
            "trace_id": trace_id,
            "idx": step_idx,
            "r_total": round(float(total_reward), 4),
            "comps": {k: round(float(v), 4) for k, v in components.items()},
            "meta": meta or {}
        }
        self.reward_logger.log(data)

    # ==========================================================
    # 2. Batch-Level Logs (Metrics)
    # ==========================================================

    def log_batch_metrics(self, global_step: int, metrics: Dict[str, float]):
        """
        [Trainer 调用] 记录每个 Batch 的聚合指标 (Success Rate, Avg Length, Loss)。
        """
        data = {
            "timestamp": time.time(),
            "step": global_step,
            "metrics": metrics
        }
        self.metric_logger.log(data)
        
        # 同时打印关键指标到控制台
        if "success_rate" in metrics:
            sr = metrics["success_rate"]
            # 简化控制台输出，防止刷屏
            if global_step % 1 == 0: 
                self.console.info(f"Step {global_step} | SR: {sr:.2%} | Success: {metrics.get('success_count',0)}")

    def log_step_metrics(self, step: int, metrics: Dict[str, float]):
        """
        [Compatibility Alias] 
        为了兼容旧版 RewardManager 的调用 (self.logger.log_step_metrics)，
        将其路由到 log_batch_metrics。
        """
        self.log_batch_metrics(global_step=step, metrics=metrics)

    # ==========================================================
    # 3. Sparse Events (System Highlights)
    # ==========================================================

    def log_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        """
        [System 调用] 记录稀疏的关键事件。
        """
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "msg": message,
            "data": data or {}
        }
        self.event_logger.log(entry)
        
        if event_type == "pioneer_found":
            self.console.info(f"🔥 [PIONEER] {message}")
        else:
            self.console.info(f"ℹ️ [{event_type}] {message}")

    # ==========================================================
    # 4. Trajectory Archival (The Black Box)
    # ==========================================================

    def log_trajectory(self, batch_idx: int, traj_data: Dict):
        """
        [EnvManager 调用] 将完整轨迹保存为独立文件。
        """
        try:
            batch_dir = os.path.join(self.trace_dir, f"batch_{batch_idx}")
            os.makedirs(batch_dir, exist_ok=True)

            meta = traj_data.get('meta', {})
            metrics = traj_data.get('metrics', {})
            
            group_id = meta.get('group_id', str(uuid.uuid4())[:8])
            task_type = meta.get('task_type', 'unknown')
            is_success = metrics.get('is_success', False)
            steps = metrics.get('total_steps', 0)
            
            status_prefix = "SUCCESS" if is_success else "FAIL"
            filename = f"{status_prefix}_{task_type}_s{steps}_{group_id}.json"
            
            filepath = os.path.join(batch_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(traj_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[Logger] Failed to save trajectory: {e}")

# 兼容旧代码的获取函数
def get_logger(exp_name="ccapo_exp"):
    return CCAPOMonitor(exp_name=exp_name)