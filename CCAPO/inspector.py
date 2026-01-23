# CCAPO/inspector.py

import os
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime

class CCAPOInspector:
    def __init__(self, log_dir_root="experiments/ccapo_logs"):
        self.log_dir_root = log_dir_root
        try:
            self.latest_run_dir = self._get_latest_run()
            print(f"🔍 [Inspector] Analyzing Run: {self.latest_run_dir}")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            self.latest_run_dir = None
        
    def _get_latest_run(self):
        # 找到最近的一个 run_YYYYMMDD_HHMMSS 文件夹
        if not os.path.exists(self.log_dir_root):
            raise FileNotFoundError(f"Log root not found: {self.log_dir_root}")
        runs = glob.glob(os.path.join(self.log_dir_root, "run_*"))
        if not runs:
            raise FileNotFoundError("No log directories found!")
        return max(runs, key=os.path.getmtime)

    def load_data(self):
        if not self.latest_run_dir: return

        # 1. Load Metrics (Batch Level)
        metrics_path = os.path.join(self.latest_run_dir, "ccapo_metrics.jsonl")
        self.df_metrics = self._read_jsonl(metrics_path)
        
        # 2. Load Rewards (Step Level)
        rewards_path = os.path.join(self.latest_run_dir, "ccapo_rewards.jsonl")
        self.df_rewards = self._read_jsonl(rewards_path)
        
        # Flatten components column
        if not self.df_rewards.empty and 'components' in self.df_rewards.columns:
            comps = pd.json_normalize(self.df_rewards['components'])
            self.df_rewards = pd.concat([self.df_rewards.drop('components', axis=1), comps], axis=1)
            
        # Flatten meta column
        if not self.df_rewards.empty and 'meta' in self.df_rewards.columns:
            metas = pd.json_normalize(self.df_rewards['meta'])
            # [Fix] 这里的 meta 展开后，列名会直接变成 'act', 'valid', 'status' 等
            self.df_rewards = pd.concat([self.df_rewards.drop('meta', axis=1), metas], axis=1)

    def _read_jsonl(self, path):
        data = []
        if not os.path.exists(path):
            print(f"⚠️ Warning: {path} not found.")
            return pd.DataFrame()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    pass
        return pd.DataFrame(data)

    def analyze_health(self):
        """维度 1: 基础生命体征"""
        print("\n=== 1. Health Check (Validity) ===")
        if self.df_rewards.empty:
            print("No reward data found.")
            return

        total_steps = len(self.df_rewards)
        print(f"Total Micro Steps Recorded: {total_steps}")

        # [Fix] 健壮性检查：确认 'valid' 列是否存在
        if 'valid' in self.df_rewards.columns:
            valid_steps = self.df_rewards[self.df_rewards['valid'] == True]
            valid_rate = len(valid_steps) / total_steps
            print(f"✅ Valid Action Rate:       {valid_rate:.2%}")
            
            if valid_rate < 0.2:
                print("   🔴 CRITICAL: 模型几乎没有输出合法指令。")
                if 'act' in self.df_rewards.columns:
                    invalid_samples = self.df_rewards[self.df_rewards['valid'] == False]['act'].head(3).values
                    print(f"   Sample Invalid Actions: {invalid_samples}")
        else:
            print("⚠️ 'valid' column missing in logs. Skipping validity check.")

    def analyze_pioneers(self):
        """维度 2: 先锋与成败"""
        print("\n=== 2. Evolution (Pioneers) ===")
        if self.df_rewards.empty: return
        
        # [Fix] 健壮性检查
        required_cols = ['trace_id', 'status']
        if not all(col in self.df_rewards.columns for col in required_cols):
            print(f"⚠️ Missing columns {required_cols}. Cannot analyze pioneers.")
            # 尝试回退到 metrics 分析
            if not self.df_metrics.empty:
                 print("   [Fallback to Metrics Log]")
                 last_metric = self.df_metrics.iloc[-1]['metrics']
                 print(f"   Success Count (Total): {last_metric.get('success_count', 'N/A')}")
            return

        unique_trajs = self.df_rewards[['trace_id', 'status']].drop_duplicates()
        status_counts = unique_trajs['status'].value_counts()
        
        total_trajs = len(unique_trajs)
        print(f"Total Trajectories: {total_trajs}")
        print(status_counts.to_string())
        
        pioneer_cnt = status_counts.get('PIONEER', 0)
        success_cnt = status_counts.get('SUCCESS', 0)
        
        if total_trajs > 0:
            print(f"🌟 Pioneer Rate: {pioneer_cnt/total_trajs:.2%}")
            print(f"📈 Success Rate: {(pioneer_cnt + success_cnt)/total_trajs:.2%}")

    def analyze_reward_composition(self):
        """维度 3: 奖励成分解剖"""
        print("\n=== 3. Reward Composition Analysis ===")
        if self.df_rewards.empty: return
        
        if 'status' not in self.df_rewards.columns:
            print("⚠️ 'status' column missing. Showing aggregate stats.")
            statuses = ['ALL']
            self.df_rewards['status'] = 'ALL'
        else:
            statuses = ['FAIL', 'SUCCESS', 'PIONEER']

        cols_to_check = ['exec', 'logic', 'milestone', 'loop']
        available_cols = [c for c in cols_to_check if c in self.df_rewards.columns]

        for status in statuses:
            subset = self.df_rewards[self.df_rewards['status'] == status]
            if subset.empty: continue
            
            print(f"\n--- Status: {status} (n={len(subset)} steps) ---")
            
            for col in available_cols:
                avg_val = subset[col].mean()
                print(f"   Avg {col.capitalize()} Reward: {avg_val:.4f}")
            
            # 简单诊断
            if status == 'FAIL' and 'logic' in available_cols:
                avg_logic = subset['logic'].mean()
                if avg_logic == 0:
                    print("   🔴 WARNING: 失败轨迹 Logic 得分为 0。STDB 可能为空或未匹配。")

    def run(self):
        self.load_data()
        self.analyze_health()
        self.analyze_pioneers()
        self.analyze_reward_composition()

if __name__ == "__main__":
    inspector = CCAPOInspector()
    inspector.run()