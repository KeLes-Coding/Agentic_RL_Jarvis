# CCAPO/inspector.py

import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings

# 忽略 pandas 的一些未来版本警告
warnings.simplefilter(action='ignore', category=FutureWarning)

class CCAPODiagnostician:
    def __init__(self, log_dir_root="experiments"):
        self.log_dir_root = log_dir_root
        self.run_dir = None
        self.df_metrics = pd.DataFrame() # 来自 Trainer (PPO, LASR, Success)
        self.df_rewards = pd.DataFrame() # 来自 CCAPO (Logic, Env, Loop)
        
        self._setup()

    def _setup(self):
        """自动定位最新的实验目录"""
        # 搜索逻辑：优先找 experiments/*/run_*，其次找 experiments/run_*
        candidates = []
        candidates.extend(glob.glob(os.path.join(self.log_dir_root, "*", "run_*")))
        candidates.extend(glob.glob(os.path.join(self.log_dir_root, "run_*")))
        
        if not candidates:
            print(f"❌ [Error] No run directories found in {self.log_dir_root}")
            return

        # 按修改时间排序，取最新的
        self.run_dir = max(candidates, key=os.path.getmtime)
        print(f"🔍 [Diagnostician] Analyzing Run: {self.run_dir}")
        print(f"   (Exp Name: {os.path.basename(os.path.dirname(self.run_dir))})")

    def load_data(self):
        if not self.run_dir: return

        # 1. Load Trainer Metrics (metrics.jsonl)
        # 这是 RayTrainer 写入的，包含 Loss, LASR, Success Rate
        m_path = os.path.join(self.run_dir, "metrics.jsonl") 
        # 兼容性检查
        if not os.path.exists(m_path):
            # 尝试寻找旧的或自定义路径
            m_path = os.path.join(self.run_dir, "ccapo_metrics.jsonl")
        
        if os.path.exists(m_path):
            self.df_metrics = self._read_jsonl(m_path)
            print(f"✅ Loaded Metrics: {len(self.df_metrics)} batches (Source: {os.path.basename(m_path)})")
        else:
            print("⚠️ Metrics log not found!")

        # 2. Load CCAPO Rewards (rewards.jsonl)
        # 这是 RewardManager 写入的，包含细粒度奖励构成
        r_path = os.path.join(self.run_dir, "rewards.jsonl")
        if not os.path.exists(r_path): 
            r_path = os.path.join(self.run_dir, "ccapo_rewards.jsonl")
            
        if os.path.exists(r_path):
            self.df_rewards = self._read_jsonl(r_path)
            self._preprocess_rewards()
            print(f"✅ Loaded Rewards: {len(self.df_rewards)} steps (Source: {os.path.basename(r_path)})")
        else:
            print("⚠️ Rewards log not found!")

    def _read_jsonl(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    # 自动展平嵌套的 'metrics' 字典 (Verl 的标准格式)
                    if 'metrics' in obj and isinstance(obj['metrics'], dict):
                        flat = obj.copy()
                        del flat['metrics']
                        flat.update(obj['metrics'])
                        data.append(flat)
                    else:
                        data.append(obj)
                except:
                    pass
        return pd.DataFrame(data)

    def _preprocess_rewards(self):
        """展平 Reward Log 的嵌套结构"""
        if self.df_rewards.empty: return
        
        # 展平 comps
        if 'comps' in self.df_rewards.columns:
            comps = pd.json_normalize(self.df_rewards['comps'])
            self.df_rewards = pd.concat([self.df_rewards.drop('comps', axis=1), comps], axis=1)
        elif 'components' in self.df_rewards.columns:
            comps = pd.json_normalize(self.df_rewards['components'])
            self.df_rewards = pd.concat([self.df_rewards.drop('components', axis=1), comps], axis=1)

        # 展平 meta
        if 'meta' in self.df_rewards.columns:
            metas = pd.json_normalize(self.df_rewards['meta'])
            self.df_rewards = pd.concat([self.df_rewards.drop('meta', axis=1), metas], axis=1)
        
        # 标准化列名
        if 'idx' in self.df_rewards.columns:
            self.df_rewards.rename(columns={'idx': 'step_idx'}, inplace=True)

    # =========================================================================
    # 模块 1: 训练健康度 (Training Health) - 核心指标
    # =========================================================================
    def diagnose_training_health(self):
        print("\n" + "="*60)
        print("🩺 MODULE 1: Training Health (PPO & Success)")
        print("="*60)
        
        if self.df_metrics.empty:
            print("❌ No metrics data.")
            return

        # 1. 成功率趋势 (Success Rate)
        # 查找包含 success 的列
        succ_cols = [c for c in self.df_metrics.columns if 'success' in c.lower()]
        if succ_cols:
            col = succ_cols[0]
            start_val = self.df_metrics[col].iloc[:5].mean() if len(self.df_metrics) > 5 else 0
            end_val = self.df_metrics[col].iloc[-5:].mean()
            print(f"📈 Success Rate ({col}):")
            print(f"   Start: {start_val:.2%} -> End: {end_val:.2%}")
            
            if end_val == 0:
                print("   🔴 [CRITICAL]: Success Rate is 0%. Agent fails to solve ANY task.")
                print("      -> 原因: 任务太难、Prompt 不对、或 Action 格式错误导致环境无法识别。")
        else:
            print("⚠️ No success rate metric found.")

        # 2. PPO 核心指标
        ppo_cols = ['actor/pg_loss', 'actor/pg_clipfrac', 'actor/ppo_kl', 'actor/entropy_loss']
        print("\n📊 PPO Vitals (Last 5 Batches Avg):")
        for col in ppo_cols:
            if col in self.df_metrics.columns:
                val = self.df_metrics[col].iloc[-5:].mean()
                print(f"   {col:<20}: {val:.6f}")
                
                # 简单诊断
                if col == 'actor/pg_clipfrac' and val > 0.4:
                    print("      ⚠️ High ClipFrac (>0.4): Policy changing too fast. Consider lowering Learning Rate.")
                if col == 'actor/ppo_kl' and val > 0.1:
                    print("      ⚠️ High KL Divergence (>0.1): Model is drifting from reference.")

    # =========================================================================
    # 模块 2: 数据污染 (Action Space)
    # =========================================================================
    def diagnose_pollution(self):
        print("\n" + "="*60)
        print("🚑 MODULE 2: Action Space Pollution Check")
        print("="*60)
        
        if self.df_rewards.empty or 'act' not in self.df_rewards.columns:
            print("⚠️ No action data available.")
            return

        top_actions = self.df_rewards['act'].value_counts().head(8)
        
        has_dirty_prefix = False
        print(f"{'Count':<6} | {'Action String':<40} | {'Status'}")
        print("-" * 60)
        for act, count in top_actions.items():
            act_str = str(act).lower()
            # 检查非法字符
            is_dirty = any(m in act_str for m in ['action:', 'thought:', '\n', 'instruction:'])
            flag = "🔴 DIRTY" if is_dirty else "✅ CLEAN"
            if is_dirty: has_dirty_prefix = True
            
            # 截断过长字符串
            display_act = (str(act)[:37] + '...') if len(str(act)) > 40 else str(act)
            print(f"{count:<6} | {display_act:<40} | {flag}")

        if has_dirty_prefix:
            print("\n❌ [DIAGNOSIS]: STDB POLLUTED! EnvManager cleaning logic failed.")
        else:
            print("\n✅ [DIAGNOSIS]: Action space looks clean.")

    # =========================================================================
    # 模块 3: LASR 审计 (Reweighting)
    # =========================================================================
    def diagnose_lasr(self):
        print("\n" + "="*60)
        print("⚖️ MODULE 3: LASR (Reweighting) Audit")
        print("="*60)

        if self.df_metrics.empty: return

        # 查找 lasr 相关列
        lasr_cols = [c for c in self.df_metrics.columns if 'lasr/' in c]
        
        if not lasr_cols:
            print("❌ LASR metrics MISSING in logs.")
            print("   -> 只有 AdvantageEstimator='ccapo' 且 'infos' 中包含成功轨迹时才会触发 LASR。")
            print(f"   -> Available Metric Keys (Sample): {list(self.df_metrics.columns)[:10]}")
            return

        recent = self.df_metrics.tail(5)
        if 'lasr/success_cnt' in recent.columns:
            avg_succ = recent['lasr/success_cnt'].mean()
            print(f"   Avg Success Trajs per Batch: {avg_succ:.1f}")
            if avg_succ < 1.0:
                print("   🔴 [ISSUE]: 几乎没有成功轨迹，LASR 退化为全 1.0 (无重加权)。")
                return

        if 'lasr/max_weight' in recent.columns:
            max_w = recent['lasr/max_weight'].max()
            min_w = recent['lasr/min_weight'].min()
            print(f"   Weight Range (Last 5): [{min_w:.4f}, {max_w:.4f}]")
            
            if abs(max_w - 1.0) < 1e-4 and abs(min_w - 1.0) < 1e-4:
                print("   🟡 [INFO]: Weights are static (all 1.0). LASR inactive.")
            else:
                print("   ✅ [INFO]: LASR is active! Data is being reweighted.")

    # =========================================================================
    # 模块 4: 奖励统治力 (Reward Scale)
    # =========================================================================
    def diagnose_reward_scale(self):
        print("\n" + "="*60)
        print("💰 MODULE 4: Reward Scale & Composition")
        print("="*60)

        if self.df_rewards.empty: return

        cols = ['logic', 'env', 'loop']
        stats = {}
        
        for c in cols:
            if c in self.df_rewards.columns:
                # 过滤掉 0 值，只看非零奖励的平均强度
                non_zeros = self.df_rewards[self.df_rewards[c].abs() > 1e-6][c]
                stats[c] = non_zeros.abs().mean() if not non_zeros.empty else 0.0
        
        print(f"{'Type':<10} | {'Avg Magnitude (Non-Zero)'}")
        print("-" * 35)
        for k, v in stats.items():
            print(f"{k:<10} | {v:.4f}")

        # 诊断
        env_mag = stats.get('env', 0)
        logic_mag = stats.get('logic', 0)
        
        if env_mag == 0:
            print("\n🔴 [CRITICAL]: Env Reward is 0.0!")
            print("   -> Agent 从未完成任务，或者 EnvManager 未正确传递 'final_env_reward'。")
            print("   -> 如果 Success Rate > 0 但 Env Reward = 0，说明数据传递链条断了。")
        elif logic_mag > 0:
            ratio = logic_mag / env_mag
            print(f"\nScale Ratio (Logic / Env): {ratio:.2f}")
            if ratio > 5.0:
                print("   ⚠️ [WARNING]: Logic Reward dominates. Risk of Reward Hacking.")

    def run(self):
        self.load_data()
        self.diagnose_training_health()
        self.diagnose_pollution()
        self.diagnose_lasr()
        self.diagnose_reward_scale()

if __name__ == "__main__":
    inspector = CCAPODiagnostician()
    inspector.run()