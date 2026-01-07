# tests/ccapo_tests/test_ccapo_core.py

import unittest
import torch
import numpy as np
import logging
import os
import shutil
from types import SimpleNamespace
from datetime import datetime

# 导入待测模块
from gigpo.core_ccapo import compute_ccapo_outcome_advantage
from agent_system.reward_manager.stdb import SuccessTrajectoryDatabase

# --- 配置 ---
LOG_DIR = "logger/test"
USE_SIMULATION_ROLLOUT = False  # <--- ✅ 仿真模式开关

class TestCCAPOCore(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """初始化日志系统"""
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR) # 清理旧日志
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 配置 Root Logger
        logging.basicConfig(level=logging.DEBUG) # 捕获所有级别的日志
        
        # 配置专门的文件 Logger
        cls.file_logger = logging.getLogger("CCAPO_TEST")
        cls.file_logger.setLevel(logging.DEBUG)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"test_run_{timestamp}.log")
        
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # 将 handler 添加到 root logger 和 CCAPO_CORE logger
        logging.getLogger().addHandler(fh)
        logging.getLogger("CCAPO_CORE").addHandler(fh)
        
        print(f"\n[Test Setup] Logs will be saved to: {log_path}")

    def setUp(self):
        self.stdb_path = os.path.join(LOG_DIR, "mock_stdb.json")
        self.stdb = SuccessTrajectoryDatabase(save_path=self.stdb_path)
        
        # 初始化 STDB 数据
        self.stdb.db = {
            "task_101": ["go north", "take apple", "eat apple"]
        }
        self.stdb.save()

    def test_lcs_logic_mock(self):
        """
        [Mock 模式] 纯数学逻辑测试
        """
        if USE_SIMULATION_ROLLOUT:
            print("Skipping Mock Test (Simulation Mode is ON)")
            return

        print("\n=== Running Mock LCS Test ===")
        logging.getLogger("CCAPO_TEST").info("Starting Mock Test...")

        # 构造输入数据 (模拟 2 条轨迹)
        # Traj 0: 完美匹配
        # Traj 1: 第一步错，第二步对
        non_tensor_batch = {
            'uid': np.array([0, 1, 2, 3]), 
            'traj_uid': np.array(['traj_0', 'traj_0', 'traj_1', 'traj_1']),
            'executed_action_str': np.array([
                "go north", "take apple",   # Traj 0 (Correct)
                "go south", "take apple"    # Traj 1 (Mixed)
            ]),
            'raw_prompt': np.array(["task_101", "task_101", "task_101", "task_101"])
        }

        # 构造 Tensor (假设 seq_len=5)
        bsz = 4
        seq_len = 5
        fake_rewards = torch.zeros((bsz, seq_len))
        fake_mask = torch.ones((bsz, seq_len))

        batch = SimpleNamespace(
            non_tensor_batch=non_tensor_batch,
            batch={
                'token_level_rewards': fake_rewards,
                'response_mask': fake_mask
            }
        )

        # 运行算法
        adv, _ = compute_ccapo_outcome_advantage(
            batch=batch,
            stdb_manager=self.stdb
        )

        # 验证
        # Traj 0: 全对 -> +1.0
        self.assertEqual(adv[0, 0].item(), 1.0)
        self.assertEqual(adv[1, 0].item(), 1.0)
        
        # Traj 1: "go south" (错) -> -0.1
        self.assertAlmostEqual(adv[2, 0].item(), -0.1, places=2)
        # Traj 1: "take apple" (虽然动作对，但前面错了，LCS 是否能匹配上取决于你用的具体 LCS 变种)
        # 在标准 LCS 中，跳过中间错误步骤后，后面的匹配是算的。
        # "go north", "take apple" vs "go south", "take apple" -> LCS 长度为 1 ("take apple")
        # 你的 core_ccapo 逻辑是基于索引匹配的，如果 seq_a[1] == seq_b[1]，它应该能匹配上。
        self.assertEqual(adv[3, 0].item(), 1.0) 

        logging.getLogger("CCAPO_TEST").info("Mock Test Passed Successfully.")

    def test_simulation_rollout(self):
        """
        [Simulation 模式] 驱动真实/伪造的 Rollout
        """
        if not USE_SIMULATION_ROLLOUT:
            print("Skipping Simulation Test (Set USE_SIMULATION_ROLLOUT=True to run)")
            return

        print("\n=== Running Simulation Rollout (Integration Test) ===")
        logging.getLogger("CCAPO_TEST").info("Starting Simulation Rollout...")

        # 这里模拟 vLLM 生成过程
        # 在真实场景下，你可以在这里调用 self.trainer.fit() 或者类似的入口
        # 由于我们没有 GPU 环境，我们这里模拟一个“生成器”生成了数据
        
        print(">> Simulating vLLM generation...")
        generated_actions = ["go north", "take key"] # 假设模型生成了这个
        
        # 构造 Batch
        non_tensor_batch = {
            'uid': np.array([0, 1]),
            'traj_uid': np.array(['sim_traj_0', 'sim_traj_0']),
            'executed_action_str': np.array(generated_actions),
            'raw_prompt': np.array(["task_101", "task_101"])
        }
        
        bsz = 2
        seq_len = 10
        batch = SimpleNamespace(
            non_tensor_batch=non_tensor_batch,
            batch={
                'token_level_rewards': torch.zeros((bsz, seq_len)),
                'response_mask': torch.ones((bsz, seq_len))
            }
        )
        
        # 运行核心算法
        adv, _ = compute_ccapo_outcome_advantage(batch, self.stdb)
        
        # 打印结果供人工检查
        print(f">> Simulated Actions: {generated_actions}")
        print(f">> STDB Anchor: {self.stdb.get_best_sequence('task_101')}")
        print(f">> Computed Advantages: {adv[:, 0].tolist()}")
        
        logging.getLogger("CCAPO_TEST").info(f"Simulated Result: {adv[:, 0].tolist()}")

if __name__ == '__main__':
    unittest.main()