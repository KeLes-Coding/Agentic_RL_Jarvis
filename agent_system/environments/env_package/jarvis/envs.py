# agent_system/environments/env_package/jarvis/envs.py (最终修正版)

import time
import os
import yaml
import logging
import pandas as pd
import subprocess
import json
from typing import List, Dict, Any, Tuple, Optional
import sys

# --- 关键修正：正确设置模块搜索路径 ---
# 1. 获取包含所有 jarvis_v2 代码的根目录
#    根据你的描述，jarvis_v2 库放在了 envs.py 的同级目录
jarvis_v2_root = os.path.join(os.path.dirname(__file__), 'jarvis_v2')

# 2. 将此根目录添加到 sys.path
if jarvis_v2_root not in sys.path:
    sys.path.insert(0, jarvis_v2_root)


# --- 关键修正：从正确的文件中导入真实的函数和类 ---
# 从 agent_manager.py 导入 discover_devices 和 cleanup_ssh_tunnels
from agent_manager import discover_devices, cleanup_ssh_tunnels
# 从 jarvis_v2 的内部模块导入 Observer 和 Actuator
from jarvis.modules.observer import Observer
from jarvis.modules.actuator import Actuator


def build_jarvis_envs(config, env_num):
    """
    这个函数由 env_manager.py 调用，用于创建并行的、批处理的 Jarvis 环境。
    """
    from agent_system.environments.base import BatchedEnv
    envs = [JarvisEnv(config) for _ in range(env_num)]
    return BatchedEnv(envs)


class JarvisEnv:
    """
    Jarvis 安卓控制环境 (Step-by-Step 交互模式)。
    """
    def __init__(self, config):
        self.logger = logging.getLogger(f"JarvisEnv_{os.getpid()}")
        
        # 从 hydra 传入的完整配置中获取 jarvis 的专属配置
        jarvis_config = config.env.jarvis
        
        # 1. 加载 Jarvis V2 自身的配置文件
        # 注意：这里的路径是相对于项目根目录的，或者你需要提供一个绝对路径
        self.jarvis_config_path = jarvis_config.jarvis_config_path
        try:
            with open(self.jarvis_config_path, "r", encoding="utf-8") as f:
                self.jarvis_internal_config = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"错误: Jarvis V2 配置文件未找到于 {self.jarvis_config_path}")
            raise
        
        # 2. 发现设备 (使用从 agent_manager.py 导入的真实函数)
        all_devices = discover_devices(self.jarvis_internal_config)
        if not all_devices:
            raise RuntimeError("未发现任何可用的安卓设备")
        
        # TODO: 在多环境并行时，需要一个机制来为每个Env实例分配唯一的设备
        # 这个逻辑需要根据你的具体硬件设置来完善
        self.device_serial = all_devices[0] # 简单起见，暂时总是选择第一台设备
        self.logger.info(f"JarvisEnv 实例将使用设备: {self.device_serial}")

        # 3. 初始化设备观察器和执行器
        self.observer = Observer(self.device_serial)
        self.actuator = Actuator(self.device_serial)

        # 4. 加载任务数据集
        dataset_path = jarvis_config.dataset_path
        if not dataset_path or not os.path.exists(dataset_path):
            raise ValueError(f"必须在配置中提供有效的数据集路径，当前路径: {dataset_path}")
        self.dataset = pd.read_parquet(dataset_path)
        self.task_idx = 0

        # 5. Episode 状态追踪
        self.current_task_prompt = ""
        self.current_ground_truth = ""
        self.episode_steps = 0
        self.max_steps_per_episode = jarvis_config.max_steps_per_episode

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.logger.info(f"开始新的 Episode (任务索引: {self.task_idx})...")
        self.actuator.home()
        time.sleep(2)
        self.episode_steps = 0

        if self.task_idx >= len(self.dataset):
            self.task_idx = 0
        
        task_data = self.dataset.iloc[self.task_idx]
        try:
            # 解析JSON格式的prompt
            prompt_content = json.loads(task_data['prompt'])[0]['content']
        except (json.JSONDecodeError, IndexError):
             # 如果解析失败，直接使用原始字符串
            prompt_content = task_data['prompt']

        self.current_task_prompt = prompt_content
        self.current_ground_truth = task_data['ground_truth_answer']
        
        observation = self._get_observation()
        
        info = {
            "task_id": f"episode_{self.task_idx}",
            "task_prompt": self.current_task_prompt,
            "ground_truth_answer": self.current_ground_truth
        }
        
        self.task_idx += 1
        return observation, info

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        self.episode_steps += 1
        
        action_str = action.get("action", "")
        self.actuator.execute(action_str)
        
        observation = self._get_observation()
        
        done = False
        reward = 0.0
        
        if self.episode_steps >= self.max_steps_per_episode:
            done = True
            reward = 0.0
            self.logger.warning("Episode 因超时而结束。")

        if action_str.startswith("finish"):
            done = True
            summary = self._parse_finish_summary(action_str)
            reward = self._calculate_final_reward(summary, self.current_ground_truth, observation)
            self.logger.info(f"Episode 由 'finish' 动作结束，最终奖励: {reward}")

        info = {
            "action_str": action_str,
            "thought": action.get("thought", "")
        }
        
        return observation, reward, done, info

    def _get_observation(self) -> Dict[str, Any]:
        screenshot_path, simplified_ui, _ = self.observer.observe()
        return {
            "screenshot_path": screenshot_path,
            "simplified_ui": simplified_ui,
        }
        
    def _parse_finish_summary(self, action_str: str) -> str:
        try:
            return action_str[action_str.find("(")+1:action_str.rfind(")")].strip("'\"")
        except:
            return ""

    def _calculate_final_reward(self, final_answer: str, ground_truth: str, last_obs: Dict) -> float:
        final_answer = final_answer.strip()
        is_correct = (final_answer == ground_truth)
        if not is_correct:
            return 0.0
        
        gui_verified = False
        ui_text = last_obs.get('simplified_ui', '')
        if ground_truth in ui_text:
            gui_verified = True
            
        if is_correct and gui_verified:
            return 1.0
        elif is_correct and not gui_verified:
            return 0.5
        else:
            return 0.0

    def close(self):
        self.logger.info("正在关闭 JarvisEnv...")
        # 使用从 agent_manager.py 导入的真实函数
        cleanup_ssh_tunnels()