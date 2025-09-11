# agent_system/environments/env_package/jarvis/envs.py

import yaml
import numpy as np  # <-- 确保导入 numpy
from typing import List, Dict, Tuple
import io
try:
    from PIL import Image
except ImportError:
    Image = None

# 假设 jarvis_v2 源码位于 agent_system/environments/env_package/jarvis_v2
from .jarvis_v2.jarvis.modules.observer import Observer
from .jarvis_v2.jarvis.modules.actuator import Actuator
from .jarvis_v2.agent_manager import discover_devices

class JarvisMultiDeviceEnv:
    """
    一个底层的、支持多设备的 Jarvis 环境。
    它封装了与一组安卓设备的直接交互 (reset, step)。
    这个类不处理复杂的 prompt 构建，只提供原始观测数据。
    """
    def __init__(self, jarvis_config_path: str, max_steps_per_episode: int):
        try:
            with open(jarvis_config_path, "r", encoding="utf-8") as f:
                self.jarvis_config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"错误: jarvis_v2 的配置文件 '{jarvis_config_path}' 未找到！")

        self.device_serials: List[str] = discover_devices(self.jarvis_config)
        if not self.device_serials:
            raise RuntimeError("未能发现任何可用的安卓设备，请检查配置或设备连接。")

        self.num_envs = len(self.device_serials)
        print(f"JarvisMultiDeviceEnv 初始化成功，管理 {self.num_envs} 台设备: {self.device_serials}")

        adb_path = self.jarvis_config.get("adb", {}).get("executable_path", "adb")
        self.observers: Dict[str, Observer] = {s: Observer(adb_path, s) for s in self.device_serials}
        self.actuators: Dict[str, Actuator] = {s: Actuator(adb_path, s) for s in self.device_serials}
        
        self.max_steps_per_episode = max_steps_per_episode
        self.episode_steps: Dict[str, int] = {s: 0 for s in self.device_serials}

        self.compression_config = self.jarvis_config.get("image_compression", {})
        if self.compression_config.get("enabled", False):
            print("图像压缩已启用。")
            if Image is None:
                raise ImportError("未安装 Pillow 库，无法进行图像压缩。请运行 `pip install Pillow`。")

    def _compress_image(self, image_bytes: bytes) -> bytes:
        if not self.compression_config.get("enabled", False) or not image_bytes:
            return image_bytes
        try:
            scale_factor = self.compression_config.get("scale_factor", 0.5)
            img = Image.open(io.BytesIO(image_bytes))
            original_width, original_height = img.size
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image_format = self.compression_config.get("format", "JPEG")
            resized_img.save(buffer, format=image_format)
            compressed_bytes = buffer.getvalue()
            return compressed_bytes
        except Exception as e:
            print(f"图像压缩失败: {e}")
            return image_bytes

    def reset(self) -> Tuple[Dict[str, List], List[Dict]]:
        """重置所有设备并返回初始观测值。"""
        obs_images = []
        obs_texts = []
        infos = []

        for serial in self.device_serials:
            self.episode_steps[serial] = 0
            self.actuators[serial].home()
            
            obs_data = self.observers[serial].get_current_observation()
            
            # --- 修改部分 ---
            screenshot_bytes = obs_data.get("screenshot_bytes")
            compressed_screenshot = self._compress_image(screenshot_bytes)
            
            # 将 bytes 转换为 NumPy 数组
            if compressed_screenshot:
                img = Image.open(io.BytesIO(compressed_screenshot)).convert("RGB")
                obs_images.append(np.array(img))
            else:
                # 如果没有截图，可以添加一个占位符，例如全黑图片
                # 注意：尺寸需要与你的模型输入匹配，这里用一个小的占位符
                obs_images.append(np.zeros((64, 64, 3), dtype=np.uint8))
            # --- 修改结束 ---

            obs_texts.append(obs_data.get("simplified_elements_str"))
            infos.append({"device_serial": serial})

        return {"image": obs_images, "text": obs_texts}, infos

    def step(self, actions: List[str]) -> Tuple[Dict[str, List], List[float], List[bool], List[Dict]]:
        """在所有设备上执行动作。"""
        obs_images, obs_texts, rewards, dones, infos = [], [], [], [], []

        for i, serial in enumerate(self.device_serials):
            action_str = actions[i]
            
            elements = self.observers[serial].get_current_observation().get("simplified_elements_list")
            status = self._dispatch_action(self.actuators[serial], action_str, elements)
            action_success = (status == "SUCCESS")
            
            self.episode_steps[serial] += 1

            obs_data = self.observers[serial].get_current_observation()
            
            # --- 修改部分 ---
            screenshot_bytes = obs_data.get("screenshot_bytes")
            compressed_screenshot = self._compress_image(screenshot_bytes)

            # 将 bytes 转换为 NumPy 数组
            if compressed_screenshot:
                img = Image.open(io.BytesIO(compressed_screenshot)).convert("RGB")
                obs_images.append(np.array(img))
            else:
                obs_images.append(np.zeros((64, 64, 3), dtype=np.uint8))
            # --- 修改结束 ---
            
            obs_texts.append(obs_data.get("simplified_elements_str"))

            done = False
            reward = 0.0
            if action_str.startswith("finish"):
                reward = 1.0
                done = True
            elif not action_success:
                reward = -0.1
            
            if self.episode_steps[serial] >= self.max_steps_per_episode:
                done = True

            rewards.append(reward)
            dones.append(done)
            infos.append({"device_serial": serial, "action_success": action_success})

        observations = {"image": obs_images, "text": obs_texts}
        return observations, rewards, dones, infos

    def _dispatch_action(self, actuator: Actuator, action_str: str, elements: list) -> str:
        try:
            action_name = action_str.split("(")[0]
            params_str = action_str[len(action_name) + 1 : -1] if "(" in action_str else ""
            if action_name in ["tap", "input_text", "swipe"] and not elements: return "FAILURE_NO_ELEMENTS"
            if action_name == "tap": result = actuator.tap(int(params_str), elements)
            elif action_name == "input_text":
                uid, text = params_str.split(",", 1)
                result = actuator.input_text(int(uid), text.strip().strip("'\""), elements)
            elif action_name == "swipe":
                s_uid, e_uid = map(int, params_str.split(","))
                result = actuator.swipe(s_uid, e_uid, elements)
            elif action_name == "back": result = actuator.back()
            elif action_name == "home": result = actuator.home()
            elif action_name == "wait": result = actuator.wait(float(params_str))
            else: return "UNKNOWN_ACTION"
            return "SUCCESS" if result else "FAILURE"
        except Exception: return "EXECUTION_ERROR"

def build_jarvis_envs(jarvis_config_path: str, max_steps: int) -> JarvisMultiDeviceEnv:
    """构建并返回一个 JarvisMultiDeviceEnv 实例"""
    return JarvisMultiDeviceEnv(jarvis_config_path, max_steps)