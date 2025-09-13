# agent_system/environments/env_package/jarvis/envs.py

import yaml
import numpy as np
import io
from typing import List, Dict, Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None

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

        self.compression_config = self.jarvis_config.get("agent", {}).get("image_compression", {})
        if self.compression_config.get("enabled", False):
            print("===图像压缩已启用。===")
            if Image is None:
                raise ImportError("未安装 Pillow 库，无法进行图像压缩。请运行 `pip install Pillow`。")
        else:
            print("===图像压缩未启用。===")

    def _compress_single_image(self, image_bytes: bytes) -> bytes:
        if not self.compression_config.get("enabled", False) or not image_bytes:
            return image_bytes
        try:
            scale_factor = self.compression_config.get("scale_factor", 0.5)
            img = Image.open(io.BytesIO(image_bytes))

            # --- 核心修改：在这里转换图像模式 ---
            # JPEG 不支持 RGBA 的 Alpha 通道，需要转换为 RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            # --- 修改结束 ---
            
            original_width, original_height = img.size
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image_format = self.compression_config.get("format", "JPEG")
            resized_img.save(buffer, format=image_format)
            print("===图像压缩成功===")
            return buffer.getvalue()
        except Exception as e:
            print(f"===图像压缩失败: {e}===")
            return image_bytes

    def reset(self) -> Tuple[Dict[str, List], List[Dict]]:
        obs_images = []
        obs_texts = []
        infos = []

        for serial in self.device_serials:
            self.episode_steps[serial] = 0
            self.actuators[serial].home()
            obs_data = self.observers[serial].get_current_observation()

            # --- 开始替换 ---
            screenshots_bytes = obs_data.get("screenshot_bytes")
            if not isinstance(screenshots_bytes, list):
                screenshots_bytes = [screenshots_bytes] if screenshots_bytes else []

            final_image_array = None
            # 检查是否有截图
            if screenshots_bytes:
                # 只处理第一张截图
                first_shot_bytes = screenshots_bytes[0]
                compressed_bytes = self._compress_single_image(first_shot_bytes)
                if compressed_bytes:
                    try:
                        img = Image.open(io.BytesIO(compressed_bytes)).convert("RGB")
                        final_image_array = np.array(img, dtype=np.uint8)
                    except Exception as e:
                        print(f"警告: 图像解码失败 - {e}")

            # 如果没有有效图像，则创建一个黑色占位图，确保数据类型正确
            if final_image_array is None:
                print(f"警告: 设备 {serial} 未能获取截图, 将使用黑色图像占位。")
                final_image_array = np.zeros((256, 256, 3), dtype=np.uint8)

            # 确保每次只添加一个Numpy数组
            obs_images.append(final_image_array)

            # 相应地，只创建一个占位符
            image_placeholders = "<image>\n"
            obs_text = obs_data.get("simplified_elements_str", "")
            obs_texts.append(f"{image_placeholders}{obs_text}")
            # --- 替换结束 ---

            infos.append({"device_serial": serial})

        print(f"--- 调试信息 [envs.py/reset] ---")
        print(f"即将返回 obs_images，共 {len(obs_images)} 个元素，首元素类型: {type(obs_images[0]) if obs_images else 'N/A'}")
        print(f"即将返回 obs_texts，共 {len(obs_texts)} 个元素，首元素内容的前100个字符: '{obs_texts[0][:100] if obs_texts else 'N/A'}'")
        print(f"检查 '<image>' token 是否在 obs_texts[0] 中: {'<image>' in obs_texts[0] if obs_texts else 'N/A'}")
        print(f"------------------------------------")

        return {"image": obs_images, "text": obs_texts}, infos

    def step(self, actions: List[str]) -> Tuple[Dict[str, List], List[float], List[bool], List[Dict]]:
        obs_images, obs_texts, rewards, dones, infos = [], [], [], [], []

        for i, serial in enumerate(self.device_serials):
            action_str = actions[i]
            elements = self.observers[serial].get_current_observation().get("simplified_elements_list")
            status = self._dispatch_action(self.actuators[serial], action_str, elements)
            action_success = (status == "SUCCESS")
            self.episode_steps[serial] += 1
            obs_data = self.observers[serial].get_current_observation()
            
            # --- 开始替换 ---
            screenshots_bytes = obs_data.get("screenshot_bytes")
            if not isinstance(screenshots_bytes, list):
                screenshots_bytes = [screenshots_bytes] if screenshots_bytes else []

            final_image_array = None
            # 检查是否有截图
            if screenshots_bytes:
                # 只处理第一张截图
                first_shot_bytes = screenshots_bytes[0]
                compressed_bytes = self._compress_single_image(first_shot_bytes)
                if compressed_bytes:
                    try:
                        img = Image.open(io.BytesIO(compressed_bytes)).convert("RGB")
                        final_image_array = np.array(img, dtype=np.uint8)
                    except Exception as e:
                        print(f"警告: 图像解码失败 - {e}")

            # 如果没有有效图像，则创建一个黑色占位图，确保数据类型正确
            if final_image_array is None:
                print(f"警告: 设备 {serial} 未能获取截图, 将使用黑色图像占位。")
                final_image_array = np.zeros((256, 256, 3), dtype=np.uint8)

            # 确保每次只添加一个Numpy数组
            obs_images.append(final_image_array)

            # 相应地，只创建一个占位符
            image_placeholders = "<image>\n"
            obs_text = obs_data.get("simplified_elements_str", "")
            obs_texts.append(f"{image_placeholders}{obs_text}")
            # --- 替换结束 ---
            
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

        # 在 step 方法的 return observations, ... 之前
        print(f"--- 调试信息 [envs.py/step] ---")
        print(f"即将返回 obs_images，共 {len(obs_images)} 个元素，首元素类型: {type(obs_images[0]) if obs_images else 'N/A'}")
        print(f"即将返回 obs_texts，共 {len(obs_texts)} 个元素，首元素内容的前100个字符: '{obs_texts[0][:100] if obs_texts else 'N/A'}'")
        print(f"检查 '<image>' token 是否在 obs_texts[0] 中: {'<image>' in obs_texts[0] if obs_texts else 'N/A'}")
        print(f"----------------------------------")

        return observations, np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), infos

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