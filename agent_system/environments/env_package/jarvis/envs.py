# agent_system/environments/env_package/jarvis/envs.py

import yaml
import numpy as np
import io
from typing import List, Dict, Tuple, Union
import logging
import re

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

        agent_config = self.jarvis_config.get("agent", {})
        self.compression_config = agent_config.get("image_compression", {})
        
        # --- 新增：从配置加载截断参数 ---
        self.max_elements_per_obs = agent_config.get("max_elements_per_obs", 70)
        self.max_str_len_per_obs = agent_config.get("max_str_len_per_obs", 10000)
        print(f"=== UI观察截断已启用: max_elements={self.max_elements_per_obs}, max_str_len={self.max_str_len_per_obs} ===")


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

            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            original_width, original_height = img.size
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image_format = self.compression_config.get("format", "JPEG")
            resized_img.save(buffer, format=image_format)
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

            # --- 新增：调用清理后台应用的方法 ---
            # print(f"--- [设备: {serial}] 正在清理后台应用... ---")
            # self.actuators[serial].clear_background_apps()
            # --- 新增结束 ---

            self.actuators[serial].home()
            # --- 修改：传递截断参数 ---
            obs_data = self.observers[serial].get_current_observation(
                max_elements=self.max_elements_per_obs,
                max_str_len=self.max_str_len_per_obs
            )
            
            screenshots_bytes = obs_data.get("screenshot_bytes")
            if not isinstance(screenshots_bytes, list):
                screenshots_bytes = [screenshots_bytes] if screenshots_bytes else []

            final_image_array = None
            compressed_bytes = None
            if screenshots_bytes:
                first_shot_bytes = screenshots_bytes[0]
                compressed_bytes = self._compress_single_image(first_shot_bytes)
                if compressed_bytes:
                    try:
                        img = Image.open(io.BytesIO(compressed_bytes)).convert("RGB")
                        final_image_array = np.array(img, dtype=np.uint8)
                    except Exception as e:
                        print(f"警告: 图像解码失败 - {e}")

            if final_image_array is None:
                final_image_array = np.zeros((256, 256, 3), dtype=np.uint8)

            obs_images.append(final_image_array)
            image_placeholders = "<image>\n"
            obs_text = obs_data.get("simplified_elements_str", "")
            obs_texts.append(f"{image_placeholders}{obs_text}")

            info_dict = {
                "device_serial": serial,
                "raw_obs_data": obs_data,
                "compressed_screenshot_bytes": compressed_bytes
            }
            infos.append(info_dict)
        return {"image": obs_images, "text": obs_texts}, infos

    def step(self, actions: List[str]) -> Tuple[Dict[str, List], List[float], List[bool], List[Dict]]:
        obs_images, obs_texts, rewards, dones, infos = [], [], [], [], []

        for i, serial in enumerate(self.device_serials):
            action_str = actions[i]
            # --- 修改：传递截断参数 ---
            pre_action_obs_data = self.observers[serial].get_current_observation(
                max_elements=self.max_elements_per_obs,
                max_str_len=self.max_str_len_per_obs
            )
            elements = pre_action_obs_data.get("simplified_elements_list")
            
            status = self._dispatch_action(self.actuators[serial], serial, action_str, elements)
            action_success = (status == "SUCCESS")
            self.episode_steps[serial] += 1
            
            # --- 修改：传递截断参数 ---
            post_action_obs_data = self.observers[serial].get_current_observation(
                max_elements=self.max_elements_per_obs,
                max_str_len=self.max_str_len_per_obs
            )
            
            screenshots_bytes = post_action_obs_data.get("screenshot_bytes")
            if not isinstance(screenshots_bytes, list):
                screenshots_bytes = [screenshots_bytes] if screenshots_bytes else []

            final_image_array = None
            compressed_bytes = None
            if screenshots_bytes:
                first_shot_bytes = screenshots_bytes[0]
                compressed_bytes = self._compress_single_image(first_shot_bytes)
                if compressed_bytes:
                    try:
                        img = Image.open(io.BytesIO(compressed_bytes)).convert("RGB")
                        final_image_array = np.array(img, dtype=np.uint8)
                    except Exception as e:
                        print(f"警告: 图像解码失败 - {e}")

            if final_image_array is None:
                final_image_array = np.zeros((256, 256, 3), dtype=np.uint8)

            obs_images.append(final_image_array)

            feedback_prefix = ""
            
            # --- 第 1 处修改：更新 format_reminder 中的 swipe 示例 ---
            format_reminder = (
                "--- CORRECT FORMAT ---\n"
                "You MUST respond in a strict, valid JSON format. Your entire output must be a single JSON object, without any markdown formatting, comments, or extra text.\n"
                'The JSON object must contain exactly two keys:\n1. "thought": Your reasoning.\n2. "action": The action to perform.\n\n'
                "--- AVAILABLE ACTIONS ---\n"
                "- `tap(uid: int)`: Example: `tap(12)`\n"
                "- `input_text(uid: int, text: str)`: Example: `input_text(5, 'hello world')`\n"
                "- `clear_text(uid: int)`\n"
                "- `enter()`\n"
                "- `swipe(direction, magnitude)`: Performs a swipe gesture.\n"
                    "\t- `direction`: The physical direction of the finger's movement: \"UP\", \"DOWN\", \"LEFT\", or \"RIGHT\".\n"
                    "\t- `magnitude`: (Optional) \"SHORT\", \"MEDIUM\", or \"LONG\". Defaults to \"MEDIUM\".\n"
                    "\t- **IMPORTANT CONTEXTUAL EXAMPLES**:\n"
                        "\t\t- To scroll down a list to see more content, you swipe your finger **UP**. Use `swipe(\"UP\", \"MEDIUM\")`.\n"
                        "\t\t- To open an app drawer from the home screen, you also swipe your finger **UP**. Use `swipe(\"UP\", \"LONG\")`.\n"
                        "\t\t- To scroll up a list to see previous content, you swipe your finger **DOWN**. Use `swipe(\"DOWN\", \"MEDIUM\")`.\n"
                "- `back()`: Example: `back()`\n"
                "- `home()`: Example: `home()`\n"
                "- `wait(seconds: float)`: Example: `wait(3.5)`\n"
                "- `finish(summary: str)`: Example: `finish(summary='Task is complete.')`\n"
            )

            if action_str.startswith("format_error"):
                reason = action_str[len("format_error(reason='"):-2]
                feedback_prefix = (
                    f"SYSTEM FEEDBACK: Your last output had a JSON format error.\n"
                    f"Error Details: {reason}\n"
                    f"{format_reminder}\nPlease correct your output and try again.\n\n"
                )
            elif not action_success:
                feedback_prefix = (
                    f"SYSTEM FEEDBACK: Your previous action failed to execute.\n"
                    f"Action Sent: {action_str}\n"
                    f"Execution Status: {status}\n"
                    f"{format_reminder}\nPlease analyze the error, check your action format and parameters, and try again.\n\n"
                )
            
            image_placeholders = "<image>\n"
            obs_text = post_action_obs_data.get("simplified_elements_str", "")
            obs_texts.append(f"{feedback_prefix}{image_placeholders}{obs_text}")
            
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
            
            task_won = done and (reward > 0)
            
            info_dict = {
                "device_serial": serial,
                "action_success": action_success,
                "won": task_won,
                "raw_obs_data": pre_action_obs_data,
                "compressed_screenshot_bytes": self._compress_single_image(pre_action_obs_data.get("screenshot_bytes", b'')),
            }
            infos.append(info_dict)
            
        observations = {"image": obs_images, "text": obs_texts}
        return observations, np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), infos

    def _dispatch_action(self, actuator: Actuator, serial: str, action_str: str, elements: list) -> str:
        print(f"--- [设备: {serial}] 正在分发动作: '{action_str}' ---")
        try:
            if action_str.startswith("format_error"):
                return f"FORMAT_ERROR: {action_str.split('reason=')[1][:-1]}"

            original_action_name = action_str.split("(")[0].strip()
            action_name = original_action_name

            if action_name == "finish":
                return "SUCCESS"
            if action_name == "click":
                action_name = "tap"

            params_str = action_str[len(original_action_name) + 1 : -1] if "(" in action_str else ""
            
            if action_name in ["tap", "input_text", "drag", "clear_text"] and not elements:
                return "FAILURE_NO_ELEMENTS"

            def extract_uid(param_part):
                numbers = re.findall(r'\d+', param_part)
                if not numbers:
                    raise ValueError(f"Cannot find a valid integer UID in parameter '{param_part}'")
                return int(numbers[0])

            if action_name == "tap":
                result = actuator.tap(extract_uid(params_str), elements)
            elif action_name == "input_text":
                uid_part, text_part = params_str.split(",", 1)
                text = text_part.strip()
                if text.startswith("text="):
                    text = text[len("text="):]
                text = text.strip("'\"")
                result = actuator.input_text(extract_uid(uid_part), text, elements)
            elif action_name == "swipe":
                direction_part, magnitude_part = params_str.split(",", 1)
                direction = direction_part.strip().strip("'\"")
                magnitude = magnitude_part.strip().strip("'\"")
                result = actuator.swipe(direction, magnitude)
            elif action_name == "drag":
                start_part, end_part = params_str.split(",", 1)
                result = actuator.drag(extract_uid(start_part), extract_uid(end_part), elements)
            # --- ✅ 新增动作解析 ✅ ---
            elif action_name == "clear_text":
                result = actuator.clear_text(extract_uid(params_str), elements)
            elif action_name == "enter":
                result = actuator.enter()
            # --- ✅ 解析结束 ✅ ---
            elif action_name == "back":
                result = actuator.back()
            elif action_name == "home":
                result = actuator.home()
            elif action_name == "wait":
                result = actuator.wait(float(params_str))
            else:
                return f"UNKNOWN_ACTION: '{action_name}' is not a valid action."

            status = "SUCCESS" if result else "FAILURE"
            print(f"--- [设备: {serial}] 动作 '{action_name}' 执行状态: {status} ---")
            return status
        except Exception as e:
            error_message = f"EXECUTION_ERROR: {repr(e)}"
            print(f"--- [设备: {serial}] 动作 '{action_str}' 执行时出错: {error_message} ---")
            return error_message


def build_jarvis_envs(jarvis_config_path: str, max_steps: int) -> JarvisMultiDeviceEnv:
    return JarvisMultiDeviceEnv(jarvis_config_path, max_steps)