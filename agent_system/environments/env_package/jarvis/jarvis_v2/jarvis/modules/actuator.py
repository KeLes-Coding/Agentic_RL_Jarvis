import subprocess
import logging
import time
import re
from typing import List, Dict, Any, Tuple


class Actuator:
    """
    Actuator（执行器）模块负责将Agent决策出的抽象动作（如 tap, swipe）
    转换为具体的ADB命令，并在安卓设备上执行这些命令。
    """

    def __init__(self, adb_path: str, device_serial: str):
        """
        初始化执行器。
        Args:
            adb_path: ADB可执行文件的路径。
            device_serial: 目标设备的序列号。
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.adb_path = adb_path
        self.device_serial = device_serial
        # --- 新增：为滑动操作初始化屏幕尺寸 ---
        self.screen_width, self.screen_height = self._get_device_dimensions()

    def _execute_adb_command(
        self, command: list[str], timeout: int = 10, check_output=False
    ) -> str | None | bool:
        """执行一个ADB命令并返回成功与否或输出"""
        cmd = [self.adb_path, "-s", self.device_serial] + command
        self.logger.info(f"执行动作: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, check=True, timeout=timeout, capture_output=True, text=True
            )
            self.logger.info("动作执行成功。")
            return result.stdout if check_output else True
        except subprocess.TimeoutExpired:
            self.logger.error(f"动作命令超时: {' '.join(cmd)}")
            return None if check_output else False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"动作命令执行失败: {' '.join(cmd)}\nStderr: {e.stderr}")
            return None if check_output else False
        except FileNotFoundError:
            self.logger.error(
                f"ADB命令未找到: {self.adb_path}。请检查config.yaml中的路径。"
            )
            return None if check_output else False

    def _get_device_dimensions(self) -> Tuple[int, int]:
        """获取设备的物理显示尺寸 (width, height)。"""
        self.logger.info("正在为 Actuator 获取设备屏幕尺寸...")
        try:
            output = self._execute_adb_command(
                ["shell", "wm", "size"], check_output=True
            )
            if output and isinstance(output, str):
                match = re.search(r"(\d+)x(\d+)", output)
                if match:
                    width, height = int(match.group(1)), int(match.group(2))
                    self.logger.info(f"获取到设备尺寸: {width}x{height}")
                    return width, height
        except Exception as e:
            self.logger.warning(f"获取设备尺寸时出错: {e}。将使用默认值。")

        self.logger.warning("无法获取设备尺寸，将使用默认值 1080x1920。")
        return 1080, 1920

    def _find_element_by_uid(
        self, uid: int, elements: List[Dict[str, Any]]
    ) -> Dict[str, Any] | None:
        """根据uid在元素列表中查找元素。"""
        for el in elements:
            if el["uid"] == uid:
                return el
        self.logger.error(f"未能在元素列表中找到 uid={uid} 的元素。")
        return None

    def tap(self, uid: int, elements: List[Dict[str, Any]]):
        """
        点击指定uid的元素。
        """
        element = self._find_element_by_uid(uid, elements)
        if not element:
            return False

        x, y = element["center"]
        self.logger.info(f"点击元素 uid={uid}，坐标 ({x}, {y})")
        return self._execute_adb_command(["shell", "input", "tap", str(x), str(y)])

    def input_text(self, uid: int, text: str, elements: List[Dict[str, Any]]):
        """
        在指定uid的元素上输入文本。
        此方法经过修改，采用逐字符发送ADB命令的方式，以提高对空格、
        特殊字符及Unicode字符（如中文、表情符号）的输入成功率。
        """
        self.logger.info(f"准备在 uid={uid} 的元素上输入文本: '{text}'")
        if not self.tap(uid, elements):
            self.logger.error("输入文本失败：前置点击操作失败。")
            return False
        time.sleep(1.5)
        self.logger.info("开始通过逐字符模拟键盘的方式输入...")
        for char in text:
            command_parts = []
            if char == " ":
                command_parts = ["shell", "input", "keyevent", "62"]
            elif char == "\n":
                command_parts = ["shell", "input", "keyevent", "66"]
            elif "a" <= char.lower() <= "z" or char.isdigit():
                command_parts = ["shell", "input", "text", char]
            else:
                self.logger.info(
                    f"字符 '{char}' 为特殊或非英文字符，使用 ADBKeyBoard 广播方式输入。"
                )
                command_parts = [
                    "shell",
                    "am",
                    "broadcast",
                    "-a",
                    "ADB_INPUT_TEXT",
                    "--es",
                    "msg",
                    f'"{char}"',
                ]
            if not self._execute_adb_command(command_parts):
                self.logger.error(
                    f"输入字符 '{char}' (命令: {' '.join(command_parts)}) 失败。"
                )
                return False
            time.sleep(0.05)
        self.logger.info(f"文本 '{text}' 输入完成。")
        return True

    def swipe(
        self, direction: str, magnitude_str: str = "medium", duration_ms: int = 400
    ):
        """
        在屏幕中心区域，沿着给定方向和幅度执行滑动操作。
        这是一个不依赖UI元素的通用滑动。

        Args:
            direction (str): 手指滑动的物理方向: "UP", "DOWN", "LEFT", "RIGHT".
            magnitude_str (str): 幅度 "SHORT", "MEDIUM", "LONG".
            duration_ms (int): 滑动持续时间（毫秒）。
        """
        self.logger.info(f"执行通用滑动: 手指方向={direction}, 幅度={magnitude_str}")

        margin_x = int(self.screen_width * 0.15)
        margin_y = int(self.screen_height * 0.15)
        center_x, center_y = self.screen_width // 2, self.screen_height // 2

        swipe_distance = {"SHORT": 0.2, "MEDIUM": 0.5, "LONG": 0.8}
        magnitude = swipe_distance.get(magnitude_str.upper(), 0.5)

        start_x, start_y, end_x, end_y = 0, 0, 0, 0

        direction = direction.upper()
        if direction == "UP":  # 手指从下向上滑
            start_x = end_x = center_x
            start_y = self.screen_height - margin_y
            end_y = int(start_y - (self.screen_height - 2 * margin_y) * magnitude)
        elif direction == "DOWN":  # 手指从上向下滑
            start_x = end_x = center_x
            start_y = margin_y
            end_y = int(start_y + (self.screen_height - 2 * margin_y) * magnitude)
        elif direction == "LEFT":  # 手指从右向左滑
            start_y = end_y = center_y
            start_x = self.screen_width - margin_x
            end_x = int(start_x - (self.screen_width - 2 * margin_x) * magnitude)
        elif direction == "RIGHT":  # 手指从左向右滑
            start_y = end_y = center_y
            start_x = margin_x
            end_x = int(start_x + (self.screen_width - 2 * margin_x) * magnitude)
        else:
            self.logger.error(f"无效的滑动方向: {direction}。")
            return False

        return self._execute_adb_command(
            [
                "shell",
                "input",
                "swipe",
                str(start_x),
                str(start_y),
                str(end_x),
                str(end_y),
                str(duration_ms),
            ]
        )

    def drag(
        self,
        start_uid: int,
        end_uid: int,
        elements: List[Dict[str, Any]],
        duration_ms: int = 800,
    ):
        """
        从一个元素的中心拖拽到另一个元素的中心。
        """
        start_element = self._find_element_by_uid(start_uid, elements)
        end_element = self._find_element_by_uid(end_uid, elements)

        if not start_element or not end_element:
            return False

        x1, y1 = start_element["center"]
        x2, y2 = end_element["center"]

        self.logger.info(
            f"从 uid={start_uid} ({x1},{y1}) 拖拽到 uid={end_uid} ({x2},{y2})"
        )
        return self._execute_adb_command(
            [
                "shell",
                "input",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(duration_ms),
            ]
        )

    def back(self):
        self.logger.info("执行返回操作。")
        return self._execute_adb_command(["shell", "input", "keyevent", "4"])

    def home(self):
        self.logger.info("执行Home操作。")
        return self._execute_adb_command(["shell", "input", "keyevent", "3"])

    def wait(self, seconds: float):
        self.logger.info(f"等待 {seconds} 秒...")
        time.sleep(seconds)
        return True
    
    def enter(self):
        """
        执行“回车”操作。
        """
        self.logger.info("执行Enter键操作。")
        return self._execute_adb_command(
            ["shell", "input", "keyevent", "66"]
        )  # KEYCODE_ENTER = 66

    def delete(self):
        """
        执行“删除”操作 (KEYCODE_DEL)。
        """
        return self._execute_adb_command(
            ["shell", "input", "keyevent", "67"]
        )  # KEYCODE_DEL = 67

    def clear_text(self, uid: int, elements: List[Dict[str, Any]]):
        """
        通过移动光标到末尾并循环发送删除键来清空指定uid的输入框。
        这是一种非常稳健的清空方式。
        """
        self.logger.info(f"准备清空 uid={uid} 的输入框...")
        element = self._find_element_by_uid(uid, elements)
        if not element:
            return False

        current_text = element.get("text", "")
        if not current_text:
            self.logger.info("输入框已为空，无需清空。")
            return True  # 视为空操作成功

        # 1. 点击元素以确保其获得焦点
        if not self.tap(uid, elements):
            self.logger.warning(f"清空前点击 uid={uid} 失败，但仍将继续尝试清空。")
        time.sleep(0.5)

        # 2. 移动光标到文本末尾 (确保从后往前删)
        # KEYCODE_MOVE_END 的键值为 123
        self._execute_adb_command(["shell", "input", "keyevent", "123"])
        time.sleep(0.1)

        # 3. 循环发送删除键
        # 为了保险起见，我们多删除几个字符
        num_deletions = len(current_text) + 5
        self.logger.info(
            f"输入框内容长度为 {len(current_text)}，将发送 {num_deletions} 次删除键。"
        )
        for i in range(num_deletions):
            if not self.delete():
                # 如果某次删除失败，就中止操作，避免无限循环或意外行为
                self.logger.error(f"在第 {i+1} 次删除时失败。")
                return False

        self.logger.info(f"输入框 (uid={uid}) 已尝试清空。")
        return True
