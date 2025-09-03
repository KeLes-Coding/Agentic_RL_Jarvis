import subprocess
import logging
import time
from typing import List, Dict, Any


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

    def _execute_adb_command(
        self, command: list[str], timeout: int = 10, check_output=False
    ):
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

        注意: 此方法依赖一个后备方案，可能需要设备上安装 "ADB Keyboard" 应用。
        """
        self.logger.info(f"准备在 uid={uid} 的元素上输入文本: '{text}'")

        # 1. 先点击元素以确保其获得焦点
        if not self.tap(uid, elements):
            self.logger.error("输入文本失败：前置点击操作失败。")
            return False

        # 2. 等待一小段时间，让键盘或输入法有机会响应
        time.sleep(0.5)

        # 3. 逐字符模拟键盘输入
        self.logger.info("开始通过逐字符模拟键盘的方式输入...")
        for char in text:
            command_parts = []

            if char == " ":
                # 使用 keyevent 62 (KEYCODE_SPACE) 输入空格，比 'input text %s' 更可靠
                command_parts = ["shell", "input", "keyevent", "62"]
            elif char == "\n":
                # 使用 keyevent 66 (KEYCODE_ENTER) 输入换行
                command_parts = ["shell", "input", "keyevent", "66"]
            elif "a" <= char.lower() <= "z" or char.isdigit():
                # 对安全的字母和数字，直接使用 'input text'，效率较高
                command_parts = ["shell", "input", "text", char]
            else:
                # 对于其他所有字符（如：标点符号、中文字符、表情符号等）
                # 使用 am broadcast 命令配合 ADB Keyboard 应用输入，这是最兼容的方式。
                # 注意：这需要设备上预先安装并启用 ADBKeyBoard.apk
                self.logger.info(
                    f"字符 '{char}' 为特殊或非英文字符，使用 ADBKeyBoard 广播方式输入。"
                )
                # 使用双引号确保特殊字符被正确传递
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

            # 执行命令
            if not self._execute_adb_command(command_parts):
                self.logger.error(
                    f"输入字符 '{char}' (命令: {' '.join(command_parts)}) 失败。"
                )
                return False

            # 在字符之间增加一个微小的延迟，模拟真实输入，提高稳定性
            time.sleep(0.05)

        self.logger.info(f"文本 '{text}' 输入完成。")
        return True

    def swipe(
        self,
        start_uid: int,
        end_uid: int,
        elements: List[Dict[str, Any]],
        duration_ms: int = 400,
    ):
        """
        从一个元素的中心滑动到另一个元素的中心。
        """
        start_element = self._find_element_by_uid(start_uid, elements)
        end_element = self._find_element_by_uid(end_uid, elements)

        if not start_element or not end_element:
            return False

        x1, y1 = start_element["center"]
        x2, y2 = end_element["center"]

        self.logger.info(
            f"从 uid={start_uid} ({x1},{y1}) 滑动到 uid={end_uid} ({x2},{y2})"
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
        """
        执行“返回”操作。
        """
        self.logger.info("执行返回操作。")
        return self._execute_adb_command(
            ["shell", "input", "keyevent", "4"]
        )  # KEYCODE_BACK = 4

    def home(self):
        """
        执行“回到主屏幕”操作。
        """
        self.logger.info("执行Home操作。")
        return self._execute_adb_command(
            ["shell", "input", "keyevent", "3"]
        )  # KEYCODE_HOME = 3

    def wait(self, seconds: float):
        """
        等待指定的秒数。
        """
        self.logger.info(f"等待 {seconds} 秒...")
        time.sleep(seconds)
        return True  # 等待操作总是“成功”的
