import subprocess
import logging
import re
import xml.etree.ElementTree as ET
import os
import time
from typing import List, Dict, Any, Tuple


class Observer:
    """
    Observer（观察者）模块负责从安卓设备获取信息。
    它通过ADB命令获取屏幕截图和UI布局（XML），并将其解析为Agent可以理解的格式。
    """

    def __init__(self, adb_path: str, device_serial: str):
        """
        初始化观察者。
        Args:
            adb_path: ADB可执行文件的路径。
            device_serial: 目标设备的序列号。
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.adb_path = adb_path
        self.device_serial = device_serial
        self.local_temp_path = f"temp_{self.device_serial}_{int(time.time() * 1000)}"
        # --- 新增：初始化时获取屏幕尺寸 ---
        self.screen_width, self.screen_height = self._get_device_dimensions()

    def _execute_adb_command(
        self, command: list[str], timeout: int = 20, check_output: bool = False
    ) -> str | bool:
        """执行ADB命令的辅助函数。"""
        base_cmd = [self.adb_path, "-s", self.device_serial]
        full_command = base_cmd + command
        self.logger.debug(f"执行ADB命令: {' '.join(full_command)}")
        try:
            result = subprocess.run(
                full_command,
                check=True,
                timeout=timeout,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            return result.stdout.strip() if check_output else True
        except subprocess.TimeoutExpired:
            self.logger.error(f"ADB命令超时: {' '.join(full_command)}")
            return "" if check_output else False
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"ADB命令执行失败: {' '.join(full_command)}\nStderr: {e.stderr.strip()}"
            )
            return "" if check_output else False
        except Exception as e:
            self.logger.error(f"执行ADB命令时发生未知错误: {e}")
            return "" if check_output else False

    # --- 新增：获取设备物理屏幕尺寸的方法 ---
    def _get_device_dimensions(self) -> Tuple[int, int]:
        """获取设备的物理显示尺寸 (width, height)。"""
        self.logger.info("正在获取设备屏幕尺寸...")
        # 命令 'wm size' 的输出通常是 "Physical size: 1080x1920"
        output = self._execute_adb_command(["shell", "wm", "size"], check_output=True)
        if output and isinstance(output, str):
            match = re.search(r"(\d+)x(\d+)", output)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
                self.logger.info(f"获取到设备尺寸: {width}x{height}")
                return width, height

        self.logger.warning("无法获取设备尺寸，将使用默认值 1080x1920。")
        return 1080, 1920  # 返回一个通用的默认值以防万一

    def get_screenshot_bytes(self) -> bytes | None:
        """获取设备当前屏幕的截图，并以字节形式返回。"""
        remote_path = "/data/local/tmp/screenshot.png"
        local_path = f"{self.local_temp_path}_screen.png"
        try:
            if not self._execute_adb_command(["shell", "screencap", "-p", remote_path]):
                return None
            if not self._execute_adb_command(["pull", remote_path, local_path]):
                return None
            with open(local_path, "rb") as f:
                return f.read()
        finally:
            self._execute_adb_command(["shell", "rm", remote_path])
            if os.path.exists(local_path):
                os.remove(local_path)

    def get_layout_xml(self) -> str | None:
        """获取设备当前UI布局的XML文件内容。"""
        remote_path = "/data/local/tmp/uidump.xml"
        local_path = f"{self.local_temp_path}_uidump.xml"
        try:
            if not self._execute_adb_command(
                ["shell", "uiautomator", "dump", remote_path]
            ):
                return None
            if not self._execute_adb_command(["pull", remote_path, local_path]):
                return None
            with open(local_path, "r", encoding="utf-8") as f:
                return f.read()
        finally:
            self._execute_adb_command(["shell", "rm", remote_path])
            if os.path.exists(local_path):
                os.remove(local_path)

    def _parse_bounds(self, bounds_str: str) -> Tuple[int, int, int, int]:
        """从字符串（如"[0,100][200,300]"）中解析出边界坐标。"""
        coords = [int(n) for n in re.findall(r"\d+", bounds_str)]
        if len(coords) == 4:
            return coords[0], coords[1], coords[2], coords[3]
        return 0, 0, 0, 0

    def _is_element_actionable(self, node: ET.Element) -> bool:
        """判断一个XML节点（UI元素）是否是可交互的。"""
        is_interactive = (
            node.get("clickable") == "true"
            or node.get("long-clickable") == "true"
            or node.get("focusable") == "true"
        )
        has_text = node.get("text", "") != "" or node.get("content-desc", "") != ""
        is_visible = node.get("displayed") != "false"
        if not is_visible or node.get("enabled") != "true":
            return False
        return is_interactive or has_text

    # --- 新增：判断元素是否在视口内的方法 ---
    def _is_node_in_viewport(self, node: ET.Element) -> bool:
        """检查一个元素是否至少部分在屏幕视口内。"""
        bounds_str = node.get("bounds")
        if not bounds_str:
            return False

        x1, y1, x2, y2 = self._parse_bounds(bounds_str)

        # 检查元素是否与屏幕边界相交
        # 元素可见的条件是：它的底部在屏幕顶部之下，顶部在屏幕底部之上，
        # 并且它的右侧在屏幕左侧之后，左侧在屏幕右侧之前。
        is_vertically_visible = y1 < self.screen_height and y2 > 0
        is_horizontally_visible = x1 < self.screen_width and x2 > 0

        return is_vertically_visible and is_horizontally_visible

    def _parse_and_simplify_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        解析XML内容，提取可交互的UI元素，并进行简化。
        - 过滤掉视口外的元素。
        - 截断过长的文本。
        """
        simplified_elements = []
        if not xml_content:
            return simplified_elements
        try:
            root = ET.fromstring(xml_content)
            uid_counter = 1
            for node in root.iter():
                # --- 修改点 1：首先检查元素是否在视口内 ---
                if not self._is_node_in_viewport(node):
                    continue

                if self._is_element_actionable(node):
                    bounds_str = node.get("bounds")
                    bounds = self._parse_bounds(bounds_str)

                    # --- 修改点 2：截断过长的文本 ---
                    text = node.get("text", "")
                    if len(text) > 200:
                        text = text[:200] + "..."

                    content_desc = node.get("content-desc", "")
                    if len(content_desc) > 200:
                        content_desc = content_desc[:200] + "..."

                    element_data = {
                        "uid": uid_counter,
                        "class": node.get("class"),
                        "text": text,  # 使用可能被截断的文本
                        "content_desc": content_desc,  # 使用可能被截断的文本
                        "resource_id": node.get("resource-id", ""),
                        "bounds": bounds,
                        "center": (
                            (bounds[0] + bounds[2]) // 2,
                            (bounds[1] + bounds[3]) // 2,
                        ),
                        "clickable": node.get("clickable") == "true",
                        "password": node.get("password") == "true",
                        "checkable": node.get("checkable") == "true",
                        "checked": node.get("checked") == "true",
                        "selected": node.get("selected") == "true",
                    }
                    simplified_elements.append(element_data)
                    uid_counter += 1
        except ET.ParseError as e:
            self.logger.error(
                f"XML解析失败: {e}\n--- XML Content ---\n{xml_content[:500]}..."
            )
        return simplified_elements

    def get_current_observation(self) -> dict:
        """
        获取当前的完整观察数据，包括截图、XML和简化后的UI元素列表。
        """
        self.logger.info("正在获取当前设备观察数据...")
        screenshot = self.get_screenshot_bytes()
        xml = self.get_layout_xml()

        # 调用的是优化后的解析方法
        simplified_elements = self._parse_and_simplify_xml(xml)

        simplified_elements_str = ""
        for el in simplified_elements:
            all_parts = []
            text_info = f"text='{el['text']}'" if el["text"] else ""
            desc_info = f"desc='{el['content_desc']}'" if el["content_desc"] else ""
            id_info = f"id='{el['resource_id']}'" if el["resource_id"] else ""
            all_parts.extend([part for part in [text_info, desc_info, id_info] if part])

            # if el.get("password"):
            #     all_parts.append("is_password")
            # if el.get("checkable"):
            #     all_parts.append("checkable")
            #     all_parts.append("checked" if el.get("checked") else "unchecked")
            # if el.get("selected"):
            #     all_parts.append("selected")
            # if el.get("clickable"):
            #     all_parts.append("clickable")

            # b = el.get("bounds")
            # if b:
            #     all_parts.append(f"bounds=[{b[0]},{b[1]}][{b[2]},{b[3]}]")

            simplified_elements_str += f"[{el['uid']}] {el['class'].split('.')[-1]} {{{', '.join(all_parts)}}}\n"

        return {
            "screenshot_bytes": screenshot,
            "xml_content": xml,
            "simplified_elements_list": simplified_elements,
            "simplified_elements_str": simplified_elements_str,
        }
