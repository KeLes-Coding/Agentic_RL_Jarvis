# test_actuator_input_comprehensive.py (V2 - 更稳健的清空逻辑)

import sys
import logging
import time
import xml.etree.ElementTree as ET
import subprocess

# --- 路径配置 ---
# ！！！重要：请将下面的路径修改为您的 actuator.py 所在的文件夹路径
# 例如：如果 actuator.py 在 /path/to/project/modules/actuator.py
# 那么这里就写 '/path/to/project'
PROJECT_ROOT_PATH = "agent_system/environments/env_package/jarvis/jarvis_v2" 
sys.path.append(PROJECT_ROOT_PATH)
# -----------------

# 导入修正后的 Actuator 类
from jarvis.modules.actuator import Actuator

# --- 测试环境配置 ---
ADB_PATH = "/home/zzh/Android/Sdk/platform-tools/adb" # 您的 ADB 路径
DEVICE_SERIAL = "emulator-5554" # 您的设备序列号
# --------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ActuatorInputTest")

def get_ui_dump_and_find_input(adb_path: str, device: str) -> dict | None:
    """
    获取UI布局，查找第一个输入框，并返回其信息及用于Actuator的元素列表。
    """
    logger.info("正在获取UI布局并查找输入框...")
    try:
        subprocess.run(
            [adb_path, "-s", device, "shell", "uiautomator", "dump", "/sdcard/uidump.xml"],
            check=True, timeout=10, capture_output=True
        )
        xml_content = subprocess.run(
            [adb_path, "-s", device, "shell", "cat", "/sdcard/uidump.xml"],
            check=True, capture_output=True, text=True, timeout=10, encoding='utf-8'
        ).stdout
        
        root = ET.fromstring(xml_content)
        for node in root.iter():
            if node.get("class") == "android.widget.EditText":
                bounds = node.get("bounds")
                current_text = node.get("text", "")
                if not bounds: continue
                
                coords = bounds.replace("][", ",").strip("[]").split(",")
                x1, y1, x2, y2 = map(int, coords)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                uid = 1
                element_info = {
                    "uid": uid,
                    "center": (center_x, center_y),
                    "bounds": [x1, y1, x2, y2],
                    "text": current_text,
                    "class": "android.widget.EditText"
                }
                logger.info(f"找到输入框！UID={uid}, 中心=({center_x}, {center_y}), 当前文本: '{current_text}'")
                return {
                    "element": element_info,
                    "elements_list": [element_info]
                }
    except Exception as e:
        logger.error(f"查找输入框时出错: {e}")
    return None

# --- V2: 修改后的清空函数 ---
def clear_input_box(actuator: Actuator, ui_info: dict):
    """
    点击并清空输入框内容（更稳健的版本）。
    此方法通过移动光标到末尾并循环发送删除键来实现，不依赖"全选"快捷键。
    """
    logger.info("正在准备测试环境 (点击并清空输入框 - 稳健模式)...")
    element = ui_info["element"]
    elements_list = ui_info["elements_list"]
    current_text = element.get("text", "")

    # 1. 点击元素以获得焦点
    if not actuator.tap(element['uid'], elements_list):
        logger.warning(f"清空前点击 uid={element['uid']} 失败，但仍将继续尝试清空。")
    time.sleep(0.5)

    if not current_text:
        logger.info("输入框已为空，无需清空。")
        return

    # 2. 移动光标到文本末尾 (确保从后往前删)
    # KEYCODE_MOVE_END 的键值为 123
    actuator._execute_adb_command(["shell", "input", "keyevent", "123"])
    time.sleep(0.1)

    # 3. 循环发送删除键 (KEYCODE_DEL 的键值为 67)
    # 为了保险起见，我们多删除几个字符
    num_deletions = len(current_text) + 5
    logger.info(f"输入框内容长度为 {len(current_text)}，将发送 {num_deletions} 次删除键。")
    for _ in range(num_deletions):
        actuator._execute_adb_command(["shell", "input", "keyevent", "67"])

    logger.info(f"输入框 (uid={element['uid']}) 已尝试清空。")
    time.sleep(0.5)
# --- 修改结束 ---

def run_all_tests():
    """执行所有输入测试用例"""
    print("\n" + "="*80)
    print("=== Actuator.input_text() 全面测试脚本 (V2) ===")
    print("="*80 + "\n")

    actuator = Actuator(adb_path=ADB_PATH, device_serial=DEVICE_SERIAL)

    test_cases = [
        {"name": "纯英文字符", "text": "hello"},
        {"name": "英文与数字", "text": "test version 123"},
        {"name": "包含空格的句子", "text": "hello world from actuator"},
        {"name": "常用标点符号", "text": "hello, world! (is it working?)"},
        {"name": "特殊符号", "text": "`~!@#$%^&*()-_=+[]\{}|;:'\",./<>?"},
        {"name": "纯中文字符", "text": "你好世界"},
        {"name": "中英混合", "text": "这是一个测试 for Actuator V2"},
        {"name": "包含Emoji", "text": "你好😊, test OK👍"},
    ]
    
    passed_count = 0
    failed_count = 0

    for i, case in enumerate(test_cases):
        print("\n" + "-"*80)
        logger.info(f"--- TestCase {i+1}/{len(test_cases)}: {case['name']} ---")
        print("-" * 80)
        
        # 1. 查找输入框 (每次循环都重新查找，以获取最新的文本内容)
        ui_info = get_ui_dump_and_find_input(ADB_PATH, DEVICE_SERIAL)
        if not ui_info:
            logger.error("[测试中止] 未能在屏幕上找到任何输入框。请先打开一个记事本或任何有输入框的应用。")
            failed_count += (len(test_cases) - i) # 之后的所有测试都算失败
            break

        # 2. 清空输入框 (使用新的稳健方法)
        clear_input_box(actuator, ui_info)
        
        # 3. 执行核心测试
        expected_text = case["text"]
        logger.info(f"执行 input_text, 预期输入: '{expected_text}'")
        input_success = actuator.input_text(ui_info["element"]["uid"], expected_text, ui_info["elements_list"])
        
        if not input_success:
            logger.error("[失败] actuator.input_text() 方法返回 False。")
            failed_count += 1
            continue

        time.sleep(1) 

        # 4. 验证结果
        logger.info("正在验证输入结果...")
        result_ui_info = get_ui_dump_and_find_input(ADB_PATH, DEVICE_SERIAL)
        if not result_ui_info:
            logger.error("[失败] 验证失败，无法再次找到输入框。")
            failed_count += 1
            continue
        
        actual_text = result_ui_info["element"]["text"]

        # 5. 报告结果
        print("\n--- 测试结果 ---")
        print(f"预期得到: '{expected_text}'")
        print(f"实际得到: '{actual_text}'")
        if actual_text == expected_text:
            logger.info(f"✅✅✅ [ 通过 ] ✅✅✅")
            passed_count += 1
        else:
            logger.error(f"❌❌❌ [ 失败 ] ❌❌❌")
            failed_count += 1

    # 最终总结
    print("\n" + "="*80)
    print("=== 测试总结 ===")
    print(f"总计测试用例: {len(test_cases)}")
    print(f"✅ 通过: {passed_count}")
    print(f"❌ 失败: {failed_count}")
    print("="*80)

if __name__ == "__main__":
    run_all_tests()