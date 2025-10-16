# -*- coding: utf-8 -*-
import subprocess
import logging
import time
import sys
import xml.etree.ElementTree as ET
import shlex

# --- 请根据您的环境修改这里的配置 ---
ADB_PATH = "/home/zzh/Android/Sdk/platform-tools/adb"
DEVICE_SERIAL = "emulator-5554"
ADB_KEYBOARD_ID = "com.android.adbkeyboard/.AdbIME"
# ------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ADBKeyboard_Diagnose")

class AdbError(Exception):
    pass

def execute_command(command: list[str], timeout: int = 20, check=True) -> str:
    logger.info(f"执行: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding='utf-8',
            timeout=timeout, check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise AdbError(f"命令执行失败。Stderr: {e.stderr.strip()}")
    except Exception as e:
        raise AdbError(f"未知错误: {e}")

def get_ui_dump_and_find_input(adb_path: str, device: str) -> tuple[int, int, str] | None:
    logger.info("正在获取UI布局并查找输入框...")
    try:
        execute_command([adb_path, "-s", device, "shell", "uiautomator", "dump", "/sdcard/uidump.xml"])
        xml_content = execute_command([adb_path, "-s", device, "shell", "cat", "/sdcard/uidump.xml"])
        
        root = ET.fromstring(xml_content)
        for node in root.iter():
            if node.get("class") == "android.widget.EditText":
                bounds = node.get("bounds")
                current_text = node.get("text", "")
                if not bounds: continue
                coords = bounds.replace("][", ",").strip("[]").split(",")
                x1, y1, x2, y2 = map(int, coords)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                logger.info(f"找到输入框！Bounds: {bounds}, 中心: ({center_x}, {center_y}), 当前文本: '{current_text}'")
                return center_x, center_y, current_text
    except Exception as e:
        logger.error(f"查找输入框时出错: {e}")
    return None

def run_diagnosis():
    print("\n" + "="*80)
    print("=== ADBKeyboard 终极诊断脚本 ===")
    print("="*80 + "\n")

    # 1. 强制设置并验证输入法
    logger.info("步骤 1: 强制设置并验证 ADBKeyboard...")
    try:
        execute_command([ADB_PATH, "-s", DEVICE_SERIAL, "shell", "ime", "enable", ADB_KEYBOARD_ID], check=False)
        execute_command([ADB_PATH, "-s", DEVICE_SERIAL, "shell", "ime", "set", ADB_KEYBOARD_ID])
        output = execute_command([ADB_PATH, "-s", DEVICE_SERIAL, "shell", "settings", "get", "secure", "default_input_method"])
        if output != ADB_KEYBOARD_ID:
            logger.error(f"[失败] 无法将 ADBKeyboard 设为默认输入法。当前为: {output}")
            return
        logger.info("[成功] ADBKeyboard 已是默认输入法。")
    except AdbError as e:
        logger.error(f"设置输入法失败: {e}")
        return

    # 2. 查找输入框
    logger.info("\n步骤 2: 查找输入框...")
    result = get_ui_dump_and_find_input(ADB_PATH, DEVICE_SERIAL)
    if not result:
        logger.error("[失败] 未能在屏幕上找到任何输入框。请先打开一个记事本应用。")
        return
    center_x, center_y, _ = result
    
    # 3. 准备测试环境：点击并清空输入框
    logger.info("\n步骤 3: 准备测试环境 (点击并清空输入框)...")
    try:
        execute_command([ADB_PATH, "-s", DEVICE_SERIAL, "shell", "input", "tap", str(center_x), str(center_y)])
        time.sleep(0.5)
        # 清空文本的命令: 长按KEYCODE_MOVE_END(全选), 然后按KEYCODE_DEL(删除)
        execute_command([ADB_PATH, "-s", DEVICE_SERIAL, "shell", "input", "keyevent", "--longpress", "123"])
        execute_command([ADB_PATH, "-s", DEVICE_SERIAL, "shell", "input", "keyevent", "67"])
        logger.info("[成功] 输入框已准备就绪。")
    except AdbError as e:
        logger.error(f"准备输入框失败: {e}")
        return
    time.sleep(1)

    # 4. 执行核心测试：发送单个中文字符
    test_char = "测"
    logger.info(f"\n步骤 4: 执行核心测试 (通过 ADBKeyboard 发送字符 '{test_char}')...")
    try:
        safe_char = shlex.quote(test_char)
        cmd_str = f"am broadcast -a ADB_INPUT_TEXT --es msg {safe_char}"
        execute_command([ADB_PATH, "-s", DEVICE_SERIAL, "shell", cmd_str])
        logger.info("[成功] 广播命令已发送。")
    except AdbError as e:
        logger.error(f"发送广播失败: {e}")
        return
    time.sleep(1) # 等待输入法响应

    # 5. 验证结果
    logger.info("\n步骤 5: 验证结果 (检查输入框内容)...")
    result = get_ui_dump_and_find_input(ADB_PATH, DEVICE_SERIAL)
    if not result:
        logger.error("[失败] 验证失败，无法再次找到输入框。")
        return
    _, _, final_text = result

    print("\n" + "="*80)
    print("=== 诊断结果 ===")
    print(f"预期输入: '{test_char}'")
    print(f"实际得到: '{final_text}'")
    print("-" * 40)
    if final_text == test_char:
        print("🎉🎉🎉 [ 结论: 有效 ] 🎉🎉🎉")
        print("您的 ADBKeyboard.apk 工作正常！问题可能出在其他地方。")
    else:
        print("💥💥💥 [ 结论: 无效 ] 💥💥💥")
        print("您的 ADBKeyboard.apk 在此模拟器环境中无法正常工作。")
        print("根本原因极有可能是【CPU架构不匹配】(模拟器为x86_64, APK为ARM)。")
    print("="*80)
    
    if final_text != test_char:
        print("\n--- 推荐解决方案 ---")
        print("1. [首选] 从官方渠道下载或自行编译一个 x86/x86_64 版本的 ADBKeyboard.apk。")
        print("   官方 GitHub 仓库: https://github.com/senzhk/ADBKeyBoard")
        print("   您可以在该仓库的 'Releases' 页面寻找预编译的APK，或按照说明使用 Android Studio 自行编译。")
        print("2. 确保您的安卓模拟器使用的是较新的、主流的系统镜像。")


if __name__ == "__main__":
    run_diagnosis()