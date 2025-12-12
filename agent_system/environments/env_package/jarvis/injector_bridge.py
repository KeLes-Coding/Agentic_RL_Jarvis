import sys
import os
import tempfile
import logging
import subprocess
import time
import re
import importlib
import json

# ==============================================================================
# 1. 路径设置与环境准备
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INJECTOR_DIR = os.path.join(CURRENT_DIR, "android_env_injector")
DATA_DIR = os.path.join(INJECTOR_DIR, "data")
SOURCE_DIR = os.path.join(INJECTOR_DIR, "source") # [新增] 源文件的绝对路径

# 将注入器目录加入 Python 搜索路径
if INJECTOR_DIR not in sys.path:
    sys.path.insert(0, INJECTOR_DIR)

# ==============================================================================
# 2. Monkey Patching - 基础工具 (Utils)
# ==============================================================================
try:
    import utils 
    import config

    # --- 修复 1: 配置文件路径 ---
    def fixed_load_json_data(filename):
        json_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(json_path):
            print(f"[InjectorBridge] 错误: 配置文件未找到: {json_path}")
            return []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[InjectorBridge] 读取配置失败 {filename}: {e}")
            return []

    utils.load_json_data = fixed_load_json_data
    
    # --- 修复 2: 日志路径 ---
    config.LOG_ROOT_DIR = os.path.join(tempfile.gettempdir(), "jarvis_injector_logs")
    if not os.path.exists(config.LOG_ROOT_DIR):
        os.makedirs(config.LOG_ROOT_DIR)

except ImportError as e:
    print(f"[InjectorBridge] 基础模块导入失败: {e}")

# ==============================================================================
# 3. 导入并 Patch 业务模块
# ==============================================================================
try:
    from modules.system import clean_background_apps, go_home
    from modules.wizards import init_markor, init_expense, init_tasks
    from modules.injector import inject_calendar
    from modules.inject_tasks import inject_tasks_db
    from modules.inject_expense import inject_expense_db
    # from modules.inject_files import inject_files_from_manifest # 不直接用，我们要替换它
    import modules.inject_files # 导入模块对象以便替换
    from modules.inject_system import inject_contacts, inject_sms_msg
    from utils import run_adb, load_json_data # 导入已被 Patch 的 utils

    # --- 修复 3: 强制 inject_files 使用绝对路径 ---
    def patched_inject_files_from_manifest(device_id, temp_dir, logger):
        logger.info(">>> 注入通用文件 (Source -> Device) [Patched] <<<")
        
        # 使用我们修复过的 load_json_data
        manifest = load_json_data("files_manifest.json")
        if not manifest:
            logger.warning("未找到文件清单 files_manifest.json，跳过文件注入。")
            return

        for item in manifest:
            src_rel = item.get("source")
            remote_path = item.get("remote_path")
            metadata = item.get("metadata", {})
            
            if not src_rel or not remote_path:
                continue
                
            # [关键修改] 使用绝对路径 SOURCE_DIR
            src_path = os.path.join(SOURCE_DIR, src_rel)
            
            # 特殊处理：installer.zip 生成
            if "installer.zip" in src_rel and metadata.get("size_mb") and not os.path.exists(src_path):
                logger.info(f"生成虚拟文件: {src_path}")
                os.makedirs(os.path.dirname(src_path), exist_ok=True)
                size = metadata["size_mb"] * 1024 * 1024
                with open(src_path, "wb") as f: f.write(os.urandom(size))

            if not os.path.exists(src_path):
                logger.warning(f"源文件缺失: {src_path} -> {remote_path}")
                continue
                
            remote_dir = os.path.dirname(remote_path)
            run_adb(device_id, ["shell", f"mkdir -p {remote_dir}"], logger=logger)
            run_adb(device_id, ["push", src_path, remote_path], logger=logger)
            
            if "touch_time" in metadata:
                ts = metadata["touch_time"]
                run_adb(device_id, ["shell", f"touch -t {ts} {remote_path}"], logger=logger)
                
        logger.info("刷新媒体扫描...")
        run_adb(device_id, ["shell", "am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/"], logger=logger)
        logger.info("文件注入完成。")

    # 执行替换
    modules.inject_files.inject_files_from_manifest = patched_inject_files_from_manifest
    print(f"[InjectorBridge] 已修复文件注入路径: {SOURCE_DIR}")

except ImportError as e:
    print(f"[InjectorBridge] 注入模块导入失败: {e}")

# ==============================================================================
# 4. Bridge 类定义
# ==============================================================================
class AndroidInjector:
    def __init__(self, adb_path: str):
        self.adb_path = adb_path
        if 'config' in sys.modules:
            config.ADB_PATH = self.adb_path
            print(f"[InjectorBridge] ADB路径重定向为: {config.ADB_PATH}")

    def _setup_dummy_logger(self, device_id):
        logger = logging.getLogger(f"bridge_{device_id}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'[Injector-{device_id}] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _fix_permissions_robust(self, device_id, logger):
        apps_to_fix = [
            ("com.simplemobiletools.calendar.pro", "databases"),
            ("org.tasks", "databases"),
            ("com.arduia.expense", "databases"),
            ("net.gsantner.markor", "files") 
        ]
        
        for pkg, rel_path in apps_to_fix:
            try:
                res = subprocess.run([self.adb_path, "-s", device_id, "shell", f"pm list packages -U {pkg}"], 
                                     capture_output=True, text=True)
                match = re.search(r"uid:(\d+)", res.stdout)
                if match:
                    uid = match.group(1)
                    base_path = f"/data/data/{pkg}"
                    target_path = f"{base_path}/{rel_path}"
                    cmds = [
                        f"chown -R {uid}:{uid} {base_path}",
                        f"chmod 770 {base_path}",
                        f"chmod -R 770 {target_path}",
                        f"restorecon -R {base_path}"
                    ]
                    for cmd in cmds:
                        subprocess.run([self.adb_path, "-s", device_id, "shell", cmd], capture_output=True)
            except Exception as e:
                logger.error(f"修复 {pkg} 权限时出错: {e}")

    def reset_environment(self, device_id: str):
        logger = self._setup_dummy_logger(device_id)
        try:
            # 1. Root
            run_adb(device_id, ["root"], logger=logger)
            time.sleep(1)

            # 2. 清理
            clean_background_apps(device_id, logger, exclude_pkgs=[])

            # 3. 初始化
            init_markor(device_id, logger)
            init_expense(device_id, logger)
            init_tasks(device_id, logger)

            # 4. 注入
            with tempfile.TemporaryDirectory() as temp_dir:
                inject_calendar(device_id, temp_dir, logger)
                inject_tasks_db(device_id, temp_dir, logger)
                inject_expense_db(device_id, temp_dir, logger)
                
                # 调用被我们替换后的函数
                modules.inject_files.inject_files_from_manifest(device_id, temp_dir, logger)
                
                inject_contacts(device_id, logger)
                inject_sms_msg(device_id, temp_dir, logger)

            # 5. 权限修复
            self._fix_permissions_robust(device_id, logger)

            # 6. 收尾清理
            pkg_calendar = getattr(config, 'PKG_CALENDAR', "com.simplemobiletools.calendar.pro")
            pkg_tasks = getattr(config, 'PKG_TASKS', "org.tasks")
            pkg_expense = getattr(config, 'PKG_EXPENSE', "com.arduia.expense")
            pkg_markor = getattr(config, 'PKG_MARKOR', "net.gsantner.markor")
            
            exclude_list = [
                pkg_calendar, pkg_tasks, pkg_expense, pkg_markor,
                "com.android.providers.telephony",
                "com.android.providers.contacts",
                "com.google.android.apps.messaging",
                "com.android.dialer"
            ]
            clean_background_apps(device_id, logger, exclude_pkgs=exclude_list)
            
            # 7. 回到桌面
            go_home(device_id, logger)

        except Exception as e:
            logger.error(f"环境重置失败: {e}")
            import traceback
            traceback.print_exc()