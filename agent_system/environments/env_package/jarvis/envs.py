# agent_system/environments/env_package/jarvis/envs.py

import yaml
import numpy as np
import io
from typing import List, Dict, Tuple, Union, Any
import logging
import re
import openai
import subprocess 
from concurrent.futures import ThreadPoolExecutor, Future # <<< 新增：导入线程池
import base64  # <<< ✅ 升级：新增导入 base64 >>>

try:
    from PIL import Image
except ImportError:
    Image = None

from .jarvis_v2.jarvis.modules.observer import Observer
from .jarvis_v2.jarvis.modules.actuator import Actuator
from .jarvis_v2.agent_manager import discover_devices

from .injector_bridge import AndroidInjector

def _evaluate_with_llm(
    summary: str, 
    ground_truth: str, 
    final_layout: str,              # <<< ✅ 升级：新增参数
    final_image_bytes: bytes,       # <<< ✅ 升级：新增参数
    llm_config: Dict[str, Any]
) -> bool:
    """
    使用外部LLM评估任务摘要与参考答案。
    <<< 升级：现在还会传入最终的截图和界面布局，以验证 summary 是否基于当前屏幕状态。>>>
    """
    # --- 增强健壮性：在调用API前检查输入 ---
    if not summary or not ground_truth:
        print("警告: LLM评估因为 summary 或 ground_truth 为空而被跳过。")
        return False
    
    # --- ✅ 升级：新增检查 final_layout 和 final_image_bytes ---
    if not final_layout:
        print("警告: LLM评估因为 final_layout (界面布局) 为空而被跳过。")
        return False
    if not final_image_bytes:
        print("警告: LLM评估因为 final_image_bytes (最终截图) 为空而被跳过。")
        return False
    
    if not llm_config or not all(k in llm_config for k in ['key', 'url', 'model']):
        print("警告: LLM评估因为 llm_config 配置不完整而被跳过。")
        return False

    # --- ✅ 升级：将图像字节编码为 Base64 ---
    try:
        image_b64 = base64.b64encode(final_image_bytes).decode('utf-8')
        
        # 简单的 MIME 类型嗅探
        if final_image_bytes.startswith(b'\x89PNG'):
            image_mime_type = "image/png"
        else:
            image_mime_type = "image/jpeg" # 默认为 JPEG (因为压缩配置默认为 JPEG)
        
        image_url = f"data:{image_mime_type};base64,{image_b64}"
        
    except Exception as e:
        print(f"LLM评估期间编码图像失败: {e}")
        return False

    try:
        client = openai.OpenAI(
            api_key=llm_config.get('key'),
            base_url=llm_config.get('url'),
        )

        # --- ✅ 升级：构建新的多模态 Prompt ---
        prompt_text = f"""
请扮演一个严格的UI自动化任务评估专家。你需要根据 [参考答案] 和 [最终屏幕截图/布局] 来严格审核 [任务摘要]。

你的核心目标是拦截那些“虽然完成了任务但没有包含具体结果”的无效摘要。

### 评估标准 (必须同时严格满足，否则为 False):

1. **信息具体性与包含度 (Crucial)**:
   - [任务摘要] 必须 **显式包含** [参考答案] 中的核心关键信息（如：具体金额、日期、特定的名称、生成的ID号等）。
   - **拒绝模糊表述**：如果摘要仅包含“我已经完成了购买”、“任务成功”、“信息已确认”等宽泛的描述，而未提及参考答案中的具体数值或内容，必须判定为 **False**。

2. **视觉一致性 (Grounding)**:
   - [任务摘要] 中声称的事实（如“价格是$50”）必须能在 [最终屏幕截图/布局] 中找到直接证据。
   - 如果摘要中的信息与截图显示不符，或者截图中根本不存在该信息，判定为 **False**。

### 输入数据

[参考答案]
{ground_truth}

[任务摘要]
{summary}

[最终屏幕截图/布局]
(参考附加截图及以下布局文本)
{final_layout}

---
### 评估结果要求
请基于以上标准进行二元判断。
"任务摘要" 是否同时满足 **信息具体性** 和 **视觉一致性**？

请仅回答 "True" 或 "False"。
"""
        # --- ✅ 升级：构建多模态 messages 列表 ---
        messages = [
            {"role": "system", "content": "你是一个评估任务完成情况的助手。你需要结合图像和文本信息来判断任务是否真正完成了。"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "low" # 使用 "low" 以节省 token 和加快速度
                        }
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model=llm_config['model'], # 确保这个模型 (例如 gpt-4o, qwen-vl-max) 支持 VLM
            messages=messages,
            temperature=0,
            max_tokens=10, # 只需要 True/False
        )
        content = response.choices[0].message.content.strip()
        return content.lower() == 'true'
    except Exception as e:
        # 检查是否是 OpenAI API 错误，提示用户模型可能不支持 VLM
        if "vision" in str(e).lower() or "image" in str(e).lower():
            print(f"LLM评估期间出错: {e}. (提示: 您配置的模型 '{llm_config['model']}' 可能不支持图像/视觉输入。)")
        else:
            print(f"LLM评估期间出错: {e}")
        return False


class JarvisMultiDeviceEnv:
    """
    一个底层的、支持多设备的 Jarvis 环境。
    它封装了与一组安卓设备的直接交互 (reset, step)。
    这个类不处理复杂的 prompt 构建，只提供原始观测数据。
    
    <<< 修改：此类现在使用 ThreadPoolExecutor 来并行执行 reset 和 step 操作。>>>
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
        self.adb_path = adb_path 

        self.env_injector = AndroidInjector(adb_path)

        self.observers: Dict[str, Observer] = {s: Observer(adb_path, s) for s in self.device_serials}
        self.actuators: Dict[str, Actuator] = {s: Actuator(adb_path, s) for s in self.device_serials}

        # [新增] 观察缓存，用于避免 step 中重复获取
        self.obs_cache: Dict[str, Dict] = {}

        # [关键] 初始化线程池和Future列表
        self.executor = ThreadPoolExecutor(max_workers=self.num_envs, thread_name_prefix="jarvis_env_worker")
        self.reset_futures: List[Future] = []
        print(f"ThreadPoolExecutor (线程池) 已启动，设置 max_workers={self.num_envs} 以实现并行I/O。")

        self.max_steps_per_episode = max_steps_per_episode
        self.episode_steps: Dict[str, int] = {s: 0 for s in self.device_serials}

        agent_config = self.jarvis_config.get("agent", {})
        self.compression_config = agent_config.get("image_compression", {})

        self.max_elements_per_obs = agent_config.get("max_elements_per_obs", 70)
        self.max_str_len_per_obs = agent_config.get("max_str_len_per_obs", 10000)
        print(f"=== UI观察截断已启用: max_elements={self.max_elements_per_obs}, max_str_len={self.max_str_len_per_obs} ===")

        self.llm_config = self.jarvis_config.get("evaluation_llm", {})
        self.tasks: Dict[str, Dict] = {s: {} for s in self.device_serials}
        if self.llm_config:
            print("=== LLM评估奖励已启用。===")
        else:
            print("警告: 在 Jarvis 配置中未找到 'llm' 配置，任务成功与否的奖励将无法计算。")

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

    # ======================= ✅ 新增：清理后台应用的方法 ✅ =======================
    def _clear_background_apps(self, serial: str):
        """
        清理指定设备上的所有非核心应用数据，将它们重置为初始状态。
        此方法会强制停止应用，然后清除其所有数据（缓存和用户数据）。
        会自动跳过一个预定义的安全核心系统应用列表。
        """
        print(f"--- [设备: {serial}] 开始全面清理应用数据... ---")
        adb_command_base = [self.adb_path, "-s", serial, "shell"]

        # 定义一个正则表达式列表，用于匹配不应被清理的核心/安全包名。
        # 这包括系统UI、设置、启动器、ADB键盘、Google Play服务等关键组件。
        SAFE_PACKAGES_REGEX = [
            r"^com\.android\.adbkeyboard$",      # ADB Keyboard (根据原有逻辑保留)
            r"^com\.android\.systemui$",         # 系统 UI
            r"^com\.android\.settings$",         # 设置
            r".*launcher.*",                   # 任何包含 "launcher" 的包名
            r"^com\.google\.android\.gms$",      # Google Play 服务
            r"^com\.android\.vending$",          # Google Play 商店 (Vending)
            r"^android$",                        # 核心操作系统包
        ]

        # 1. 获取所有已安装的包名，而不仅仅是第三方包
        try:
            list_packages_cmd = adb_command_base + ["pm", "list", "packages"]
            result = subprocess.run(list_packages_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
            packages_output = result.stdout.strip()
            all_packages = [line.split(":")[-1] for line in packages_output.splitlines() if line.startswith("package:")]
        except subprocess.CalledProcessError as e:
            print(f"--- [设备: {serial}] 错误: 获取所有包列表失败 - {e.stderr} ---")
            return
        except Exception as e:
            print(f"--- [设备: {serial}] 错误: 解析包列表时发生意外错误 - {e} ---")
            return

        if not all_packages:
            print(f"--- [设备: {serial}] 未找到任何应用包，跳过清理。 ---")
            return

        cleared_count = 0
        skipped_count = 0

        for package_name in all_packages:
            if not package_name:
                continue

            # 2. 检查包是否在安全列表中，如果在则跳过
            is_safe = any(re.search(pattern, package_name) for pattern in SAFE_PACKAGES_REGEX)
            if is_safe:
                skipped_count += 1
                continue

            # 3. 执行强制停止和数据清理
            # 首先强制停止应用，确保它没有在运行
            try:
                force_stop_cmd = adb_command_base + ["am", "force-stop", package_name]
                subprocess.run(force_stop_cmd, capture_output=True, check=False, timeout=10) # check=False 忽略停止失败
            except Exception:
                # 强制停止失败通常不是严重问题（例如，应用可能已经停止），所以忽略异常
                pass

            # 然后清理应用的所有数据（包括缓存），实现重置效果
            try:
                clear_data_cmd = adb_command_base + ["pm", "clear", package_name]
                subprocess.run(clear_data_cmd, capture_output=True, text=True, check=True, timeout=20)
                # print(f"--- [设备: {serial}] 成功清理应用: {package_name} ---") # 可以取消注释以进行详细调试
                cleared_count += 1
            except subprocess.CalledProcessError as e:
                # 记录清理失败的警告，这可能因为权限问题或应用是不可变的
                print(f"--- [设备: {serial}] 警告: 清理应用 {package_name} 数据时出现问题 - {e.stderr.strip()} ---")
            except subprocess.TimeoutExpired:
                print(f"--- [设备: {serial}] 错误: 清理应用 {package_name} 数据时超时。 ---")
            except Exception as e:
                print(f"--- [设备: {serial}] 错误: 清理应用 {package_name} 数据时发生意外错误 - {e} ---")

        print(f"--- [设备: {serial}] 应用数据清理完成。清理数量: {cleared_count}, 跳过 (核心/安全) 数量: {skipped_count} ---")
    # ===========================================================================

    # [方法 1: 修正] 辅助方法，用于匹配 SSH 配置 (包含对 main 节点的修复)
    def _get_ssh_config_for_device(self, serial: str) -> Dict:
        """
        根据设备 serial (例如 localhost:15555) 查找对应的 SSH 配置。
        修正：兼容 config.yaml 中存在 'main' 嵌套层级的情况。
        """
        try:
            port = int(serial.split(":")[-1])
        except ValueError:
            return None

        # [修正点] 获取 device_providers 配置节点
        providers = self.jarvis_config.get("device_providers", {})
        if not providers and "main" in self.jarvis_config:
            providers = self.jarvis_config["main"].get("device_providers", {})

        ssh_configs = providers.get("ssh_forward_tunnel", {}).get("ssh_connections", [])
        
        best_match = None
        for config in ssh_configs:
            start_port = config.get("local_start_port", 0)
            if start_port <= port:
                if best_match is None or start_port > best_match.get("local_start_port", 0):
                    best_match = config
        return best_match

    # [方法 2: 核心逻辑] 阻塞式等待所有服务器完成重置
    def _trigger_remote_batch_reset(self):
        """
        通过 SSH 触发所有远程服务器执行本地的 inject_env 脚本。
        【关键同步机制】：此方法会阻塞，直到所有远程服务器返回 "ALL_DONE" 信号或超时。
        只有此方法成功返回，后续的 Rollout 才会开始。
        """
        print("--- [Remote Reset] 正在握手远程服务器，等待 'ALL_DONE' 信号... ---")
        
        unique_hosts = {} 
        for serial in self.device_serials:
            config = self._get_ssh_config_for_device(serial)
            if config:
                key = f"{config['ssh_user']}@{config['ssh_host']}:{config.get('ssh_port', 22)}"
                unique_hosts[key] = config
        
        if not unique_hosts:
            print("--- [Remote Reset] 未发现 SSH 隧道设备，跳过。 ---")
            return

        def _exec_ssh_and_wait(host_key, config):
            user = config['ssh_user']
            host = config['ssh_host']
            port = str(config.get('ssh_port', 22))
            remote_script_dir = config.get("remote_project_path", "~/agentic_rl_jarvis/KeLes-Coding/-inject_env")
            
            # 远程命令：进入目录 -> 执行脚本 (带强制参数)
            remote_cmd = f"cd {remote_script_dir} && python3 main.py --force"
            
            ssh_cmd = [
                "ssh", 
                "-p", port,
                "-o", "StrictHostKeyChecking=no",
                f"{user}@{host}",
                remote_cmd
            ]
            
            # print(f"--- [Remote Reset] Waiting for {host}... ---")
            try:
                # [核心等待点] capture_output=True 会等待进程结束并获取输出
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=180) # 给3分钟超时
                
                # [关键校验] 检查返回码 AND 输出中是否包含成功信号
                if result.returncode == 0 and "ALL_DONE" in result.stdout:
                    print(f"--- [Remote Reset] ✅ {host} 信号确认: ALL_DONE ---")
                    return True
                else:
                    print(f"--- [Remote Reset] ❌ {host} 失败或无信号!\nCode: {result.returncode}\nOut: {result.stdout}\nErr: {result.stderr} ---")
                    return False
            except subprocess.TimeoutExpired:
                print(f"--- [Remote Reset] ❌ {host} 等待信号超时! ---")
                return False
            except Exception as e:
                print(f"--- [Remote Reset] ❌ {host} 连接异常: {e} ---")
                return False

        # 并行等待所有服务器
        all_success = True
        with ThreadPoolExecutor(max_workers=len(unique_hosts)) as ssh_executor:
            futures = {ssh_executor.submit(_exec_ssh_and_wait, k, v): k for k, v in unique_hosts.items()}
            for future in futures:
                if not future.result():
                    all_success = False

        if not all_success:
            print("--- [Remote Reset] ⚠️  警告: 部分服务器重置失败，后续 Rollout 可能受影响。 ---")
        else:
            print("--- [Remote Reset] 🎉 所有服务器重置完毕，准备开启 Rollout (Observation)。 ---")

    # [方法 3: 流程控制] 先等待远程，再开启本地
    def start_background_reset(self):
        """
        [关键] 在当前 Rollout 结束后调用。
        流程：
        1. (Blocking) 触发并等待远程服务器完成物理重置 (inject_env)。
        2. (Async) 远程准备好后，提交本地的 Observation 获取任务。
        """
        if self.reset_futures:
            print("--- 警告: 后台重置已在运行中，跳过重复触发 ---")
            return

        print(f"--- [Sync-Barrier] 🛑 正在等待 {self.num_envs} 台设备的远程环境重置... ---")
        
        # 1. 同步屏障：在此处阻塞，直到收到所有服务器的 ALL_DONE
        # 虽然这会暂停主线程一小会，但保证了 RL 在干净的环境中开始
        try:
            self._trigger_remote_batch_reset()
        except Exception as e:
            print(f"--- [Sync-Barrier] 严重错误: 远程重置触发失败: {e} ---")

        # 2. 只有在上面执行完后，才开始本地的 Rollout 任务 (Home + Get Obs)
        print(f"--- [Async-Rollout] 🚀 环境已就绪，正在后台获取初始 Observation... ---")
        self.reset_futures = []
        for serial in self.device_serials:
            # task=None 表示只做 Observation 获取
            self.reset_futures.append(self.executor.submit(self._reset_device, serial, None))

    # [方法 3: 修改] 单机 Reset 逻辑，跳过物理注入
    def _reset_device(self, serial: str, task: Dict = None) -> Tuple[np.ndarray, str, Dict]:
        """
        (辅助函数) 在单独的线程中重置单个设备。
        如果 task 为 None，则只执行物理环境重置（注入/清理），不更新 self.tasks 状态。
        """
        mode = "BACKGROUND" if task is None else "FOREGROUND"
        try:
            # 1. 物理重置 (优化点：如果是 SSH 设备，跳过本地注入，假设远程脚本已完成)
            ssh_config = self._get_ssh_config_for_device(serial)
            if not ssh_config and hasattr(self, 'env_injector'):
                 # 仅当它是本地设备或者是未配置 SSH 映射的设备时，才跑原来的慢速逻辑
                 print(f"--- [设备: {serial}] 使用本地慢速重置 (非SSH设备) ---")
                 self.env_injector.reset_environment(serial)
            else:
                # 远程设备：什么都不做，等待下面的 UI 刷新
                pass

            # 2. 状态更新 (仅在前台重置时进行)
            if task is not None:
                self.episode_steps[serial] = 0
                self.tasks[serial] = task if task else {}

            # 3. 回到桌面并获取观察
            self.actuators[serial].home() 
            try:
                # 稍微等待一下，确保远程重置后的应用冷启动完成（如果脚本里杀了进程）
                self.actuators[serial].wait(1.5) 
            except Exception as e:
                print(f"--- [设备: {serial}] 在reset的等待期间发生错误: {e} ---")

            obs_data = self.observers[serial].get_current_observation(
                max_elements=self.max_elements_per_obs,
                max_str_len=self.max_str_len_per_obs
            )

            # [新增] 将初始观察存入缓存
            self.obs_cache[serial] = obs_data

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

            image_placeholders = "<image>\n"
            obs_text = obs_data.get("simplified_elements_str", "")
            final_obs_text = f"{image_placeholders}{obs_text}"

            info_dict = {
                "device_serial": serial,
                "raw_obs_data": obs_data,
                "compressed_screenshot_bytes": compressed_bytes
            }
            
            return final_image_array, final_obs_text, info_dict
        
        except Exception as e:
            print(f"--- [设备: {serial}] 线程 'reset' ({mode}) 失败: {e} ---")
            return (
                np.zeros((256, 256, 3), dtype=np.uint8), 
                "<image>\nERROR: Reset failed.", 
                {"device_serial": serial, "raw_obs_data": {}, "compressed_screenshot_bytes": None, "error": str(e)}
            )

    def reset(self, tasks: List[Dict] = None) -> Tuple[Dict[str, List], List[Dict]]:
        obs_images = []
        obs_texts = []
        infos = []

        device_tasks = []
        if tasks:
            device_tasks = [tasks[i] if i < len(tasks) else {} for i in range(self.num_envs)]
        else:
            device_tasks = [{} for _ in range(self.num_envs)]

        # --- 策略 A: 优先使用后台重置结果 (Pipeline 模式) ---
        if self.reset_futures:
            print(f"--- [Reset] ♻️  检测到 {len(self.reset_futures)} 个后台重置任务，正在等待完成并获取结果... ---")
            for i, future in enumerate(self.reset_futures):
                try:
                    # 获取结果 (如果后台还没跑完，这里会阻塞)
                    img, text, info = future.result() 
                    
                    # [关键] 补全任务状态 (因为后台重置时没有任务信息)
                    serial = self.device_serials[i]
                    self.tasks[serial] = device_tasks[i]
                    self.episode_steps[serial] = 0
                    
                    obs_images.append(img)
                    obs_texts.append(text)
                    infos.append(info)
                except Exception as e:
                    print(f"--- [Reset] 获取后台结果失败: {e}")
                    obs_images.append(np.zeros((256, 256, 3), dtype=np.uint8))
                    obs_texts.append("<image>\nERROR")
                    infos.append({"device_serial": "unknown", "error": str(e)})
            
            # 清空列表，为下一轮做准备
            self.reset_futures = []

        # --- 策略 B: 执行标准同步重置 (首轮运行或无后台任务时) ---
        else:
            print("--- [Reset] ⚠️  无后台任务 (First Run / Cold Start)，正在执行同步阻塞式重置... ---")
            
            # [核心修复点] 必须在这里显式触发远程重置！
            # 否则 _reset_device 里的 SSH 分支会直接 pass，导致“假重置”
            try:
                self._trigger_remote_batch_reset()
            except Exception as e:
                print(f"--- [Reset] 远程批量重置触发失败: {e} ---")

            # 远程重置完成后，再并行采集 Observation
            futures: List[Future] = []
            for i, serial in enumerate(self.device_serials):
                task = device_tasks[i]
                futures.append(self.executor.submit(self._reset_device, serial, task))

            for future in futures:
                try:
                    img, text, info = future.result()
                    obs_images.append(img)
                    obs_texts.append(text)
                    infos.append(info)
                except Exception as e:
                    print(f"--- 严重错误: 'reset' 线程 'future.result()' 失败: {e} ---")
                    obs_images.append(np.zeros((256, 256, 3), dtype=np.uint8))
                    obs_texts.append("<image>\nERROR: Future result failed.")
                    infos.append({"device_serial": "unknown", "error": str(e)})

        return {"image": obs_images, "text": obs_texts}, infos
        
    # ======================= ✅ 修改：_handle_finish_action 方法签名和调用 ✅ =======================
    def _handle_finish_action(
        self, 
        action_str: str, 
        serial: str, 
        final_layout: str,              # <<< ✅ 升级：新增参数
        final_image_bytes: bytes        # <<< ✅ 升级：新增参数
    ) -> Tuple[float, bool]:
        """
        处理 finish 动作，包括解析、评估和奖励计算。
        返回 (奖励, 任务是否完成) 的元组。
        """
        summary = ""
        try:
            # 1. 优先尝试用正则表达式精确提取
            # re.DOTALL 允许 '.' 匹配换行符
            match = re.search(r"summary=['\"](.*?)['\"]", action_str, re.DOTALL)
            if match:
                summary = match.group(1).strip()
            else:
                # 2. 如果正则失败，使用更宽松的回退方法提取括号内的内容
                start_index = action_str.find('(')
                end_index = action_str.rfind(')')
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    potential_summary = action_str[start_index + 1:end_index].strip()
                    # 尝试去除可能存在的 "summary=" 前缀
                    if potential_summary.lower().strip().startswith("summary="):
                        summary = potential_summary[len("summary="):].strip().strip("'\" ")
                    else:
                        # 如果没有 "summary=" 前缀，就认为整个括号内容都是摘要
                        summary = potential_summary.strip("'\" ")

            if not summary:
                print(f"--- [设备: {serial}] 错误: 无法从 'finish' 动作中解析出有效的 summary。 Action: '{action_str}' ---")
                return 0.0, False

            # 3. 获取当前任务的 ground_truth 和 llm_config
            task_info = self.tasks.get(serial, {})
            ground_truth = task_info.get("ground_truth_answer")

            # 4. 检查评估所需的所有信息是否都存在
            if not ground_truth:
                print(f"--- [设备: {serial}] 警告: 缺少 'ground_truth_answer'，无法进行LLM评估。奖励设为0。 ---")
                return 0.0, False
            if not self.llm_config:
                print(f"--- [设备: {serial}] 警告: 缺少 'llm_config'，无法进行LLM评估。奖励设为0。 ---")
                return 0.0, False

            # 5. --- ✅ 升级：使用带重试机制和 VLM 的 LLM 评估 ---
            task_completed = _evaluate_with_llm(
                summary, 
                ground_truth, 
                final_layout,          # <<< ✅ 升级：传入
                final_image_bytes,     # <<< ✅ 升级：传入
                self.llm_config
            )

            # 6. 根据评估结果设置奖励
            reward = 1.0 if task_completed else 0.0
            print(f"--- [设备: {serial}] 'finish' 动作评估完成 (VLM)。Summary: '{summary}'. 任务是否成功: {task_completed}, 奖励: {reward} ---")

            return reward, task_completed

        except Exception as e:
            print(f"--- [设备: {serial}] 在 'finish' 动作处理期间发生严重错误: {e} ---")
            return 0.0, False # 发生任何异常都返回失败状态和0奖励
    # =======================================================================================

    # ======================= ✅ 新增：动态生成针对性 Feedback 的方法 ✅ =======================
    def _get_targeted_feedback(self, action_str: str, status: str) -> str:
        """
        根据失败的动作和状态，生成有针对性的反馈提示。
        """
        base_feedback = (
            "SYSTEM FEEDBACK: Your last action was not successful.\n"
            "Please follow the required JSON format: {\"thought\": \"your reasoning\", \"action\": \"your_action(...)\"}.\n"
        )

        action_examples = {
            "tap": "e.g., `tap(12)` to tap the element with uid 12.",
            "input_text": "e.g., `input_text(5, 'hello world')` to type 'hello world' into the element with uid 5.",
            "clear_text": "e.g., `clear_text(8)` to clear text from the element with uid 8.",
            "swipe": "e.g., `swipe(\"UP\", \"MEDIUM\")` to scroll down a list.",
            "enter": "e.g., `enter()` to press the enter key.",
            "back": "e.g., `back()` to go to the previous screen.",
            "home": "e.g., `home()` to return to the home screen.",
            "wait": "e.g., `wait(3.0)` to wait for 3 seconds.",
            "finish": "e.g., `finish(summary='Task is complete.')` to end the episode with a summary."
        }

        # Case 1: JSON 格式错误
        if action_str.startswith("format_error"):
            reason = action_str[len("format_error(reason='"):-2]
            return (
                f"SYSTEM FEEDBACK: Your last output had a JSON format error.\n"
                f"Error Details: {reason}\n"
                "You MUST respond in a strict JSON format. Example:\n"
                "{\"thought\": \"I need to tap the login button.\", \"action\": \"tap(21)\"}\n"
                "Please correct your output and try again.\n\n"
            )

        # Case 2: 动作执行失败
        action_name = action_str.split("(")[0].strip()

        feedback = (
            f"SYSTEM FEEDBACK: Your previous action failed to execute.\n"
            f"Action Sent: `{action_str}`\n"
            f"Execution Status: {status}\n"
        )

        # 提供针对性的示例
        if action_name in action_examples:
            feedback += f"Correct format for `{action_name}`: {action_examples[action_name]}\n"

        feedback += "Please analyze the error, check your action format and parameters, and try again.\n\n"

        return feedback
    # ====================================================================================

    # <<< ✅ 升级：修改 _step_device 方法以传递 VLM 所需信息 ✅ >>>
    def _step_device(self, serial: str, action_str: str) -> Tuple[np.ndarray, str, float, bool, Dict]:
        """
        (辅助函数) 在单独的线程中执行单个设备的 step。
        返回 (obs_image, obs_text, reward, done, info)
        """
        try:
            # [优化] 1. 获取 Pre-Action Observation (优先从缓存取)
            if serial in self.obs_cache:
                pre_action_obs_data = self.obs_cache[serial]
            else:
                # 缓存未命中（理论上不应发生），回退到主动获取
                print(f"--- [设备: {serial}] 警告: 缓存未命中，主动获取 pre-action obs ---")
                pre_action_obs_data = self.observers[serial].get_current_observation(
                    max_elements=self.max_elements_per_obs,
                    max_str_len=self.max_str_len_per_obs
                )

            elements = pre_action_obs_data.get("simplified_elements_list")

            status = self._dispatch_action(self.actuators[serial], serial, action_str, elements)
            action_success = (status == "SUCCESS")
            self.episode_steps[serial] += 1

            post_action_obs_data = self.observers[serial].get_current_observation(
                max_elements=self.max_elements_per_obs,
                max_str_len=self.max_str_len_per_obs
            )

            # [优化] 4. 更新缓存，供下一步骤作为 pre-action 使用
            self.obs_cache[serial] = post_action_obs_data

            screenshots_bytes = post_action_obs_data.get("screenshot_bytes")
            if not isinstance(screenshots_bytes, list):
                screenshots_bytes = [screenshots_bytes] if screenshots_bytes else []

            final_image_array = None
            compressed_bytes_post_action = None # <<< ✅ 升级：这是 VLM 需要的截图
            if screenshots_bytes:
                first_shot_bytes = screenshots_bytes[0]
                compressed_bytes_post_action = self._compress_single_image(first_shot_bytes)
                if compressed_bytes_post_action:
                    try:
                        img = Image.open(io.BytesIO(compressed_bytes_post_action)).convert("RGB")
                        final_image_array = np.array(img, dtype=np.uint8)
                    except Exception as e:
                        print(f"警告: 图像解码失败 - {e}")

            if final_image_array is None:
                final_image_array = np.zeros((256, 256, 3), dtype=np.uint8)

            feedback_prefix = ""

            # ======================= (这部分 feedback 逻辑不变) =======================
            if action_str.startswith("format_error") or not action_success:
                feedback_prefix = self._get_targeted_feedback(action_str, status)
            # ===============================================================================

            if action_str.startswith("swipe"):
                swipe_reminder = (
                    "\n--- SWIPE ACTION TIP ---\n"
                    "Remember the rule for vertical scrolling:\n"
                    "  - To see content BELOW (scroll down), use `swipe(\"UP\")`.\n"
                    "  - To see content ABOVE (scroll up), use `swipe(\"DOWN\")`.\n"
                )
                feedback_prefix += swipe_reminder

            image_placeholders = "<image>\n"
            obs_text_content = post_action_obs_data.get("simplified_elements_str", "") # <<< ✅ 升级：这是 VLM 需要的布局
            final_obs_text = f"{feedback_prefix}{image_placeholders}{obs_text_content}"

            done = False
            reward = 0.0
            task_completed = False

            if action_str.startswith("finish"):
                done = True
                # --- ✅ 升级：传入 VLM 所需的参数 ---
                reward, task_completed = self._handle_finish_action(
                    action_str, 
                    serial, 
                    obs_text_content,                 # 传入
                    compressed_bytes_post_action    # 传入
                )

            elif not action_success:
                # 动作失败的惩罚
                reward = -0.1

            if self.episode_steps[serial] >= self.max_steps_per_episode:
                done = True
                if not action_str.startswith("finish"):
                    # 超时，任务未完成
                    task_completed = False
            
            # 动作成功但未结束的微小正奖励（可选，目前为0）
            # if action_success and not done:
            #     reward = 0.01 

            compressed_bytes_pre_action = self._compress_single_image(pre_action_obs_data.get("screenshot_bytes", b''))
            
            info_dict = {
                "device_serial": serial,
                "action_success": action_success,
                "won": task_completed,
                "task_completed": task_completed,
                "raw_obs_data": pre_action_obs_data, # info 中通常携带动作前的观察
                "compressed_screenshot_bytes": compressed_bytes_pre_action,
                # --- ✅ [CCAPO V3] 关键修正：将 status 传递出去 ---
                "action_status": status
            }
            
            return final_image_array, final_obs_text, float(reward), bool(done), info_dict

        except Exception as e:
            print(f"--- [设备: {serial}] 线程 'step' 失败: {e} ---")
            # 返回一个表示失败的空状态
            return (
                np.zeros((256, 256, 3), dtype=np.uint8), 
                f"<image>\nERROR: Step failed due to {e}", 
                -0.1,  # 惩罚
                True,  # 终止这个出问题的 episode
                {"device_serial": serial, "action_success": False, "won": False, "task_completed": False, "error": str(e)}
            )

    # <<< 修改：`step` 方法现在使用线程池 >>>
    def step(self, actions: List[str]) -> Tuple[Dict[str, List], List[float], List[bool], List[Dict]]:
        obs_images, obs_texts, rewards, dones, infos = [], [], [], [], []

        # 健壮性检查：确保 actions 列表长度与设备数匹配
        if len(actions) != self.num_envs:
            print(f"--- 错误: 传入的 actions 数量 ({len(actions)}) 与设备数 ({self.num_envs}) 不匹配! ---")
            # 这是一个严重的逻辑错误，但我们尝试通过广播第一个动作来恢复，而不是崩溃
            safe_action = actions[0] if actions else "wait(1.0)"
            print(f"--- 警告: 将使用动作 '{safe_action}' 广播到所有 {self.num_envs} 个设备。 ---")
            actions = [safe_action] * self.num_envs

        # <<< 修改：使用线程池并行执行 step >>>
        futures: List[Future] = []
        for i, serial in enumerate(self.device_serials):
            futures.append(self.executor.submit(self._step_device, serial, actions[i]))

        # 按顺序收集结果
        for future in futures:
            try:
                img, text, reward, done, info = future.result()
                obs_images.append(img)
                obs_texts.append(text)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            except Exception as e:
                print(f"--- 严重错误: 'step' 线程 'future.result()' 失败: {e} ---")
                # 添加占位符以保持批处理(batch)大小一致
                obs_images.append(np.zeros((256, 256, 3), dtype=np.uint8))
                obs_texts.append("<image>\nERROR: Future result failed.")
                rewards.append(-0.1) # 惩罚
                dones.append(True) # 终止
                infos.append({"device_serial": "unknown", "error": str(e)})

        observations = {"image": obs_images, "text": obs_texts}
        return observations, np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), infos

    def _dispatch_action(self, actuator: Actuator, serial: str, action_str: str, elements: list) -> str:
        # print(f"--- [设备: {serial}] 正在分发动作: '{action_str}' ---") # 训练时可以注释掉减少日志
        try:
            # 1. 优先处理来自上游的格式错误标记
            if action_str.startswith("format_error"):
                return f"FORMAT_ERROR: {action_str.split('reason=')[1][:-1]}"

            # 2. 基础解析：分离 动作名 和 参数
            # 严格假设格式为 "action_name(params)" 或 "action_name"
            if "(" in action_str:
                action_name = action_str.split("(")[0].strip()
                # 提取括号内的内容，不进行复杂的正则匹配，只取两头
                params_str = action_str[len(action_name) + 1 :].strip().rstrip(")")
            else:
                action_name = action_str.strip()
                params_str = ""

            # 3. 特殊动作处理
            if action_name == "finish":
                return "SUCCESS"
            
            # 检查是否需要元素列表
            if action_name in ["tap", "input_text", "drag", "clear_text"] and not elements:
                return "FAILURE_NO_ELEMENTS"

            # 4. 动作分发 (Strict Mode)
            result = False
            
            if action_name == "tap":
                # [RL Strict] 直接转换为 int，如果包含非数字字符(如 tap(id=12))将抛出 ValueError
                uid = int(params_str.strip())
                result = actuator.tap(uid, elements)

            elif action_name == "input_text":
                # [RL Strict] 严格要求用逗号分隔: uid, text
                if "," not in params_str:
                    raise ValueError("input_text requires 2 arguments: uid, text")
                uid_part, text_part = params_str.split(",", 1)
                
                uid = int(uid_part.strip())
                # 只去除首尾的引号，不处理 text= 前缀
                text = text_part.strip().strip("'\"") 
                result = actuator.input_text(uid, text, elements)

            elif action_name == "swipe":
                if "," not in params_str:
                     raise ValueError("swipe requires 2 arguments: direction, magnitude")
                direction_part, magnitude_part = params_str.split(",", 1)
                
                direction = direction_part.strip().strip("'\"")
                magnitude = magnitude_part.strip().strip("'\"")
                result = actuator.swipe(direction, magnitude)

            elif action_name == "drag":
                if "," not in params_str:
                     raise ValueError("drag requires 2 arguments: start_uid, end_uid")
                start_part, end_part = params_str.split(",", 1)
                
                # [RL Strict] 两个参数都必须直接是整数
                result = actuator.drag(int(start_part.strip()), int(end_part.strip()), elements)

            elif action_name == "clear_text":
                uid = int(params_str.strip())
                result = actuator.clear_text(uid, elements)

            elif action_name == "enter":
                result = actuator.enter()

            elif action_name == "back":
                result = actuator.back()

            elif action_name == "home":
                result = actuator.home()

            elif action_name == "wait":
                # 保持对浮点数的支持
                result = actuator.wait(float(params_str.strip()))

            else:
                print(f"--- [设备: {serial}] 警告: 未知动作 '{action_name}' ---")
                return f"UNKNOWN_ACTION: '{action_name}' is not a valid action."

            status = "SUCCESS" if result else "FAILURE"
            print(f"--- [设备: {serial}] 动作 '{action_name}' 执行状态: {status} ---")
            return status

        except ValueError as e:
            # 捕获 int转换失败、解包失败等 Python 基础错误
            # 这会将错误信息反馈给 RL，使其学习到格式错误
            error_message = f"ARGUMENT_ERROR: {str(e)}. Check your action format."
            print(f"--- [设备: {serial}] 参数解析错误: {error_message} (Action: {action_str}) ---")
            return error_message
            
        except Exception as e:
            error_message = f"EXECUTION_ERROR: {repr(e)}"
            print(f"--- [设备: {serial}] 动作 '{action_str}' 执行时出错: {error_message} ---")
            return error_message

    # <<< 新增：添加一个 close 方法来清理线程池 >>>
    def close(self):
        """
        关闭环境并清理资源，特别是线程池。
        """
        print(f"--- 正在关闭 JarvisMultiDeviceEnv (管理 {self.num_envs} 台设备)，清理线程池... ---")
        self.executor.shutdown(wait=True)
        print("--- 线程池已关闭。 ---")

    def __del__(self):
        # 作为一个保障，以防用户忘记调用 .close()
        if hasattr(self, 'executor'):
            # 在 __del__ 中，我们不等待线程完成，只是触发关闭
            self.executor.shutdown(wait=False)
            print(f"--- (via __del__) 触发线程池关闭。 ---")


def build_jarvis_envs(jarvis_config_path: str, max_steps: int) -> JarvisMultiDeviceEnv:
    return JarvisMultiDeviceEnv(jarvis_config_path, max_steps)