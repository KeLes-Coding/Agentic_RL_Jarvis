import logging
import yaml
import datetime
import time
import os
import json
import io  # 用于处理二进制数据流
import re  # 导入re模块用于字符替换

# 新增：从datetime模块导入timezone和timedelta以处理时区
from datetime import timezone, timedelta

# 导入Pillow库，如果失败则优雅降级
try:
    from PIL import Image
except ImportError:
    Image = None

from .logger_setup import setup_logging
from .info_pool import InfoPoolManager
from .modules.observer import Observer
from .modules.actuator import Actuator
from .llm.client import LLMClient
from .llm import prompts


class JarvisAgent:
    """
    JarvisAgent是系统的核心，负责编排“观察-思考-行动-记录”的完整循环。
    """

    def __init__(
        self,
        config: dict,
        device_serial: str,
        info_pool: InfoPoolManager,
        run_start_time: datetime.datetime,  # 接收带时区的开始时间
    ):
        """
        初始化Agent的所有组件。

        Args:
            config: 从config.yaml加载的配置字典。
            device_serial: 当前Agent负责的设备序列号。
            info_pool: 用于记录轨迹的InfoPoolManager实例。
            run_start_time: 本次任务运行的开始时间 (带时区)。
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.device_serial = device_serial
        self.info_pool = info_pool
        self.run_start_time = run_start_time  # 保存任务开始时间

        # 初始化三大核心模块
        adb_path = config.get("adb", {}).get("executable_path", "adb")
        self.observer = Observer(adb_path, device_serial)
        self.actuator = Actuator(adb_path, device_serial)

        # 将全局的proxy配置传入LLMClient
        self.llm_client = LLMClient(
            config=config.get("llm", {}), proxy_config=config.get("proxy", {})
        )

        agent_config = config.get("agent", {})
        llm_config = config.get("llm", {})

        # 读取当前provider的is_vlm设置
        api_mode = llm_config.get("api_mode", "openai")
        self.is_vlm = (
            llm_config.get("providers", {}).get(api_mode, {}).get("is_vlm", False)
        )
        self.logger.info(f"VLM mode is {'ENABLED' if self.is_vlm else 'DISABLED'}.")

        # 新增: 读取重试配置
        retry_config = agent_config.get("retry_on_error", {})
        self.retry_enabled = retry_config.get("enabled", True)
        self.max_retries = retry_config.get("attempts", 3)

        # 读取压缩配置
        self.compression_config = agent_config.get("image_compression", {})
        if self.compression_config.get("enabled") and not Image:
            self.logger.error(
                "Image compression is enabled, but Pillow is not installed. Please run 'pip install Pillow'. Disabling compression."
            )
            self.compression_config["enabled"] = False

    def _compress_image(self, image_bytes: bytes) -> bytes:
        """
        根据配置对给定的图片二进制数据进行压缩。
        """
        if (
            not self.compression_config.get("enabled", False)
            or not image_bytes
            or not Image
        ):
            return image_bytes

        try:
            scale_factor = self.compression_config.get("scale_factor", 0.5)
            self.logger.info(f"Compressing image with scale factor: {scale_factor}")

            # 将二进制数据读入Pillow Image对象
            img = Image.open(io.BytesIO(image_bytes))

            # 计算新的尺寸
            original_width, original_height = img.size
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # 调整图片大小
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 将压缩后的图片存回二进制流
            buffer = io.BytesIO()
            resized_img.save(buffer, format="PNG")

            compressed_bytes = buffer.getvalue()
            original_size_kb = len(image_bytes) / 1024
            compressed_size_kb = len(compressed_bytes) / 1024
            self.logger.info(
                f"Image compressed: {original_size_kb:.2f} KB -> {compressed_size_kb:.2f} KB"
            )

            return compressed_bytes
        except Exception as e:
            self.logger.error(f"Failed to compress image: {e}", exc_info=True)
            return image_bytes  # 压缩失败则返回原图

    def _dispatch_action(self, action_str: str, elements: list) -> str:
        """
        解析LLM返回的动作字符串，并调用对应的Actuator方法。
        """
        try:
            action_name = action_str.split("(")[0]
            params_str = (
                action_str[len(action_name) + 1 : -1] if "(" in action_str else ""
            )

            if action_name in ["tap", "input_text", "swipe"] and not elements:
                self.logger.error("动作执行失败：UI元素列表为空，无法定位元素。")
                return "FAILURE_NO_ELEMENTS"

            if action_name == "tap":
                uid = int(params_str)
                result = self.actuator.tap(uid, elements)
            elif action_name == "input_text":
                uid, text_to_input = params_str.split(",", 1)
                text_to_input = text_to_input.strip().strip("'\"")
                result = self.actuator.input_text(int(uid), text_to_input, elements)
            elif action_name == "swipe":
                start_uid, end_uid = map(int, params_str.split(","))
                result = self.actuator.swipe(start_uid, end_uid, elements)
            elif action_name == "back":
                result = self.actuator.back()
            elif action_name == "home":
                result = self.actuator.home()
            elif action_name == "wait":
                seconds = float(params_str)
                result = self.actuator.wait(seconds)
            else:
                self.logger.error(f"未知的动作: {action_name}")
                return "UNKNOWN_ACTION"

            return "SUCCESS" if result else "FAILURE"
        except Exception as e:
            self.logger.error(
                f"解析或执行动作 '{action_str}' 时出错: {e}", exc_info=True
            )
            return "EXECUTION_ERROR"

    def run(self, task: str):
        """
        启动Agent的核心控制循环来完成指定任务。
        """
        self.logger.info(f"启动新任务: '{task}' 在设备 {self.device_serial} 上。")

        prev_thought, prev_action, prev_screenshot_bytes = "", "", None

        # 新增: 初始化token计数器
        total_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        max_steps = self.config.get("agent", {}).get("max_steps", 15)
        self.logger.info(f"任务最大执行步数: {max_steps}")

        final_status, final_summary = (
            "MAX_STEPS_REACHED",
            f"Task stopped after reaching {max_steps} steps.",
        )

        for i in range(1, max_steps + 1):
            self.logger.info(f"--- 开始第 {i} 步 ---")
            step_start_time = time.time()

            # --- 重试循环开始 ---
            llm_response, raw_llm_response, step_tokens = None, None, None
            observation_data = None
            last_error = None

            # 根据配置决定实际的尝试次数
            attempts = self.max_retries if self.retry_enabled else 1
            for attempt in range(attempts):
                try:
                    # 1. 观察
                    observation_data = self.observer.get_current_observation()

                    # 2. (可选) 压缩图片
                    current_screenshot_bytes = self._compress_image(
                        observation_data.get("screenshot_bytes")
                    )

                    simplified_ui = observation_data.get("simplified_elements_str")

                    if not current_screenshot_bytes or not simplified_ui:
                        # 这是一个关键性失败，无法重试
                        final_status, final_summary = (
                            "CRITICAL_FAILURE",
                            "Failed to get complete observation.",
                        )
                        self.logger.error(final_summary)
                        # 跳出所有循环
                        i = max_steps + 1
                        break

                    # 3. 思考 (准备并调用LLM)
                    images_for_llm = []
                    if self.is_vlm:
                        if i == 1:
                            images_for_llm.append(current_screenshot_bytes)
                        else:
                            if prev_screenshot_bytes:
                                images_for_llm.append(prev_screenshot_bytes)
                            images_for_llm.append(current_screenshot_bytes)
                    else:
                        self.logger.info(
                            "VLM mode is disabled. Sending text-only prompt."
                        )

                    if (
                        i == 1 and attempt == 0
                    ):  # 只在第一步的第一次尝试时使用初始prompt
                        prompt_text = prompts.get_step_1_prompt(task, simplified_ui)
                    else:
                        prompt_text = prompts.get_intermediate_prompt(
                            task, prev_thought, prev_action, simplified_ui
                        )

                    # 调用LLM
                    llm_response, raw_llm_response, step_tokens = self.llm_client.query(
                        prompt_text, images=images_for_llm
                    )

                    # 验证响应格式
                    thought = llm_response.get("thought")
                    action_str = llm_response.get("action")
                    if thought is None or action_str is None:
                        # 即使API调用成功，如果内容不符合我们的格式要求，也视为错误
                        raise AttributeError(
                            f"LLM response missing 'thought' or 'action' key. Response: {llm_response}"
                        )

                    # 成功获取并验证了响应，跳出重试循环
                    last_error = None
                    break

                except (AttributeError, json.JSONDecodeError, TypeError) as e:
                    last_error = e
                    self.logger.warning(
                        f"步骤 {i} 尝试 {attempt + 1}/{attempts} 失败: {e}. 正在重试..."
                    )
                    time.sleep(2)  # 等待2秒后重试

            # 如果所有重试都失败了
            if last_error:
                final_status, final_summary = (
                    "CRITICAL_FAILURE",
                    f"Failed after {attempts} attempts. Last error: {last_error}",
                )
                self.logger.error(final_summary)
                break  # 终止主循环

            if i > max_steps:  # 检查是否是因为观察失败而跳出
                break

            # --- 重试循环结束 ---

            # 累加token
            if step_tokens:
                total_token_usage["prompt_tokens"] += step_tokens.get(
                    "prompt_tokens", 0
                )
                total_token_usage["completion_tokens"] += step_tokens.get(
                    "completion_tokens", 0
                )
                total_token_usage["total_tokens"] += step_tokens.get("total_tokens", 0)

            thought = llm_response.get("thought", "LLM did not provide a thought.")
            action_str = llm_response.get(
                "action", "error(reason='No action returned')"
            )
            elements_list = observation_data.get("simplified_elements_list")

            # 4. 行动
            self.logger.info(f"LLM Thought: {thought}")
            self.logger.info(f"LLM Action: {action_str}")

            if action_str.startswith("finish"):
                final_summary = (
                    action_str.replace("finish(", "")[:-1].strip().strip("'\"")
                )
                final_status = "SUCCESS"
                self.logger.info(f"任务完成！LLM总结: {final_summary}")
                execution_status = "TASK_COMPLETED"
            else:
                execution_status = self._dispatch_action(action_str, elements_list)

            # 5. 记录
            step_log = {
                "step_id": i,
                # --- 修正点: 使用与任务开始时相同的时区记录步骤时间 ---
                "timestamp": datetime.datetime.now(
                    self.run_start_time.tzinfo
                ).isoformat(),
                "overall_task": task,
                "observation": {
                    "simplified_elements_str": simplified_ui,
                },
                "llm_prompt": prompt_text,  # 记录完整的prompt
                "raw_llm_response": raw_llm_response,  # 记录原始回复
                "llm_response": llm_response,  # 记录解析后的回复
                "execution": {
                    "validated_action": action_str,
                    "status": execution_status,
                },
                "cycle_duration_ms": int((time.time() - step_start_time) * 1000),
                "screenshot_bytes": current_screenshot_bytes,
                "xml_content": observation_data.get("xml_content"),
            }
            self.info_pool.record_step(step_log)

            if execution_status == "TASK_COMPLETED":
                break

            # 6. 为下一次循环更新状态
            prev_thought, prev_action, prev_screenshot_bytes = (
                thought,
                action_str,
                current_screenshot_bytes,
            )
            time.sleep(1)

        self.info_pool.finalize_run(
            status=final_status,
            summary=final_summary,
            run_start_time=self.run_start_time,
            task=task,
            token_usage=total_token_usage,  # 传递token信息
        )


def agent_worker(device_serial: str, task: str):
    """
    Agent的工作函数，由主进程 (agent_manager.py) 为每个设备调用。
    负责为一次任务运行设置好所有环境。
    """
    # 定义北京时区 (UTC+8)
    beijing_tz = timezone(timedelta(hours=8))
    # 使用北京时区获取当前时间作为任务开始时间
    run_start_time = datetime.datetime.now(beijing_tz)

    # 1. 生成本次运行的唯一目录
    safe_task_name = "".join(c for c in task if c.isalnum() or c in " _-").rstrip()[:50]
    timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")  # 使用带时区的时间

    # --- 关键修改：替换device_serial中的非法字符 ---
    safe_device_serial = re.sub(r"[:\\/]", "_", device_serial)

    run_path = os.path.join(
        "jarvis", "runs", f"{timestamp}_{safe_task_name}_{safe_device_serial}"
    )
    os.makedirs(run_path, exist_ok=True)

    # 2. 加载主配置
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("错误: config.yaml 未找到！程序将在没有配置的情况下退出。")
        return
    except Exception as e:
        print(f"加载config.yaml时出错: {e}")
        return

    # 3. 初始化日志系统
    log_file_path = os.path.join(run_path, "agent_run.log")
    setup_logging(config, log_file_path=log_file_path)

    logger = logging.getLogger(f"Worker-{device_serial}")
    logger.info(f"Agent Worker已为设备 {device_serial} 启动。")
    logger.info(f"所有日志和产出将保存在: {run_path}")

    info_pool = None
    try:
        # 4. 初始化信息池
        info_pool = InfoPoolManager(run_directory=run_path)

        # 5. 实例化并运行Agent
        agent = JarvisAgent(
            config=config,
            device_serial=device_serial,
            info_pool=info_pool,
            run_start_time=run_start_time,  # 传递带时区的开始时间
        )
        agent.run(task=task)

    except Exception as e:
        # 捕获初始化阶段（如LLMClient认证失败）的严重错误
        logger.critical(f"Agent Worker 初始化或运行时遇到致命错误: {e}", exc_info=True)
        # 尝试在出错时也能finalize run
        if info_pool:
            info_pool.finalize_run(
                status="CRITICAL_FAILURE",
                summary=f"Agent因异常中断: {e}",
                run_start_time=run_start_time,
                task=task,
            )

    logger.info(f"Agent Worker 已完成在设备 {device_serial} 上的任务。")
