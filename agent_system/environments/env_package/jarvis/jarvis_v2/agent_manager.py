# agent_manager.py (Refactored V8 - Schedulers Encapsulated)

import multiprocessing
import time
import os
import yaml
import logging
import platform
import sys
import atexit
from typing import List, Dict, Any

# 动态添加`device_management`到系统路径，使其可被导入
sys.path.append(os.path.join(os.path.dirname(__file__)))

# 导入现有的 agent_worker
from jarvis.agent import agent_worker

# 从新的、独立的模块中导入设备提供者
from device_management.device_providers import (
    DeviceProvider,
    LocalDeviceProvider,
    RemoteIPDeviceProvider,
    SSHReverseTunnelDeviceProvider,
    SSHForwardTunnelDeviceProvider,
    get_ssh_tunnel_processes,
)


# --- 全局基础日志配置 ---
# 为管理器设置日志格式，以便与Agent的日志区分开
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [Manager] - %(levelname)s - %(message)s",
    force=True,  # 强制覆盖任何现有配置
)


# --- 程序退出时自动调用的清理函数 ---
def cleanup_ssh_tunnels():
    """
    程序退出时自动调用的清理函数，用于终止所有后台SSH隧道进程。
    """
    # --- 修改: 从解耦的模块中获取进程列表 ---
    ssh_processes = get_ssh_tunnel_processes()
    if not ssh_processes:
        return

    logging.info("正在清理后台SSH隧道进程...")
    for process in ssh_processes:
        if process.poll() is None:  # 进程仍在运行
            try:
                process.terminate()
                process.wait(timeout=5)
                logging.info(f"已终止SSH隧道进程 (PID: {process.pid})")
            except Exception as e:
                logging.error(f"清理SSH隧道进程 (PID: {process.pid}) 时出错: {e}")
    logging.info("SSH隧道清理完毕。")


# --- 注册清理函数 ---
atexit.register(cleanup_ssh_tunnels)


def discover_devices(config: Dict[str, Any]) -> List[str]:
    """
    使用解耦的设备提供者来发现所有可用设备。
    """
    adb_path = config.get("adb", {}).get("executable_path", "adb")
    provider_configs = config.get("main", {}).get("device_providers", {})
    all_devices = []

    # 定义并实例化所有提供者
    providers: List[DeviceProvider] = [
        LocalDeviceProvider(provider_configs.get("local", {}), adb_path),
        RemoteIPDeviceProvider(provider_configs.get("remote_ip", {}), adb_path),
        SSHReverseTunnelDeviceProvider(
            provider_configs.get("ssh_reverse_tunnel", {}), adb_path
        ),
        SSHForwardTunnelDeviceProvider(
            provider_configs.get("ssh_forward_tunnel", {}), adb_path
        ),
    ]

    # 遍历所有提供者并收集设备
    for provider in providers:
        try:
            found_devices = provider.find_devices()
            if found_devices:
                all_devices.extend(found_devices)
        except Exception as e:
            logging.error(
                f"设备提供者 {provider.__class__.__name__} 发生错误: {e}", exc_info=True
            )

    unique_devices = sorted(list(set(all_devices)))
    logging.info(f"发现 {len(unique_devices)} 台唯一可用设备: {unique_devices}")
    return unique_devices


# --- 原有的 Agent 进程包装器 (为 polling 调度器保留) ---
def polling_agent_process_wrapper(
    device_serial: str, task: str, device_status_dict: Dict[str, str]
):
    """
    这是 Agent 工作进程的包装器。
    它的核心职责是：在调用真正的 agent_worker 前后，正确地更新共享的设备状态。
    """
    try:
        # 1. 在任务开始前，立刻将设备状态标记为 'busy'
        device_status_dict[device_serial] = "busy"
        logging.info(f"设备 [{device_serial}] 已锁定，开始执行任务: '{task[:50]}...'")

        # 2. 调用真正的 Agent 工作函数
        # 注意：agent_worker内部已经有完整的日志和异常处理
        agent_worker(device_serial, task)

    except Exception as e:
        # 这是一个兜底的异常捕获，以防 agent_worker 本身在初始化阶段就崩溃
        logging.error(
            f"设备 [{device_serial}] 的 Agent 包装器捕获到致命错误: {e}", exc_info=True
        )
    finally:
        # 3. 无论任务成功、失败还是崩溃，最后都必须将设备状态改回 'idle'
        device_status_dict[device_serial] = "idle"
        logging.info(f"设备 [{device_serial}] 已释放，任务 '{task[:50]}...' 结束。")


# --- 新增：为 Worker Pool 模型设计的常驻工人进程 ---
def worker_pool_agent_process_wrapper(
    device_serial: str, task_queue: multiprocessing.JoinableQueue
):
    """
    这是一个长期运行的"工人"进程，为工作池模型服务。
    它会不断地从任务队列中获取任务并执行，直到收到一个'None'信号。
    """
    logging.info(f"工人进程已为设备 [{device_serial}] 启动，等待任务...")
    while True:
        try:
            task = task_queue.get()
            if task is None:
                logging.info(f"设备 [{device_serial}] 收到结束信号，工人进程退出。")
                task_queue.task_done()
                break

            logging.info(f"设备 [{device_serial}] 领取到新任务: '{task[:50]}...'")
            agent_worker(device_serial, task)
            logging.info(
                f"设备 [{device_serial}] 已完成任务: '{task[:50]}...'，现在空闲。"
            )

        except Exception as e:
            logging.error(
                f"设备 [{device_serial}] 的工人进程捕获到致命错误: {e}", exc_info=True
            )
        finally:
            if "task" in locals() and task is not None:
                task_queue.task_done()


# --- 调度器实现 1: 原始的轮询调度器 ---
def run_polling_scheduler(
    config: Dict[str, Any], tasks: List[str], all_devices: List[str]
):
    """
    采用基于共享状态轮询的调度策略来管理设备和任务。
    """
    logging.info("-" * 40)
    logging.info("任务调度器启动 (轮询模型)")
    logging.info(f"共有 {len(tasks)} 个任务待执行。")
    logging.info("-" * 40)

    manager = multiprocessing.Manager()
    task_queue = manager.Queue()
    for task in tasks:
        task_queue.put(task)

    device_status = manager.dict({device: "idle" for device in all_devices})
    active_processes = {}

    while not task_queue.empty() or any(
        p.is_alive() for p in active_processes.values()
    ):
        finished_devices = [
            device
            for device, process in active_processes.items()
            if not process.is_alive()
        ]
        for device in finished_devices:
            active_processes[device].join()
            del active_processes[device]

        if not task_queue.empty():
            for device_serial in all_devices:
                if (
                    device_status.get(device_serial) == "idle"
                    and device_serial not in active_processes
                ):
                    if task_queue.empty():
                        break
                    task_to_run = task_queue.get()
                    logging.info(
                        f"调度新任务: '{task_to_run[:50]}...' -> 分配给空闲设备 [{device_serial}]"
                    )
                    process = multiprocessing.Process(
                        target=polling_agent_process_wrapper,
                        args=(device_serial, task_to_run, device_status),
                    )
                    process.start()
                    active_processes[device_serial] = process

        time.sleep(2)


# --- 调度器实现 2: 高效的工作池调度器 ---
def run_worker_pool_scheduler(
    config: Dict[str, Any], tasks: List[str], all_devices: List[str]
):
    """
    采用"生产者-消费者"工作池模型来高效地调度任务。
    """
    logging.info("-" * 40)
    logging.info("任务调度器启动 (工作池模型)")
    logging.info(f"将为 {len(all_devices)} 台设备分配 {len(tasks)} 个任务。")
    logging.info("-" * 40)

    task_queue = multiprocessing.JoinableQueue()

    worker_processes = []
    for device_serial in all_devices:
        process = multiprocessing.Process(
            target=worker_pool_agent_process_wrapper,
            args=(device_serial, task_queue),
            daemon=True,
        )
        process.start()
        worker_processes.append(process)

    for task in tasks:
        task_queue.put(task)
    logging.info(f"所有 {len(tasks)} 个任务已放入队列。")

    task_queue.join()
    logging.info("队列中的所有任务均已被工人领取和处理。")

    for _ in worker_processes:
        task_queue.put(None)
    logging.info("已向所有工人进程发送结束信号。")

    for p in worker_processes:
        p.join(timeout=10)


def main():
    """
    主函数，负责加载配置、发现设备，并选择一个调度策略来执行任务。
    """
    # 1. 加载配置
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(
            "错误: config.yaml 未找到！请从 config.template.yaml 复制并修改。"
        )
        return

    main_config = config.get("main", {})
    tasks = main_config.get("tasks", [])

    if not tasks:
        logging.warning("任务列表为空，程序将不执行任何操作。")
        return

    all_devices = discover_devices(config)
    if not all_devices:
        logging.error("未发现任何可用的安卓设备，程序退出。")
        return

    try:
        # --- 在这里选择你想要的调度方式 ---
        # run_polling_scheduler(config, tasks, all_devices)
        run_worker_pool_scheduler(config, tasks, all_devices)

    except KeyboardInterrupt:
        logging.info("捕获到用户中断信号 (Ctrl+C)，将开始清理...")

    finally:
        # atexit 会在这里自动处理SSH隧道的清理
        logging.info("-" * 40)
        logging.info("所有任务均已执行完毕或程序已中断。调度器正在退出。")


if __name__ == "__main__":
    # 在非Linux系统上，'spawn' 启动方法更稳定
    if platform.system() != "Linux":
        multiprocessing.set_start_method("spawn", force=True)
    main()
