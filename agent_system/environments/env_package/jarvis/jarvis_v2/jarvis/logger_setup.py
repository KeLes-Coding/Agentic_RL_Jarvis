import logging
import os
import sys

# 不再需要 RotatingFileHandler，因为每个run的日志是独立的
from logging import FileHandler


def setup_logging(config: dict, log_file_path: str | None = None):
    """
    根据配置初始化日志系统。
    可以为本次运行指定一个专用的日志文件。

    Args:
        config: 包含 'logging' 键的配置字典。
        log_file_path (str | None): 如果提供，则将文件日志输出到此路径。
    """
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO").upper()
    log_format = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    # 控制台 handler 始终添加
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 如果提供了专用的日志文件路径，则添加 FileHandler
    if log_file_path:
        # 不再需要从config中读取文件配置，直接使用提供的路径
        try:
            # 确保目录存在 (虽然我们会在agent_worker中创建，但这里作为安全保障)
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # 使用标准的 FileHandler，因为日志文件与run生命周期绑定
            file_handler = FileHandler(log_file_path, "a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            # 不再需要全局的 logging.info, logger将在worker中获取

        except Exception as e:
            # 如果日志文件创建失败，至少保证控制台日志能工作
            logging.error(f"创建文件日志处理器失败: {e}", exc_info=True)
