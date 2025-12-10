import logging
import os
import sys
from datetime import datetime

def setup_logging(log_dir='logs', log_level='INFO', log_file_prefix=None):
    """初始化日志系统，返回 logger 对象"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if log_file_prefix:
        filename = f"{log_file_prefix}_{timestamp}.log"
    else:
        filename = f"log_{timestamp}.log"
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger()  # root logger
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    # 清除已有处理器
    logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.info(f"日志初始化完成，日志文件: {log_path}")
    return logger
