import contextvars
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# 供日志 Formatter 读取（检索：speaker= / iter= / step= / model=）
speaker_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "speaker_id", default="-"
)
iteration_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "iteration_id", default="-"
)
step_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "step", default="-"
)
model_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "model", default="-"
)


class OrchestratorContextFilter(logging.Filter):
    """将上下文注入 LogRecord，供统一格式使用。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.speaker_id = speaker_id_var.get()
        record.iteration_id = iteration_id_var.get()
        record.step = step_var.get()
        record.model = model_var.get()
        return True


ORCHESTRATOR_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-7s | speaker=%(speaker_id)s | "
    "iter=%(iteration_id)s | step=%(step)s | model=%(model)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def set_orchestrator_context(
    *,
    speaker_id: Optional[str] = None,
    iteration_id: Optional[str] = None,
    step: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """更新当前协程/线程的日志上下文字段（未传入的项保持不变）。"""
    if speaker_id is not None:
        speaker_id_var.set(speaker_id)
    if iteration_id is not None:
        iteration_id_var.set(iteration_id)
    if step is not None:
        step_var.set(step)
    if model is not None:
        model_var.set(model)


def reset_orchestrator_context() -> None:
    """恢复为默认占位符。"""
    speaker_id_var.set("-")
    iteration_id_var.set("-")
    step_var.set("-")
    model_var.set("-")


def short_model_hint(path_or_id: Optional[str], max_len: int = 72) -> str:
    """日志中 model 字段过长时截断，避免单行爆炸。"""
    if path_or_id is None or path_or_id == "":
        return "-"
    s = str(path_or_id).replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def ensure_early_console_logging() -> None:
    """在读取完整 logging 配置前，为 root 挂载带上下文的控制台 Handler。"""
    root = logging.getLogger()
    if root.handlers:
        return
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter(ORCHESTRATOR_LOG_FORMAT, DATE_FORMAT))
    h.addFilter(OrchestratorContextFilter())
    root.addHandler(h)
    root.setLevel(logging.INFO)


def setup_logging(log_dir="logs", log_level="INFO", log_file_prefix=None):
    """初始化日志系统，返回 root logger（与历史行为一致）。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_file_prefix:
        filename = f"{log_file_prefix}_{timestamp}.log"
    else:
        filename = f"log_{timestamp}.log"
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    ctx_filter = OrchestratorContextFilter()
    fmt = logging.Formatter(ORCHESTRATOR_LOG_FORMAT, DATE_FORMAT)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(ctx_filter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(
        getattr(logging, log_level.upper(), logging.INFO)
    )
    console_handler.setFormatter(fmt)
    console_handler.addFilter(ctx_filter)
    logger.addHandler(console_handler)

    logger.info(f"日志初始化完成，日志文件: {log_path}")
    return logger
