"""子进程步骤执行。"""

import subprocess
import traceback
from typing import Any, List, Optional

from logging_utils import set_orchestrator_context, short_model_hint


def run_step(
    name: str,
    cmd: List[str],
    logger: Any,
    capture_output: bool = True,
    model: Optional[str] = None,
) -> bool:
    """执行步骤，带错误处理和日志记录。

    capture_output: 为 True 时捕获子进程输出，失败时写入日志；为 False 时
    子进程 stdout/stderr 直接输出到当前控制台（用于训练等需实时查看 loss 的步骤）。

    model: 写入日志上下文的 model 字段（如模型路径或 HF id），便于检索。
    """
    m = model if model not in (None, "") else "-"
    set_orchestrator_context(
        step=name,
        model=short_model_hint(m) if m != "-" else "-",
    )
    logger.info(f"=== 开始步骤: {name} ===")
    logger.info(f"命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=capture_output,
            text=True if capture_output else None,
        )
        if capture_output and result.stdout:
            logger.debug(f"步骤 {name} 输出:\n{result.stdout}")
        logger.info(f"=== 步骤 {name} 完成 ===\n")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"步骤失败 | step={name} | exit_code={e.returncode} | "
            f"cmd={' '.join(cmd)}"
        )
        logger.error(error_msg)
        if capture_output:
            if e.stdout:
                logger.error(f"标准输出:\n{e.stdout}")
            if e.stderr:
                logger.error(f"标准错误:\n{e.stderr}")
        logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
        raise
    except Exception as e:
        error_msg = (
            f"步骤执行异常 | step={name} | type={type(e).__name__} | "
            f"cmd={' '.join(cmd)} | err={e}"
        )
        logger.error(error_msg)
        logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
        raise
