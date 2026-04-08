#!/usr/bin/env python3
"""自动化迭代总控脚本（兼容入口）。

核心逻辑见包 `orchestrator_core`。
"""

from orchestrator_core.config_loader import load_config
from orchestrator_core.step_runner import run_step

__all__ = ["load_config", "run_step"]

if __name__ == "__main__":
    from orchestrator_core.cli import main

    main()
