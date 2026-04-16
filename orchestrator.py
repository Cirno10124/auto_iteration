#!/usr/bin/env python3
"""自动化迭代总控脚本（兼容入口）。

核心逻辑位于 `src/orchestrator_core`。
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from orchestrator_core.config_loader import load_config
from orchestrator_core.step_runner import run_step

__all__ = ["load_config", "run_step"]

if __name__ == "__main__":
    from orchestrator_core.cli import main

    main()
