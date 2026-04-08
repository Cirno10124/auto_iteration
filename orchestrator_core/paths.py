"""项目根目录（与 legacy `orchestrator.py` 同级），用于解析各阶段脚本路径。"""

import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_PKG_DIR)
