#!/usr/bin/env python3
"""校验源码目录约束。

规则：
1) 根目录仅允许白名单 Python 文件（入口/测试装配）。
2) 根目录 scripts/ 仅允许 shell/ps1 运维脚本，不允许 Python 文件。
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROOT_PY_ALLOWLIST = {"orchestrator.py", "conftest.py"}
SCRIPTS_DIR = ROOT / "scripts"


def collect_root_python_violations() -> list[str]:
    violations: list[str] = []
    for item in ROOT.glob("*.py"):
        if item.name not in ROOT_PY_ALLOWLIST:
            violations.append(str(item.relative_to(ROOT)))
    return violations


def collect_scripts_python_violations() -> list[str]:
    if not SCRIPTS_DIR.exists():
        return []
    return [
        str(path.relative_to(ROOT))
        for path in SCRIPTS_DIR.rglob("*.py")
        if path.is_file()
    ]


def main() -> int:
    root_violations = collect_root_python_violations()
    scripts_violations = collect_scripts_python_violations()

    if not root_violations and not scripts_violations:
        print("Source layout check passed.")
        return 0

    print("Source layout check failed.")
    if root_violations:
        print("\n[Root Python violations]")
        for file in root_violations:
            print(f"- {file}")
    if scripts_violations:
        print("\n[scripts/ Python violations]")
        for file in scripts_violations:
            print(f"- {file}")

    print("\nExpected layout:")
    print("- Python source files should be placed under src/")
    print("- Root only allows: orchestrator.py, conftest.py")
    print("- scripts/ only allows shell/ps1 ops scripts")
    return 1


if __name__ == "__main__":
    sys.exit(main())
