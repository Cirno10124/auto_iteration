import os
import sys

import pytest

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _get_case_description(item: pytest.Item) -> str:
    """提取测试用例中文说明，优先使用函数文档首行。"""
    doc = getattr(item.function, "__doc__", None) if hasattr(item, "function") else None
    if doc:
        first_line = doc.strip().splitlines()[0].strip()
        if first_line:
            return first_line
    return f"未提供说明（{item.name}）"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """在收集阶段缓存用例说明，供终端汇总展示。"""
    case_desc = {}
    for item in items:
        case_desc[item.nodeid] = _get_case_description(item)
    setattr(config, "_case_desc_map", case_desc)


def pytest_terminal_summary(
    terminalreporter: "pytest.TerminalReporter", exitstatus: int, config: pytest.Config
) -> None:
    """输出中文用例摘要，便于测试人员快速查看覆盖内容。"""
    case_desc_map = getattr(config, "_case_desc_map", {})
    if not case_desc_map:
        return

    terminalreporter.write_sep("=", "测试用例摘要")

    for report in terminalreporter.stats.get("passed", []):
        desc = case_desc_map.get(report.nodeid, "未提供说明")
        terminalreporter.write_line(f"[通过] {report.nodeid} -> {desc}")

    for report in terminalreporter.stats.get("failed", []):
        desc = case_desc_map.get(report.nodeid, "未提供说明")
        terminalreporter.write_line(f"[失败] {report.nodeid} -> {desc}")

    for report in terminalreporter.stats.get("skipped", []):
        desc = case_desc_map.get(report.nodeid, "未提供说明")
        terminalreporter.write_line(f"[跳过] {report.nodeid} -> {desc}")
