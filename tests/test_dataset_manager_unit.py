import csv

import pytest

from scripts.dataset_manager import (  # noqa: E402
    check_data_size,
    read_existing_csv,
    write_csv,
)

pytestmark = pytest.mark.unit


class DummyLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg):
        self.warnings.append(msg)

    def info(self, msg):
        self.infos.append(msg)


def test_read_existing_csv_skips_empty_transcript(tmp_path):
    """读取 CSV 时跳过空 transcript 记录。"""
    csv_path = tmp_path / "train.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_filepath", "text"])
        writer.writerow(["a.wav", "hello"])
        writer.writerow(["b.wav", ""])

    paths, rows = read_existing_csv(str(csv_path), skip_empty_transcript=True)
    assert "a.wav" in paths
    assert "b.wav" not in paths
    assert rows == [("a.wav", "hello")]


def test_write_csv_writes_header_and_rows(tmp_path):
    """写入 CSV 时应包含表头和全部数据行。"""
    out = tmp_path / "out.csv"
    logger = DummyLogger()
    write_csv(str(out), [("x.wav", "text_x"), ("y.wav", "text_y")], logger)

    content = out.read_text(encoding="utf-8").splitlines()
    assert content[0] == "audio_filepath,text"
    assert "x.wav,text_x" in content
    assert "y.wav,text_y" in content


def test_check_data_size_warns_for_small_or_empty():
    """数据量为空或过小时应产生告警。"""
    logger = DummyLogger()
    check_data_size(0, "训练集", logger)
    check_data_size(5, "验证集", logger)
    assert any("为空" in msg for msg in logger.warnings)
    assert any("数据量很少" in msg for msg in logger.warnings)
