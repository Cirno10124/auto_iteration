#!/usr/bin/env python3
"""
构建 THCHS30 固定分集 manifests：
- train.csv: 使用“自动标注目录”中的标签（用于无标注训练集演示）
- val.csv/test.csv: 使用 THCHS30 官方 dev/test 标签（保持原始分集）

默认适配目录：
  thchs30/data_thchs30/{train,dev,test,data}

示例：
  python scripts/build_thchs30_manifests.py \
    --thchs30_root thchs30/data_thchs30 \
    --train_labels_dir exp/thchs30_demo/labels/train \
    --output_dir exp/thchs30_demo/manifests
"""

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def list_audio_files(audio_dir: Path) -> List[Path]:
    exts = ("*.wav", "*.flac", "*.mp3")
    files: List[Path] = []
    for ext in exts:
        files.extend(audio_dir.rglob(ext))
    return sorted(files)


def read_text_file(path: Path) -> List[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return [ln.strip() for ln in content.splitlines() if ln.strip()]


def resolve_thchs_transcript(
    trn_path: Path, thchs_data_dir: Path
) -> Optional[str]:
    """
    读取 THCHS30 的转写文本，兼容两种情况：
    1) 真实转写文件（第一行即中文）
    2) 软链接退化文件（第一行是 ../data/xxx.wav.trn）
    """
    lines = read_text_file(trn_path)
    if not lines:
        return None

    first = lines[0]
    if first.startswith("../data/"):
        target = (trn_path.parent / first).resolve()
        if not target.exists():
            target = thchs_data_dir / Path(first).name
        target_lines = read_text_file(target)
        if not target_lines:
            return None
        return normalize_text(target_lines[0])

    return normalize_text(first)


def build_train_rows(
    train_audio_dir: Path, train_labels_dir: Path, strict: bool, logger: logging.Logger
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    missing = 0
    empty = 0
    audios = list_audio_files(train_audio_dir)
    logger.info("扫描 train 音频: %s 条", len(audios))

    for audio_path in audios:
        rel = audio_path.relative_to(train_audio_dir)
        txt_path = (train_labels_dir / rel).with_suffix(".txt")
        if not txt_path.exists():
            missing += 1
            if strict:
                raise FileNotFoundError(f"缺少 train 标签: {txt_path}")
            continue
        text_lines = read_text_file(txt_path)
        if not text_lines:
            empty += 1
            if strict:
                raise ValueError(f"train 标签为空: {txt_path}")
            continue
        text = normalize_text(text_lines[0])
        if not text:
            empty += 1
            continue
        rows.append((str(audio_path.resolve()), text))

    logger.info(
        "train 清单统计: valid=%s, missing_label=%s, empty_label=%s",
        len(rows),
        missing,
        empty,
    )
    return rows


def build_eval_rows(
    split_name: str,
    split_audio_dir: Path,
    thchs_data_dir: Path,
    strict: bool,
    logger: logging.Logger,
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    missing = 0
    audios = list_audio_files(split_audio_dir)
    logger.info("扫描 %s 音频: %s 条", split_name, len(audios))

    for audio_path in audios:
        trn_path = Path(str(audio_path) + ".trn")
        if not trn_path.exists():
            missing += 1
            if strict:
                raise FileNotFoundError(f"缺少 {split_name} 标签文件: {trn_path}")
            continue
        text = resolve_thchs_transcript(trn_path, thchs_data_dir)
        if not text:
            missing += 1
            if strict:
                raise ValueError(f"{split_name} 标签不可读: {trn_path}")
            continue
        rows.append((str(audio_path.resolve()), text))

    logger.info(
        "%s 清单统计: valid=%s, missing_or_invalid_label=%s",
        split_name,
        len(rows),
        missing,
    )
    return rows


def write_csv(csv_path: Path, rows: List[Tuple[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_filepath", "text"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 THCHS30 固定分集 manifests（train 自动标签，dev/test 官方标签）"
    )
    parser.add_argument(
        "--thchs30_root",
        type=Path,
        default=Path("thchs30/data_thchs30"),
        help="THCHS30 根目录，默认 thchs30/data_thchs30",
    )
    parser.add_argument(
        "--train_labels_dir",
        type=Path,
        required=True,
        help="训练集自动标注目录（通常来自 labeler 输出）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="输出 manifest 目录",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="遇到缺失标签或空标签即报错退出",
    )
    args = parser.parse_args()

    logger = setup_logging()
    thchs_root = args.thchs30_root
    train_audio_dir = thchs_root / "train"
    dev_audio_dir = thchs_root / "dev"
    test_audio_dir = thchs_root / "test"
    thchs_data_dir = thchs_root / "data"

    required_dirs = [
        train_audio_dir,
        dev_audio_dir,
        test_audio_dir,
        thchs_data_dir,
        args.train_labels_dir,
    ]
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"目录不存在: {d}")

    train_rows = build_train_rows(
        train_audio_dir, args.train_labels_dir, args.strict, logger
    )
    val_rows = build_eval_rows(
        "dev", dev_audio_dir, thchs_data_dir, args.strict, logger
    )
    test_rows = build_eval_rows(
        "test", test_audio_dir, thchs_data_dir, args.strict, logger
    )

    write_csv(args.output_dir / "train.csv", train_rows)
    write_csv(args.output_dir / "val.csv", val_rows)
    write_csv(args.output_dir / "test.csv", test_rows)

    logger.info("manifest 输出完成: %s", args.output_dir.resolve())
    logger.info(
        "最终条数: train=%s, val=%s, test=%s",
        len(train_rows),
        len(val_rows),
        len(test_rows),
    )


if __name__ == "__main__":
    main()
