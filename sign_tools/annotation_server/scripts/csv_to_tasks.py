#!/usr/bin/env python3
"""从训练清单 CSV（audio_filepath,text）生成标注工具所需的 tasks.json。"""

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "annotation_tool.config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "audioRoot": "./data/audio",
        "tasksFile": "./data/tasks.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 CSV 生成 tasks.json（供 annotation_server 使用）"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="输入 CSV 路径；省略则使用 <annotation_server>/data/input.csv",
    )
    parser.add_argument(
        "--batch-id",
        default="default",
        help="写入每条任务的 batchId（默认 default）",
    )
    args = parser.parse_args()

    config = load_config()

    if args.csv_path:
        csv_path = Path(args.csv_path).expanduser()
        if not csv_path.is_absolute():
            csv_path = (ROOT_DIR / csv_path).resolve()
    else:
        csv_path = ROOT_DIR / "data" / "input.csv"

    tasks_rel = config.get("tasksFile", "./data/tasks.json")
    tasks_file = (ROOT_DIR / tasks_rel).resolve() if not Path(tasks_rel).is_absolute() else Path(tasks_rel)
    source_note = ROOT_DIR / "data" / "source_csv_path.txt"

    if not csv_path.is_file():
        raise SystemExit(f"找不到输入文件: {csv_path}")

    tasks = []
    seen_ids: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("CSV 无表头或为空，无法解析。")
        fields = {h.strip() for h in reader.fieldnames if h}
        if "audio_filepath" not in fields or "text" not in fields:
            raise SystemExit(
                "CSV 表头需包含 audio_filepath 与 text 两列（与 manifests/*.csv 一致）。"
            )
        for row in reader:
            audio_fp = (row.get("audio_filepath") or "").strip()
            text = (row.get("text") or "").strip()
            if not audio_fp:
                continue

            stem = Path(audio_fp).stem
            n = seen_ids.get(stem, 0) + 1
            seen_ids[stem] = n
            task_id = stem if n == 1 else f"{stem}__{n}"

            audio_name = Path(audio_fp).name
            tasks.append(
                {
                    "id": task_id,
                    "audioPath": audio_name,
                    "originalPath": audio_fp,
                    "pseudoLabel": text,
                    "confidence": 1.0,
                    "batchId": args.batch_id,
                    "priority": 0,
                }
            )

    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_note.parent.mkdir(parents=True, exist_ok=True)
    source_note.write_text(str(csv_path.resolve()), encoding="utf-8")
    print(
        f"已写入 {tasks_file}，共 {len(tasks)} 条任务。源 CSV: {csv_path.resolve()}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
