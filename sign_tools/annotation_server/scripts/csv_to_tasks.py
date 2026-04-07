import csv
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "annotation_tool.config.json"


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "audioRoot": "./data/audio",
        "tasksFile": "./data/tasks.json",
    }


def main():
    """
    从简单的 CSV 待标列表生成 tasks.json。

    默认假设 CSV 结构为（无表头）:
      audio_filepath,text
    例如：
      /path/to/audio.wav,這是一句測試文本

    使用方式（在 annotation_server 目录下）：
      - 使用默认输入文件：
          python scripts/csv_to_tasks.py
        将读取 data/input.csv
      - 指定任意 CSV 路径（绝对或相对）：
          python scripts/csv_to_tasks.py /path/to/your.csv
    """
    config = load_config()

    # 1) 若提供了命令行参数，则直接使用该路径；
    # 2) 否则，退回到默认的 data/input.csv。
    if len(sys.argv) >= 2:
        csv_path = Path(sys.argv[1]).expanduser()
        if not csv_path.is_absolute():
            csv_path = (ROOT_DIR / csv_path).resolve()
    else:
        csv_path = ROOT_DIR / "data" / "input.csv"

    # 输出 tasks.json 路径，遵循 annotation_tool.config.json 中的配置
    tasks_file = ROOT_DIR / config.get("tasksFile", "./data/tasks.json")
    # 记录最近一次使用的源 CSV 路径，便于回填
    source_note = ROOT_DIR / "data" / "source_csv_path.txt"

    if not csv_path.exists():
        raise SystemExit(f"找不到输入文件: {csv_path}")

    tasks = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["audio_filepath", "text"])
        for row in reader:
            audio_fp = (row.get("audio_filepath") or "").strip()
            text = (row.get("text") or "").strip()
            if not audio_fp:
                continue

            audio_name = Path(audio_fp).name
            audio_id = Path(audio_name).stem

            task = {
                "id": audio_id,
                "audioPath": audio_name,
                "originalPath": audio_fp,
                "pseudoLabel": text,
                "confidence": 1.0,
                "batchId": "default",
                "priority": 0,
            }
            tasks.append(task)

    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 写入源 CSV 路径记录
    source_note.parent.mkdir(parents=True, exist_ok=True)
    source_note.write_text(str(csv_path), encoding="utf-8")
    print(f"已写入 {tasks_file}，共 {len(tasks)} 条任务。源 CSV: {csv_path}")


if __name__ == "__main__":
    main()

