#!/usr/bin/env python3
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--test_manifest")
    p.add_argument("--output_file", required=True)
    p.add_argument("--metric", default="cer")
    p.add_argument("--language")
    p.add_argument("--task")
    p.add_argument("--base_model_path")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    key = (args.metric or "cer").upper()

    # 为了让 orchestrator 走到 “跳过转换并进入下一轮” 分支：
    # tuned_metric <= baseline_metric
    # - 当评估 model_dir=tests/model 时写 0.10
    # - 当评估 model_dir=tests/model/best_model (或其它 baseline) 时写 0.20
    val = 0.10
    norm = args.model_dir.replace("\\", "/")
    if norm.endswith("/best_model"):
        val = 0.20

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(f"{key}: {val}\n")


if __name__ == "__main__":
    main()


