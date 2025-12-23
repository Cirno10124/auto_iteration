#!/usr/bin/env python3
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--model_name_or_path", required=True)
    # 兼容 orchestrator 传入的其它参数（忽略）
    p.add_argument("--device")
    p.add_argument("--compression_ratio_threshold")
    p.add_argument("--logprob_threshold")
    p.add_argument("--max_samples")
    p.add_argument("--temperature")
    args = p.parse_args()

    os.makedirs(args.labels_dir, exist_ok=True)
    calls_path = os.path.join(args.labels_dir, "_labeler_calls.txt")
    with open(calls_path, "a", encoding="utf-8") as f:
        f.write(args.model_name_or_path + "\n")

    for root, _, files in os.walk(args.audio_dir):
        rel = os.path.relpath(root, args.audio_dir)
        out_dir = (
            args.labels_dir
            if rel == "."
            else os.path.join(args.labels_dir, rel)
        )
        os.makedirs(out_dir, exist_ok=True)
        for fn in files:
            if not fn.lower().endswith((".wav", ".flac", ".mp3")):
                continue
            base = os.path.splitext(fn)[0]
            with open(os.path.join(out_dir, base + ".txt"), "w", encoding="utf-8") as out:
                out.write("dummy\n")


if __name__ == "__main__":
    main()


