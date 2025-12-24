#!/usr/bin/env python3
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # 生成最小可用的 CSV（真实训练不会跑到）
    header = "audio,text\n"
    row = "dummy.wav,dummy\n"
    for name in ["train.csv", "val.csv", "test.csv"]:
        with open(
            os.path.join(args.output_dir, name), "w", encoding="utf-8"
        ) as f:
            f.write(header)
            f.write(row)


if __name__ == "__main__":
    main()
