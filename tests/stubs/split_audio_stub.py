#!/usr/bin/env python3
import argparse
import os
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    # 兼容 orchestrator 传入的其它参数（忽略）
    p.add_argument("--sample_rate")
    p.add_argument("--frame_duration")
    p.add_argument("--vad_aggressiveness")
    p.add_argument("--min_segment_duration")
    p.add_argument("--merge_threshold")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dst = os.path.join(args.output_dir, os.path.basename(args.input))
    shutil.copy2(args.input, dst)


if __name__ == "__main__":
    main()
