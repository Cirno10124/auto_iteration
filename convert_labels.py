#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="音频文件目录"
    )
    parser.add_argument(
        "--labels_dir", type=str, required=True, help="标注文本目录"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="CSV 输出目录"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="训练集比例"
    )
    parser.add_argument(
        "--valid_ratio", type=float, default=0.1, help="验证集比例"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="测试集比例"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # 收集标签文件并映射到对应的音频文件
    label_files = glob.glob(os.path.join(args.labels_dir, "*", "*.txt"))
    pairs = []
    for txt_file in label_files:
        parent = os.path.basename(os.path.dirname(txt_file))
        base = os.path.splitext(os.path.basename(txt_file))[0]
        wav_path = os.path.join(args.audio_dir, parent, base + ".wav")
        if not os.path.exists(wav_path):
            continue
        pairs.append((wav_path, txt_file))
    random.shuffle(pairs)
    total = len(pairs)
    n_train = int(total * args.train_ratio)
    n_valid = int(total * args.valid_ratio)
    splits = {
        "train": pairs[:n_train],
        "validation": pairs[n_train : n_train + n_valid],
        "test": pairs[n_train + n_valid :],
    }

    # 生成 CSV 文件
    for split, files in splits.items():
        csv_path = os.path.join(args.output_dir, f"{split}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["audio_filepath", "text"])
            # 遍历映射对，处理文本并添加标点和 EOS
            for wav_path, txt_path in files:
                with open(txt_path, "r", encoding="utf-8") as lf:
                    text = lf.read().strip()
                # 若末尾不是标点符号，则添加中文句号
                if text and text[-1] not in "。！？；，,?!;":
                    text += "。"
                # 训练集文本后添加 Whisper EOS
                if split == "train":
                    text += "<|endoftext|>"
                writer.writerow([wav_path, text])
        print(f"已生成 {split} 集 CSV: {csv_path}")


if __name__ == "__main__":
    main()
