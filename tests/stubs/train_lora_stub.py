#!/usr/bin/env python3
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir", required=True)
    # 兼容其它训练参数（忽略）
    p.add_argument("--train_manifest")
    p.add_argument("--eval_manifest")
    p.add_argument("--test_manifest")
    p.add_argument("--num_train_epochs")
    p.add_argument("--train_batch_size")
    p.add_argument("--eval_batch_size")
    p.add_argument("--mixed_precision")
    p.add_argument("--gradient_accumulation_steps")
    p.add_argument("--language")
    p.add_argument("--task")
    p.add_argument("--max_new_tokens")
    p.add_argument("--no_repeat_ngram_size")
    p.add_argument("--length_penalty")
    p.add_argument("--eval_metric")
    p.add_argument("--repetition_penalty")
    p.add_argument("--learning_rate")
    p.add_argument("--warmup_steps")
    p.add_argument("--weight_decay")
    p.add_argument("--lora_r")
    p.add_argument("--lora_alpha")
    p.add_argument("--lora_dropout")
    p.add_argument("--target_modules")
    p.add_argument("--checkpoint_steps")
    p.add_argument("--checkpoint_epochs")
    p.add_argument("--early_stopping_patience")
    p.add_argument("--early_stopping_threshold")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--save_merged_model", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    calls_path = os.path.join(args.output_dir, "training_calls.txt")
    with open(calls_path, "a", encoding="utf-8") as f:
        f.write(args.model_name_or_path + "\n")

    # 模拟训练产出：创建 best_model 目录，供下一轮 prev_model 选择
    os.makedirs(os.path.join(args.output_dir, "best_model"), exist_ok=True)


if __name__ == "__main__":
    main()


