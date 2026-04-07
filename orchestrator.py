#!/usr/bin/env python3
import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

from logging_utils import setup_logging  # 公共日志模块

# 全局日志对象
logger = None


def load_config(config_path, logger=None):
    """加载配置文件，支持 JSONC 格式（带注释的 JSON）"""

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 移除 JSONC 注释（单行注释 // 和块注释 /* */）
        # 移除单行注释（// 开头的行，但不在字符串内）
        lines = content.split("\n")
        cleaned_lines = []
        in_string = False
        escape_next = False

        for line in lines:
            cleaned_line = ""
            i = 0
            while i < len(line):
                char = line[i]

                if escape_next:
                    cleaned_line += char
                    escape_next = False
                    i += 1
                    continue

                if char == "\\":
                    escape_next = True
                    cleaned_line += char
                    i += 1
                    continue

                if char == '"':
                    in_string = not in_string
                    cleaned_line += char
                    i += 1
                    continue

                if not in_string:
                    # 检查单行注释
                    if i < len(line) - 1 and line[i : i + 2] == "//":
                        break  # 跳过该行剩余部分
                    # 检查块注释开始
                    if i < len(line) - 1 and line[i : i + 2] == "/*":
                        # 查找块注释结束
                        j = line.find("*/", i + 2)
                        if j != -1:
                            i = j + 2
                            continue
                        else:
                            # 跨行块注释，需要特殊处理
                            i += 2
                            continue
                    # 非注释且不在字符串中，正常保留字符并前进
                    cleaned_line += char
                    i += 1
                else:
                    cleaned_line += char
                    i += 1

            cleaned_lines.append(cleaned_line)

        cleaned_content = "\n".join(cleaned_lines)

        # 移除块注释（可能跨行）
        # 使用正则表达式移除块注释，但要小心字符串内的内容
        def remove_block_comments(text):
            result = []
            i = 0
            in_string = False
            escape_next = False

            while i < len(text):
                char = text[i]

                if escape_next:
                    result.append(char)
                    escape_next = False
                    i += 1
                    continue

                if char == "\\":
                    escape_next = True
                    result.append(char)
                    i += 1
                    continue

                if char == '"':
                    in_string = not in_string
                    result.append(char)
                    i += 1
                    continue

                if (
                    not in_string
                    and i < len(text) - 1
                    and text[i : i + 2] == "/*"
                ):
                    # 找到块注释开始，查找结束
                    j = text.find("*/", i + 2)
                    if j != -1:
                        i = j + 2
                        continue
                    else:
                        # 未找到结束标记，保留原样
                        result.append(char)
                        i += 1
                        continue

                result.append(char)
                i += 1

            return "".join(result)

        cleaned_content = remove_block_comments(cleaned_content)

        # 移除尾随逗号：JSONC 常见写法允许在 '}' / ']' 前写逗号，但 json.loads 不允许。
        # 这里在不进入字符串的前提下，删除紧挨着闭合符前的 ','。
        def remove_trailing_commas(text: str) -> str:
            out = []
            i = 0
            in_string = False
            escape_next = False
            n = len(text)

            while i < n:
                ch = text[i]

                if escape_next:
                    out.append(ch)
                    escape_next = False
                    i += 1
                    continue

                if ch == "\\":
                    out.append(ch)
                    escape_next = True
                    i += 1
                    continue

                if ch == '"':
                    out.append(ch)
                    in_string = not in_string
                    i += 1
                    continue

                if not in_string and ch == ",":
                    # 预读下一个非空白字符，若是 '}' 或 ']' 则跳过该逗号
                    j = i + 1
                    while j < n and text[j] in " \t\r\n":
                        j += 1
                    if j < n and text[j] in "}]":
                        i += 1
                        continue

                out.append(ch)
                i += 1

            return "".join(out)

        cleaned_content = remove_trailing_commas(cleaned_content)

        # 解析 JSON
        config = json.loads(cleaned_content)

        if logger:
            logger.info(f"成功加载配置文件: {config_path}")
        else:
            print(f"成功加载配置文件: {config_path}")
        return config
    except FileNotFoundError:
        error_msg = f"配置文件不存在: {config_path}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"错误: {error_msg}")
        raise
    except json.JSONDecodeError as e:
        error_msg = f"配置文件格式错误: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"错误: {error_msg}")
        raise
    except Exception as e:
        error_msg = f"加载配置文件时出错: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"错误: {error_msg}")
        raise


def run_step(name, cmd, logger, capture_output=True):
    """执行步骤，带错误处理和日志记录。

    capture_output: 为 True 时捕获子进程输出，失败时写入日志；为 False 时
    子进程 stdout/stderr 直接输出到当前控制台（用于训练等需实时查看 loss 的步骤）。
    """
    logger.info(f"=== 开始步骤: {name} ===")
    logger.info(f"命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=capture_output,
            text=True if capture_output else None,
        )
        if capture_output and result.stdout:
            logger.debug(f"步骤 {name} 输出:\n{result.stdout}")
        logger.info(f"=== 步骤 {name} 完成 ===\n")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"步骤 {name} 失败，退出码: {e.returncode}"
        logger.error(error_msg)
        if capture_output:
            if e.stdout:
                logger.error(f"标准输出:\n{e.stdout}")
            if e.stderr:
                logger.error(f"标准错误:\n{e.stderr}")
        logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
        raise
    except Exception as e:
        error_msg = f"步骤 {name} 执行时出现未预期错误: {e}"
        logger.error(error_msg)
        logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动化迭代总控脚本")
    parser.add_argument(
        "--config", type=str, required=True, help="配置文件路径（JSON格式）"
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="覆盖配置项，格式: key1=value1 key2=value2",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="*",
        help="指定要训练的说话人ID列表，空或未指定则训练 config中所有 speakers 或原始目录",
    )
    args = parser.parse_args()

    # 先加载配置文件（logger还未初始化）
    config = load_config(args.config, logger=None)

    # 处理命令行覆盖
    if args.override:
        for override in args.override:
            if "=" not in override:
                print(f"警告: 忽略无效的覆盖项 '{override}'（格式应为 key=value）")
                continue
            key, value = override.split("=", 1)
            # 简单的嵌套键支持（如 paths.audio_dir）
            keys = key.split(".")
            target = config
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            # 尝试转换值类型
            try:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            except Exception:
                pass
            target[keys[-1]] = value
            print(f"已覆盖配置项 {key} = {value}")

    # 处理说话人筛选
    speakers_map = config.get("speakers", {})
    if args.speakers:
        selected = set(args.speakers)
        # 过滤 mapping
        speakers_map = {
            k: v for k, v in speakers_map.items() if k in selected
        }
        missing = selected - speakers_map.keys()
        for m in missing:
            print(f"警告: 配置中不存在说话人 {m}")
    # 构造说话人列表 [(id, raw_dir)]，若无 mapping 则使用单一原始目录
    if speakers_map:
        speaker_list = list(speakers_map.items())
    else:
        # 若未配置 speakers，则使用 config 中的 raw_audio_dir
        speaker_list = [
            (None, config.get("paths", {}).get("raw_audio_dir"))
        ]

    # 初始化日志系统
    log_config = config.get("logging", {})
    # 将日志路径解析为相对于脚本所在目录的绝对路径
    log_dir = log_config.get("log_dir", "logs")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, log_dir)
    log_level = log_config.get("log_level", "INFO")
    log_file_prefix = log_config.get("log_file_prefix", "orchestrator")
    logger = setup_logging(log_dir, log_level, log_file_prefix)

    # 重新记录配置加载（现在logger已初始化）
    logger.info(f"成功加载配置文件: {args.config}")

    # 从配置中提取参数
    paths = config.get("paths", {})
    audio_split = config.get("audio_split", {})
    labeling = config.get("labeling", {})
    training = config.get("training", {})
    iteration = config.get("iteration", {})

    raw_audio_dir = paths.get("raw_audio_dir")
    split_script = paths.get("split_script")
    audio_dir = paths.get("audio_dir", "audio_chunks")
    labels_dir = paths.get("labels_dir", "labels")
    manifest_dir = paths.get("manifest_dir", "manifests")
    model_dir = paths.get("model_dir", "out/model")
    ggml_dir = paths.get("ggml_dir", "ggml_model")

    interval = iteration.get("interval", 86400)
    once = iteration.get("once", False)
    # 可选：限制最大迭代轮数（用于测试或批处理）。<=0 表示不限制
    max_iterations = int(iteration.get("max_iterations", 0) or 0)
    test_size = iteration.get("test_size", 0)
    annotation_ratio = iteration.get("annotation_ratio", 0.0)
    skip_manifest = iteration.get("skip_manifest", False)
    stop_after_labels = iteration.get("stop_after_labels", False)
    # 新增：允许跳过自动标注步骤，直接从已有标签开始构建清单和训练
    skip_labeling = iteration.get("skip_labeling", False)

    # 获取脚本目录
    base = os.path.dirname(__file__)
    # 允许通过配置覆盖各阶段脚本路径（便于测试注入 stub 脚本）
    labeler_script = paths.get("labeler_script") or os.path.join(
        base, "labeler.py"
    )
    dataset_manager_script = paths.get(
        "dataset_manager_script"
    ) or os.path.join(base, "dataset_manager.py")
    train_script = paths.get("train_script") or os.path.join(
        base, "train_lora.py"
    )
    evaluator_script = paths.get("evaluator_script") or os.path.join(
        base, "evaluator.py"
    )
    converter_script = paths.get("converter_script") or os.path.join(
        base, "converter.py"
    )
    # 默认切分脚本路径
    if not split_script:
        # 假设 split_audio.py 与 orchestrator.py 同级或指定目录内
        split_script = os.path.join(base, "split_audio.py")

    logger.info("=" * 60)
    logger.info("自动化迭代流程开始")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"音频目录: {audio_dir}")
    logger.info(f"标签目录: {labels_dir}")
    logger.info(f"清单目录: {manifest_dir}")
    logger.info(f"模型目录: {model_dir}")
    logger.info(f"GGML目录: {ggml_dir}")
    logger.info(f"迭代周期: {interval}秒")
    logger.info("=" * 60)

    # 按说话人迭代
    for spk, raw_audio_dir in speaker_list:
        iteration_count = 0
        # 更新各阶段目录
        if spk:
            audio_dir = os.path.join(paths.get("audio_dir"), spk)
            labels_dir = os.path.join(paths.get("labels_dir"), spk)
            manifest_dir = os.path.join(paths.get("manifest_dir"), spk)
            model_dir = os.path.join(paths.get("model_dir"), spk)
            ggml_dir = os.path.join(paths.get("ggml_dir"), spk)
        else:
            raw_audio_dir = paths.get("raw_audio_dir")
            audio_dir = paths.get("audio_dir")
            labels_dir = paths.get("labels_dir")
            manifest_dir = paths.get("manifest_dir")
            model_dir = paths.get("model_dir")
            ggml_dir = paths.get("ggml_dir")
        # 确保目录存在
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(manifest_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(ggml_dir, exist_ok=True)
        while True:
            # 在每轮开始前检查是否达到最大迭代次数（避免 continue 跳过底部检查）
            if max_iterations > 0 and iteration_count >= max_iterations:
                logger.info(f"已达到 max_iterations={max_iterations}，退出迭代循环")
                break
            iteration_count += 1
            logger.info("=" * 60)
            logger.info(f"开始第 {iteration_count} 轮迭代")
            logger.info("=" * 60)
            # 如果不是第一次迭代，尝试使用上轮最佳模型进行标注和训练
            prev_model = None
            if iteration_count > 1:
                best_model_dir = os.path.join(model_dir, "best_model")
                if os.path.isdir(best_model_dir):
                    prev_model = best_model_dir
                else:
                    prev_model = model_dir
                logger.info(f"检测到上轮模型: {prev_model}，用于本轮标注和训练")

            try:
                # 0. 音频切分（若提供原始长音频目录，则对每个文件进行切分）
                if raw_audio_dir:
                    sample_rate = str(
                        audio_split.get("sample_rate", 16000)
                    )
                    frame_duration = str(
                        audio_split.get("frame_duration", 30)
                    )
                    vad_aggressiveness = str(
                        audio_split.get("vad_aggressiveness", 2)
                    )
                    min_segment_duration = str(
                        audio_split.get("min_segment_duration", 1500)
                    )
                    merge_threshold = str(
                        audio_split.get("merge_threshold", 15)
                    )

                    for ext in ("*.wav", "*.flac", "*.mp3"):
                        for infile in glob.glob(
                            os.path.join(raw_audio_dir, ext)
                        ):
                            run_step(
                                f"切分音频 {os.path.basename(infile)}",
                                [
                                    sys.executable,
                                    split_script,
                                    "--input",
                                    infile,
                                    "--output_dir",
                                    audio_dir,
                                    "--sample_rate",
                                    sample_rate,
                                    "--frame_duration",
                                    frame_duration,
                                    "--vad_aggressiveness",
                                    vad_aggressiveness,
                                    "--min_segment_duration",
                                    min_segment_duration,
                                    "--merge_threshold",
                                    merge_threshold,
                                ],
                                logger,
                            )

                # 1. 自动标注（可选）：根据 skip_labeling 决定是否执行
                # 默认行为与之前保持一致：若未启用 skip_labeling，则始终尝试自动标注，
                # 由 labeler.py 自行跳过已存在的非空标注文件。
                if skip_labeling:
                    logger.info("已启用 skip_labeling，跳过自动标注步骤，直接使用现有标签。")
                else:
                    # 支持使用微调后的模型进行标注（自迭代功能）
                    max_samples = 0  # 不限制标注样本数
                    temperature = labeling.get("temperature", 1.0)
                    # 支持指定设备，-1 表示 CPU，>=0 表示 GPU 设备编号
                    device = labeling.get("device", -1)
                    # 优先使用上轮模型，否则使用配置模型
                    labeling_model = (
                        prev_model
                        if prev_model
                        else labeling.get(
                            "model_name_or_path",
                            "openai/whisper-large-v3-turbo",
                        )
                    )
                    compression_ratio_threshold = str(
                        labeling.get("compression_ratio_threshold", 1.35)
                    )
                    logprob_threshold = str(
                        labeling.get("logprob_threshold", -1.0)
                    )
                    logger.info(f"使用标注模型: {labeling_model}")
                    run_step(
                        "标签生成",
                        [
                            sys.executable,
                            labeler_script,
                            "--audio_dir",
                            audio_dir,
                            "--labels_dir",
                            labels_dir,
                            "--model_name_or_path",
                            labeling_model,
                            "--device",
                            str(device),
                            "--compression_ratio_threshold",
                            compression_ratio_threshold,
                            "--logprob_threshold",
                            logprob_threshold,
                            "--max_samples",
                            str(max_samples),
                            "--temperature",
                            str(temperature),
                        ],
                        logger,
                    )

                # 2. 构建清单（可跳过）
                if skip_manifest:
                    logger.info("已启用 skip_manifest，跳过清单构建。")
                else:
                    run_step(
                        "清单构建",
                        [
                            sys.executable,
                            dataset_manager_script,
                            "--audio_dir",
                            audio_dir,
                            "--labels_dir",
                            labels_dir,
                            "--output_dir",
                            manifest_dir,
                        ],
                        logger,
                    )
                    # 检查清单文件记录数
                    train_csv = os.path.join(manifest_dir, "train.csv")
                    val_csv = os.path.join(manifest_dir, "val.csv")
                    test_csv = os.path.join(manifest_dir, "test.csv")

                    def count_records(csv_path):
                        try:
                            with open(
                                csv_path, "r", encoding="utf-8"
                            ) as f:
                                lines = f.read().splitlines()
                            return max(0, len(lines) - 1)
                        except Exception:
                            return 0

                    train_count = count_records(train_csv)
                    val_count = count_records(val_csv)
                    test_count = count_records(test_csv)
                    logger.info(
                        f"清单写入数量: train={train_count}, val={val_count}, test={test_count}"
                    )
                    if (
                        train_count == 0
                        and val_count == 0
                        and test_count == 0
                    ):
                        logger.error("清单为空，标签生成或数据管理未产生任何数据，终止流程。")
                        sys.exit(1)
                    # 清单已重建，删除训练检查点以便本轮从头训练（避免沿用旧检查点导致数据与参数不一致）
                    checkpoint_dir = os.path.join(model_dir, "checkpoint")
                    if os.path.isdir(checkpoint_dir):
                        shutil.rmtree(checkpoint_dir)
                        logger.info(f"已删除检查点目录 {checkpoint_dir}，本轮训练将从头开始")

                # 抽取部分标签用于人工标注
                if annotation_ratio > 0:
                    import random
                    import shutil

                    # 准备 annotation 目录
                    anno_labels = labels_dir + "_annotation"
                    anno_audio = audio_dir + "_annotation"
                    os.makedirs(anno_labels, exist_ok=True)
                    os.makedirs(anno_audio, exist_ok=True)
                    # 递归获取所有标签文件
                    txt_paths = []
                    for root2, _, files2 in os.walk(labels_dir):
                        for fname2 in files2:
                            if fname2.endswith(".txt"):
                                txt_paths.append(
                                    os.path.join(root2, fname2)
                                )
                    total = len(txt_paths)
                    logger.info(f"总共找到 {total} 条标签文件（包含子目录）")
                    if total == 0:
                        logger.warning("未找到任何标签文件，跳过抽样")
                    else:
                        # 计算抽样数量，至少 1，至多 total
                        n = max(1, int(total * annotation_ratio))
                        n = min(n, total)
                        logger.info(f"抽样数量: {n}")
                        samples = random.sample(txt_paths, n)
                        for tpath in samples:
                            # 计算相对路径和创建目标子目录
                            rel = os.path.relpath(tpath, labels_dir)
                            subdir = os.path.dirname(rel)
                            target_txt_dir = os.path.join(
                                anno_labels, subdir
                            )
                            os.makedirs(target_txt_dir, exist_ok=True)
                            shutil.copy(
                                tpath,
                                os.path.join(
                                    target_txt_dir, os.path.basename(tpath)
                                ),
                            )
                            # 对应音频的相对路径
                            wav_rel = os.path.splitext(rel)[0] + ".wav"
                            src_wav = os.path.join(audio_dir, wav_rel)
                            if os.path.exists(src_wav):
                                target_wav_dir = os.path.join(
                                    anno_audio, subdir
                                )
                                os.makedirs(target_wav_dir, exist_ok=True)
                                shutil.copy(
                                    src_wav,
                                    os.path.join(
                                        target_wav_dir,
                                        os.path.basename(src_wav),
                                    ),
                                )
                        logger.info(
                            f"已抽取 {n} 条文件到人工标注目录: {anno_labels}, {anno_audio}"
                        )

                # 如果只运行到标注阶段，退出流程
                if stop_after_labels:
                    logger.info("已启用 stop_after_labels，流程结束。")
                    sys.exit(0)

                # 3. 如果 test_size>0，截断 manifest
                if test_size > 0:
                    logger.info(f"测试模式：截取每个 split 前 {test_size} 条样本")
                    for fname in ["train.csv", "val.csv", "test.csv"]:
                        path = os.path.join(manifest_dir, fname)
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                lines = f.read().splitlines()
                            header, rest = lines[0], lines[1:]
                            truncated = rest[:test_size]
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(header + "\n")
                                f.write("\n".join(truncated) + "\n")
                            logger.info(
                                f"已截断 {fname} 到 {len(truncated)} 条记录"
                            )
                        except Exception as e:
                            logger.error(f"截断 {fname} 失败: {e}")
                            logger.error(
                                f"错误堆栈:\n{traceback.format_exc()}"
                            )

                # 4. LoRA 微调 (使用已验证 train_lora.py)
                train_csv = os.path.join(manifest_dir, "train.csv")
                val_csv = os.path.join(manifest_dir, "val.csv")
                test_csv = os.path.join(manifest_dir, "test.csv")

                # 构建训练命令，使用配置文件中的训练超参
                train_cmd = [
                    sys.executable,
                    train_script,
                    "--train_manifest",
                    train_csv,
                    "--eval_manifest",
                    val_csv,
                    "--test_manifest",
                    test_csv,
                    "--model_name_or_path",
                    prev_model
                    if prev_model
                    else training.get(
                        "model_name_or_path",
                        "openai/whisper-large-v3-turbo",
                    ),
                    "--output_dir",
                    model_dir,
                    "--num_train_epochs",
                    str(training.get("num_train_epochs", 3)),
                    "--train_batch_size",
                    str(training.get("train_batch_size", 4)),
                    "--eval_batch_size",
                    str(training.get("eval_batch_size", 4)),
                    "--mixed_precision",
                    training.get("mixed_precision", "fp16"),
                    "--gradient_accumulation_steps",
                    str(training.get("gradient_accumulation_steps", 4)),
                    "--language",
                    training.get("language", "zh"),
                    "--task",
                    training.get("task", "transcribe"),
                    "--max_new_tokens",
                    str(training.get("max_new_tokens", 256)),
                    "--no_repeat_ngram_size",
                    str(training.get("no_repeat_ngram_size", 3)),
                    "--length_penalty",
                    str(training.get("length_penalty", 1.2)),
                    "--eval_metric",
                    training.get("eval_metric", "cer"),
                    "--repetition_penalty",
                    str(training.get("repetition_penalty", 2.0)),
                ]

                # 添加可选训练超参
                if "learning_rate" in training:
                    train_cmd.extend(
                        ["--learning_rate", str(training["learning_rate"])]
                    )
                if "warmup_steps" in training:
                    train_cmd.extend(
                        ["--warmup_steps", str(training["warmup_steps"])]
                    )
                if "weight_decay" in training:
                    train_cmd.extend(
                        ["--weight_decay", str(training["weight_decay"])]
                    )
                if "lora_r" in training:
                    train_cmd.extend(["--lora_r", str(training["lora_r"])])
                if "lora_alpha" in training:
                    train_cmd.extend(
                        ["--lora_alpha", str(training["lora_alpha"])]
                    )
                if "lora_dropout" in training:
                    train_cmd.extend(
                        ["--lora_dropout", str(training["lora_dropout"])]
                    )
                if "target_modules" in training:
                    train_cmd.extend(
                        ["--target_modules", training["target_modules"]]
                    )
                if training.get("gradient_checkpointing", False):
                    train_cmd.append("--gradient_checkpointing")
                if (
                    "checkpoint_steps" in training
                    and training["checkpoint_steps"]
                ):
                    train_cmd.extend(
                        [
                            "--checkpoint_steps",
                            str(training["checkpoint_steps"]),
                        ]
                    )
                if "checkpoint_epochs" in training:
                    train_cmd.extend(
                        [
                            "--checkpoint_epochs",
                            str(training["checkpoint_epochs"]),
                        ]
                    )
                if "early_stopping_patience" in training:
                    train_cmd.extend(
                        [
                            "--early_stopping_patience",
                            str(training["early_stopping_patience"]),
                        ]
                    )
                if "early_stopping_threshold" in training:
                    train_cmd.extend(
                        [
                            "--early_stopping_threshold",
                            str(training["early_stopping_threshold"]),
                        ]
                    )
                if training.get("save_merged_model", False):
                    train_cmd.append("--save_merged_model")

                # 训练步骤不捕获输出，便于在控制台查看每个 epoch 的 loss
                run_step("模型训练", train_cmd, logger, capture_output=False)

                # 5. 模型评估
                eval_cmd = [
                    sys.executable,
                    evaluator_script,
                    "--model_dir",
                    model_dir,
                    "--test_manifest",
                    test_csv,
                    "--output_file",
                    os.path.join(manifest_dir, "eval_results.txt"),
                    "--metric",
                    training.get("eval_metric", "cer"),
                    "--language",
                    training.get("language", "zh"),
                    "--task",
                    training.get("task", "transcribe"),
                ]
                # 传递基础模型路径（用于 LoRA 模型合并）
                if "model_name_or_path" in training:
                    eval_cmd.extend(
                        [
                            "--base_model_path",
                            training["model_name_or_path"],
                        ]
                    )
                run_step("模型评估", eval_cmd, logger)

                # 基线评估：评估上一轮模型或基准模型
                baseline_eval_file = os.path.join(
                    manifest_dir, "eval_baseline.txt"
                )
                base_model_path = (
                    prev_model
                    if prev_model
                    else training.get("model_name_or_path")
                )
                base_eval_cmd = [
                    sys.executable,
                    evaluator_script,
                    "--model_dir",
                    base_model_path,
                    "--test_manifest",
                    test_csv,
                    "--output_file",
                    baseline_eval_file,
                    "--metric",
                    training.get("eval_metric", "cer"),
                    "--language",
                    training.get("language", "zh"),
                    "--task",
                    training.get("task", "transcribe"),
                ]
                run_step("基线评估", base_eval_cmd, logger)

                # 解析并比较基线与微调后指标
                def parse_metric(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith(
                                    f"{training.get('eval_metric', 'cer').upper()}: "
                                ):
                                    return float(
                                        line.split(":")[1].strip()
                                    )
                    except Exception:
                        return None

                baseline_metric = parse_metric(baseline_eval_file)
                tuned_metric = parse_metric(
                    os.path.join(manifest_dir, "eval_results.txt")
                )
                logger.info(
                    f"基线指标: {baseline_metric}, 微调后指标: {tuned_metric}"
                )
                # 只有当微调后指标优于基线时才进行模型转换
                if (
                    tuned_metric is None
                    or baseline_metric is None
                    or tuned_metric <= baseline_metric
                ):
                    logger.info("微调后指标未优于基线，跳过模型转换，进入下一轮迭代")
                    continue

                # 6. 转换 GGML

                # 6. 转换 GGML
                # 使用 H5 转换脚本进行 GGML 模型转换
                run_step(
                    "模型转换",
                    [
                        sys.executable,
                        converter_script,
                        "--model_dir",
                        model_dir,
                        "--output_dir",
                        ggml_dir,
                        "--use_h5_to_ggml",
                    ],
                    logger,
                )

                # 重命名 GGML 模型文件为 原始模型名-日期-finetune.bin
                logger.info("开始重命名 GGML 模型文件...")
                orig_model = os.path.basename(os.path.normpath(model_dir))
                date_str = datetime.datetime.now().strftime("%Y%m%d")
                new_name = f"{orig_model}-{date_str}-finetune.bin"
                # 查找输出目录下的 bin 文件
                bin_files = glob.glob(os.path.join(ggml_dir, "*.bin"))
                if bin_files:
                    old_path = bin_files[0]
                    new_path = os.path.join(ggml_dir, new_name)
                    os.rename(old_path, new_path)
                    logger.info(
                        f"已将 {os.path.basename(old_path)} 重命名为 {new_name}"
                    )
                else:
                    logger.warning("未找到需要重命名的 GGML bin 文件")

                logger.info("=" * 60)
                logger.info(f"第 {iteration_count} 轮迭代完成")
                logger.info("=" * 60)

            except KeyboardInterrupt:
                logger.info("\n收到中断信号，正在退出...")
                sys.exit(0)
            except Exception as e:
                # 打印详细错误信息以便排查
                error_msg = f"迭代管道出现错误: {e}"
                logger.error(error_msg)
                logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
                logger.info("等待下次周期继续...")

            if once:
                logger.info("已启用 once 模式，执行一次后退出")
                break

            logger.info(f"等待 {interval} 秒后开始下一轮迭代...\n")
            time.sleep(interval)
