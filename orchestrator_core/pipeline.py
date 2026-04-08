"""自动化迭代主流程（按说话人、按轮次）。"""

import datetime
import glob
import os
import random
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from logging_utils import (
    iteration_id_var,
    set_orchestrator_context,
    short_model_hint,
    speaker_id_var,
    step_var,
)

from orchestrator_core.paths import PROJECT_ROOT
from orchestrator_core.step_runner import run_step


def _run_step_with_policy(
    *,
    logger: Any,
    step_name: str,
    cmd: List[str],
    model_hint: str = "-",
    capture_output: bool = True,
    retries: int = 0,
    on_failure: str = "raise",
) -> bool:
    """统一步骤失败策略：retry / skip / terminate(raise)。"""
    attempts = max(1, int(retries) + 1)
    for attempt in range(1, attempts + 1):
        try:
            return run_step(
                step_name,
                cmd,
                logger,
                capture_output=capture_output,
                model=model_hint,
            )
        except Exception as e:
            is_last = attempt >= attempts
            logger.warning(
                "步骤异常 | step=%s | attempt=%s/%s | on_failure=%s | err=%s",
                step_name,
                attempt,
                attempts,
                on_failure,
                e,
            )
            if not is_last:
                continue
            if on_failure == "skip":
                logger.warning("步骤已跳过 | step=%s", step_name)
                return False
            logger.error("步骤致命失败 | step=%s | action=terminate", step_name)
            raise
    return False


def run_orchestrator_loop(
    logger: Any,
    config: Dict[str, Any],
    config_path: str,
    speaker_list: List[Tuple[Optional[str], Any]],
) -> None:
    """执行与 legacy orchestrator 相同的主循环。"""
    paths = config.get("paths", {})
    audio_split = config.get("audio_split", {})
    labeling = config.get("labeling", {})
    training = config.get("training", {})
    iteration = config.get("iteration", {})

    audio_dir = paths.get("audio_dir", "audio_chunks")
    labels_dir = paths.get("labels_dir", "labels")
    manifest_dir = paths.get("manifest_dir", "manifests")
    model_dir = paths.get("model_dir", "out/model")
    ggml_dir = paths.get("ggml_dir", "ggml_model")

    interval = iteration.get("interval", 86400)
    once = iteration.get("once", False)
    max_iterations = int(iteration.get("max_iterations", 0) or 0)
    test_size = iteration.get("test_size", 0)
    annotation_ratio = iteration.get("annotation_ratio", 0.0)
    skip_manifest = iteration.get("skip_manifest", False)
    stop_after_labels = iteration.get("stop_after_labels", False)
    skip_labeling = iteration.get("skip_labeling", False)
    retry_split = int(iteration.get("retry_split", 1) or 1)
    retry_label = int(iteration.get("retry_label", 1) or 1)
    retry_manifest = int(iteration.get("retry_manifest", 1) or 1)
    retry_train = int(iteration.get("retry_train", 0) or 0)
    retry_eval = int(iteration.get("retry_eval", 0) or 0)
    retry_convert = int(iteration.get("retry_convert", 0) or 0)

    base = PROJECT_ROOT
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
    split_script = paths.get("split_script")
    if not split_script:
        split_script = os.path.join(base, "split_audio.py")

    logger.info("=" * 60)
    logger.info("自动化迭代流程开始")
    logger.info("=" * 60)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"音频目录: {audio_dir}")
    logger.info(f"标签目录: {labels_dir}")
    logger.info(f"清单目录: {manifest_dir}")
    logger.info(f"模型目录: {model_dir}")
    logger.info(f"GGML目录: {ggml_dir}")
    logger.info(f"迭代周期: {interval}秒")
    logger.info("=" * 60)

    for spk, raw_audio_dir in speaker_list:
        set_orchestrator_context(
            speaker_id=spk if spk else "-",
            iteration_id="-",
            step="speaker_init",
            model="-",
        )
        iteration_count = 0
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
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(manifest_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(ggml_dir, exist_ok=True)
        while True:
            if max_iterations > 0 and iteration_count >= max_iterations:
                logger.info(
                    f"已达到 max_iterations={max_iterations}，退出迭代循环"
                )
                break
            iteration_count += 1
            set_orchestrator_context(
                speaker_id=spk if spk else "-",
                iteration_id=str(iteration_count),
                step="pipeline",
                model="-",
            )
            logger.info("=" * 60)
            logger.info(f"开始第 {iteration_count} 轮迭代")
            logger.info("=" * 60)
            prev_model = None
            if iteration_count > 1:
                best_model_dir = os.path.join(model_dir, "best_model")
                if os.path.isdir(best_model_dir):
                    prev_model = best_model_dir
                else:
                    prev_model = model_dir
                logger.info(f"检测到上轮模型: {prev_model}，用于本轮标注和训练")

            try:
                if raw_audio_dir:
                    sample_rate = str(audio_split.get("sample_rate", 16000))
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
                            _run_step_with_policy(
                                logger=logger,
                                step_name=f"切分音频 {os.path.basename(infile)}",
                                cmd=[
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
                                model_hint=f"in={os.path.basename(infile)}",
                                retries=retry_split,
                                on_failure="skip",
                            )

                if skip_labeling:
                    logger.info(
                        "已启用 skip_labeling，跳过自动标注步骤，直接使用现有标签。"
                    )
                else:
                    max_samples = 0
                    temperature = labeling.get("temperature", 1.0)
                    device = labeling.get("device", -1)
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
                    _run_step_with_policy(
                        logger=logger,
                        step_name="标签生成",
                        cmd=[
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
                        model_hint=short_model_hint(labeling_model),
                        retries=retry_label,
                        on_failure="raise",
                    )

                if skip_manifest:
                    logger.info("已启用 skip_manifest，跳过清单构建。")
                else:
                    _run_step_with_policy(
                        logger=logger,
                        step_name="清单构建",
                        cmd=[
                            sys.executable,
                            dataset_manager_script,
                            "--audio_dir",
                            audio_dir,
                            "--labels_dir",
                            labels_dir,
                            "--output_dir",
                            manifest_dir,
                        ],
                        model_hint="-",
                        retries=retry_manifest,
                        on_failure="raise",
                    )
                    train_csv = os.path.join(manifest_dir, "train.csv")
                    val_csv = os.path.join(manifest_dir, "val.csv")
                    test_csv = os.path.join(manifest_dir, "test.csv")

                    def count_records(csv_path: str) -> int:
                        try:
                            with open(
                                csv_path, "r", encoding="utf-8"
                            ) as f:
                                lines = f.read().splitlines()
                            return max(0, len(lines) - 1)
                        except (OSError, UnicodeError):
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
                        logger.error(
                            "清单为空，标签生成或数据管理未产生任何数据，终止流程。"
                        )
                        sys.exit(1)
                    checkpoint_dir = os.path.join(model_dir, "checkpoint")
                    if os.path.isdir(checkpoint_dir):
                        shutil.rmtree(checkpoint_dir)
                        logger.info(
                            f"已删除检查点目录 {checkpoint_dir}，本轮训练将从头开始"
                        )

                if annotation_ratio > 0:
                    anno_labels = labels_dir + "_annotation"
                    anno_audio = audio_dir + "_annotation"
                    os.makedirs(anno_labels, exist_ok=True)
                    os.makedirs(anno_audio, exist_ok=True)
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
                        n = max(1, int(total * annotation_ratio))
                        n = min(n, total)
                        logger.info(f"抽样数量: {n}")
                        samples = random.sample(txt_paths, n)
                        for tpath in samples:
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

                if stop_after_labels:
                    logger.info("已启用 stop_after_labels，流程结束。")
                    sys.exit(0)

                if test_size > 0:
                    logger.info(
                        f"测试模式：截取每个 split 前 {test_size} 条样本"
                    )
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
                        except (OSError, IndexError, UnicodeError) as e:
                            logger.error(f"截断 {fname} 失败: {e}")
                            logger.error(
                                f"错误堆栈:\n{traceback.format_exc()}"
                            )

                train_csv = os.path.join(manifest_dir, "train.csv")
                val_csv = os.path.join(manifest_dir, "val.csv")
                test_csv = os.path.join(manifest_dir, "test.csv")

                train_src_model = (
                    prev_model
                    if prev_model
                    else training.get(
                        "model_name_or_path",
                        "openai/whisper-large-v3-turbo",
                    )
                )

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
                    train_src_model,
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

                _run_step_with_policy(
                    logger=logger,
                    step_name="模型训练",
                    cmd=train_cmd,
                    model_hint=short_model_hint(train_src_model),
                    capture_output=False,
                    retries=retry_train,
                    on_failure="raise",
                )

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
                if "model_name_or_path" in training:
                    eval_cmd.extend(
                        [
                            "--base_model_path",
                            training["model_name_or_path"],
                        ]
                    )
                _run_step_with_policy(
                    logger=logger,
                    step_name="模型评估",
                    cmd=eval_cmd,
                    model_hint=short_model_hint(model_dir),
                    retries=retry_eval,
                    on_failure="raise",
                )

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
                _run_step_with_policy(
                    logger=logger,
                    step_name="基线评估",
                    cmd=base_eval_cmd,
                    model_hint=short_model_hint(base_model_path),
                    retries=retry_eval,
                    on_failure="raise",
                )

                def parse_metric(file_path: str) -> Optional[float]:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith(
                                    f"{training.get('eval_metric', 'cer').upper()}: "
                                ):
                                    return float(
                                        line.split(":")[1].strip()
                                    )
                    except (OSError, ValueError):
                        return None

                baseline_metric = parse_metric(baseline_eval_file)
                tuned_metric = parse_metric(
                    os.path.join(manifest_dir, "eval_results.txt")
                )
                logger.info(
                    f"基线指标: {baseline_metric}, 微调后指标: {tuned_metric}"
                )
                if (
                    tuned_metric is None
                    or baseline_metric is None
                    or tuned_metric <= baseline_metric
                ):
                    logger.info(
                        "微调后指标未优于基线，跳过模型转换，进入下一轮迭代"
                    )
                    continue

                _run_step_with_policy(
                    logger=logger,
                    step_name="模型转换",
                    cmd=[
                        sys.executable,
                        converter_script,
                        "--model_dir",
                        model_dir,
                        "--output_dir",
                        ggml_dir,
                        "--use_h5_to_ggml",
                    ],
                    model_hint=short_model_hint(model_dir),
                    retries=retry_convert,
                    on_failure="raise",
                )

                logger.info("开始重命名 GGML 模型文件...")
                orig_model = os.path.basename(os.path.normpath(model_dir))
                date_str = datetime.datetime.now().strftime("%Y%m%d")
                new_name = f"{orig_model}-{date_str}-finetune.bin"
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
                logger.error(
                    "迭代管道出现错误 | speaker=%s | iter=%s | step=%s | "
                    "type=%s | err=%s",
                    speaker_id_var.get(),
                    iteration_id_var.get(),
                    step_var.get(),
                    type(e).__name__,
                    e,
                )
                logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
                logger.info("等待下次周期继续...")

            if once:
                logger.info("已启用 once 模式，执行一次后退出")
                break

            logger.info(f"等待 {interval} 秒后开始下一轮迭代...\n")
            time.sleep(interval)
