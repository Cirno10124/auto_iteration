"""运行前配置校验（必填项、类型、范围）。"""

from typing import Any, Dict, List


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int_like(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return float(v).is_integer()
    return False


def validate_config(config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    required_sections = [
        "paths",
        "audio_split",
        "labeling",
        "training",
        "iteration",
        "logging",
    ]
    for sec in required_sections:
        if sec not in config or not isinstance(config.get(sec), dict):
            errors.append(f"缺少配置节或类型错误: {sec}")

    paths = config.get("paths", {})
    for k in ["audio_dir", "labels_dir", "manifest_dir", "model_dir", "ggml_dir"]:
        if not isinstance(paths.get(k), str) or not paths.get(k):
            errors.append(f"paths.{k} 必须是非空字符串")

    audio_split = config.get("audio_split", {})
    if not isinstance(audio_split.get("sample_rate"), int) or audio_split.get(
        "sample_rate", 0
    ) <= 0:
        errors.append("audio_split.sample_rate 必须是正整数")
    if not isinstance(audio_split.get("frame_duration"), int) or audio_split.get(
        "frame_duration", 0
    ) <= 0:
        errors.append("audio_split.frame_duration 必须是正整数")
    if not isinstance(audio_split.get("vad_aggressiveness"), int) or not (
        0 <= audio_split.get("vad_aggressiveness", -1) <= 3
    ):
        errors.append("audio_split.vad_aggressiveness 必须在 [0,3]")
    if not _is_number(audio_split.get("min_segment_duration")) or audio_split.get(
        "min_segment_duration", 0
    ) <= 0:
        errors.append("audio_split.min_segment_duration 必须为正数")
    if not _is_number(audio_split.get("merge_threshold")) or audio_split.get(
        "merge_threshold", -1
    ) < 0:
        errors.append("audio_split.merge_threshold 必须为 >=0 的数值")

    labeling = config.get("labeling", {})
    if not isinstance(labeling.get("model_name_or_path"), str) or not labeling.get(
        "model_name_or_path"
    ):
        errors.append("labeling.model_name_or_path 必须是非空字符串")
    if not _is_number(labeling.get("temperature")) or labeling.get("temperature", 0) <= 0:
        errors.append("labeling.temperature 必须为 >0 的数值")
    if not _is_int_like(labeling.get("device")):
        errors.append("labeling.device 必须是整数（-1 表示 CPU）")

    training = config.get("training", {})
    if not isinstance(training.get("model_name_or_path"), str) or not training.get(
        "model_name_or_path"
    ):
        errors.append("training.model_name_or_path 必须是非空字符串")
    for k in ["num_train_epochs", "train_batch_size", "eval_batch_size"]:
        if not isinstance(training.get(k), int) or training.get(k, 0) <= 0:
            errors.append(f"training.{k} 必须是正整数")
    if not _is_number(training.get("gradient_accumulation_steps")) or training.get(
        "gradient_accumulation_steps", 0
    ) <= 0:
        errors.append("training.gradient_accumulation_steps 必须为 >0 的数值")

    iteration = config.get("iteration", {})
    if not isinstance(iteration.get("interval"), int) or iteration.get("interval", -1) < 0:
        errors.append("iteration.interval 必须是 >=0 的整数")
    if not isinstance(iteration.get("once"), bool):
        errors.append("iteration.once 必须是布尔值")
    if not isinstance(iteration.get("skip_manifest"), bool):
        errors.append("iteration.skip_manifest 必须是布尔值")
    if "stop_after_manifests" in iteration and not isinstance(
        iteration.get("stop_after_manifests"), bool
    ):
        errors.append("iteration.stop_after_manifests 必须是布尔值")
    if "stop_after_labels" in iteration and not isinstance(
        iteration.get("stop_after_labels"), bool
    ):
        errors.append(
            "iteration.stop_after_labels 必须是布尔值（已废弃，请改用 stop_after_manifests）"
        )
    if not isinstance(iteration.get("skip_labeling"), bool):
        errors.append("iteration.skip_labeling 必须是布尔值")
    if not _is_number(iteration.get("annotation_ratio")) or not (
        0.0 <= float(iteration.get("annotation_ratio", -1)) <= 1.0
    ):
        errors.append("iteration.annotation_ratio 必须在 [0,1]")

    logging_cfg = config.get("logging", {})
    if not isinstance(logging_cfg.get("log_dir"), str) or not logging_cfg.get("log_dir"):
        errors.append("logging.log_dir 必须是非空字符串")
    if not isinstance(logging_cfg.get("log_file_prefix"), str) or not logging_cfg.get(
        "log_file_prefix"
    ):
        errors.append("logging.log_file_prefix 必须是非空字符串")
    if str(logging_cfg.get("log_level", "")).upper() not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        errors.append("logging.log_level 必须是 DEBUG/INFO/WARNING/ERROR/CRITICAL")

    return errors
