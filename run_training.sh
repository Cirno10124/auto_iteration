#!/usr/bin/env bash
# 简易完整流程脚本：构建清单 → 训练 → 评估 → 转换
# 依赖 orchestrator.py，通过 --config 与 --override 控制行为
set -e
cd "$(dirname "$0")"

# 配置文件路径（可通过环境变量覆盖）
CONFIG="${ORCHESTRATOR_CONFIG:-orchestrator_config.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "错误: 配置文件不存在: $CONFIG"
  echo "请在该目录创建 orchestrator_config.json，或设置环境变量 ORCHESTRATOR_CONFIG 指向已有配置。"
  exit 1
fi

# 使用已切分好的音频目录进行完整训练流程：
# - paths.raw_audio_dir 置空，避免再次执行切分
# - paths.audio_dir 指向已分段目录（默认 segments）
# - paths.labels_dir / manifest_dir / model_dir / ggml_dir 指向各自输出目录
# - iteration.once=true 仅执行一轮
python orchestrator.py \
  --config "$CONFIG" \
  --override \
    "paths.raw_audio_dir=" \
    "paths.audio_dir=segments" \
    "paths.labels_dir=out/labels" \
    "paths.manifest_dir=out/manifests" \
    "paths.model_dir=out/model" \
    "paths.ggml_dir=out/ggml_model" \
    "iteration.once=true" \
    "iteration.test_size=0" \
    "iteration.annotation_ratio=0.0" \
    "iteration.skip_manifest=false" \
    "iteration.stop_after_labels=false"

