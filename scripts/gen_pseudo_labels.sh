#!/usr/bin/env bash
# 简易伪标签生成：切分 → 标注 → 抽样，然后退出（不构建清单、不训练）
# 依赖 orchestrator.py，通过 --config 与 --override 控制行为
set -e
cd "$(dirname "$0")"

# 配置文件路径（可经环境变量覆盖）
CONFIG="${ORCHESTRATOR_CONFIG:-orchestrator_config.json}"

# 以下路径与比例可通过 override 覆盖，此处为默认伪标签流程取值
RAW_AUDIO_DIR="${RAW_AUDIO_DIR:-raw_audio_jcr}"
AUDIO_DIR="${AUDIO_DIR:-segments}"
LABELS_DIR="${LABELS_DIR:-out/labels}"
MANIFEST_DIR="${MANIFEST_DIR:-out/manifests}"
ANNOTATION_RATIO="${ANNOTATION_RATIO:-0.1}"

if [[ ! -f "$CONFIG" ]]; then
  echo "错误: 配置文件不存在: $CONFIG"
  echo "请在该目录创建 orchestrator_config.json，或设置环境变量 ORCHESTRATOR_CONFIG 指向已有配置。"
  exit 1
fi

python orchestrator.py \
  --config "$CONFIG" \
  --override \
    "paths.raw_audio_dir=$RAW_AUDIO_DIR" \
    "paths.audio_dir=$AUDIO_DIR" \
    "paths.labels_dir=$LABELS_DIR" \
    "paths.manifest_dir=$MANIFEST_DIR" \
    "iteration.annotation_ratio=$ANNOTATION_RATIO" \
    "iteration.skip_manifest=true" \
    "iteration.stop_after_manifests=true" \
    "iteration.once=true"
