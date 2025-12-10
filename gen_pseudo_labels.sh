#!/usr/bin/env bash
# 简易伪标签生成脚本：仅切分→标注→抽样
set -e
cd "$(dirname "$0")"

python orchestrator.py \
  --raw_audio_dir raw_audio_jcr \  # 原始音频目录，请根据实际路径修改
  --audio_dir segments \            # VAD 切片输出目录
  --labels_dir out/labels \      # 标注输出目录
  --manifest_dir out/manifests \ # 清单输出目录（用于后续流程）
  --annotation_ratio 0.1 \       # 抽样比例
  --skip_manifest \              # 跳过清单构建，只做标注和抽样
  --stop_after_labels \          # 完成后退出
  --once
