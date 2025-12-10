#!/usr/bin/env bash
# 简易完整流程脚本：构建清单→训练→评估→转换
set -e
cd "$(dirname "$0")"

python orchestrator.py \
  --audio_dir segments \             # 使用已切分的音频
  --labels_dir out/labels \       # 使用已生成的标签
  --manifest_dir out/manifests \  # 清单输出目录
  --model_dir out/model \         # 模型输出目录
  --ggml_dir out/ggml_model \      # GGML 输出目录
  --once
