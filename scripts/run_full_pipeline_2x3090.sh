#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_full_pipeline_2x3090.sh [stable|aggressive]
#
# Before run:
# 1) Put your unlabeled audio segments (.wav/.flac/.mp3) into:
#    /exp/auto_iter/audio_chunks
# 2) Export HF token:
#    export HF_HUB_TOKEN=your_token
# 3) Adjust paths in config_template/orchestrator_config.2x3090.*.json if needed.

MODE="${1:-stable}"
if [[ "${MODE}" != "stable" && "${MODE}" != "aggressive" ]]; then
  echo "Invalid mode: ${MODE}. Use stable or aggressive."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${ROOT_DIR}/config_template/orchestrator_config.2x3090.${MODE}.json"

if [[ ! -f "${CFG}" ]]; then
  echo "Config not found: ${CFG}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1
mkdir -p /exp/auto_iter/{audio_chunks,labels,manifests,model,ggml,logs}

echo "[INFO] Running mode: ${MODE}"
echo "[INFO] Config: ${CFG}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python "${ROOT_DIR}/orchestrator.py" --config "${CFG}"

echo "[INFO] Pipeline finished. Evaluation files:"
echo "  - /exp/auto_iter/manifests/eval_baseline.txt"
echo "  - /exp/auto_iter/manifests/eval_results.txt"
