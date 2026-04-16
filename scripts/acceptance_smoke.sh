#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-config_template/orchestrator_config.test.json}"
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"
SKIP_PYTEST="${SKIP_PYTEST:-0}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"

run_step() {
  local name="$1"
  local cmd="$2"
  echo ""
  echo "=== ${name} ==="
  echo "${cmd}"
  eval "${cmd}"
}

echo "验收冒烟脚本（Bash）"
echo "CONFIG_PATH=${CONFIG_PATH}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "配置文件不存在: ${CONFIG_PATH}" >&2
  exit 1
fi

run_step "Python 版本检查" "python --version"

if [[ "${SKIP_PIP_INSTALL}" != "1" ]]; then
  run_step "安装依赖" "pip install -r requirements.txt"
fi

if [[ "${SKIP_GPU_CHECK}" != "1" ]]; then
  echo ""
  echo "=== GPU 检查（失败不终止） ==="
  if ! nvidia-smi; then
    echo "未检测到可用 GPU 或命令不可用，继续执行。"
  fi
fi

run_step "输出当前版本" "git rev-parse --short HEAD"
run_step "执行单次全流程冒烟" "python orchestrator.py --config \"${CONFIG_PATH}\" --override iteration.once=true"

if [[ "${SKIP_PYTEST}" != "1" ]]; then
  run_step "执行轻量测试集" "pytest -q"
fi

echo ""
echo "=== 冒烟执行完成 ==="
echo "请按 docs/验收部署与测试清单.md 核对产物目录、日志和评估报告。"
