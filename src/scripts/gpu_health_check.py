#!/usr/bin/env python3
"""GPU 容器环境健康检查。

检查项：
1) nvidia-smi 可调用
2) torch 可导入
3) torch.cuda.is_available() 为 True
4) 至少检测到 1 块 GPU
"""

import argparse
import subprocess


def _run_nvidia_smi() -> None:
    cmd = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi 执行失败(exit={result.returncode}): {result.stderr.strip()}"
        )
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi 未返回 GPU 信息")
    print(f"[OK] nvidia-smi: detected {len(lines)} GPU(s)")
    for i, ln in enumerate(lines, start=1):
        print(f"  - GPU{i}: {ln}")


def _run_torch_check() -> None:
    import torch

    print(f"[INFO] torch version: {torch.__version__}")
    print(f"[INFO] torch cuda build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() == False")
    cnt = torch.cuda.device_count()
    if cnt <= 0:
        raise RuntimeError("torch.cuda.device_count() <= 0")
    print(f"[OK] torch cuda available, device_count={cnt}")
    for i in range(cnt):
        print(f"  - torch GPU{i}: {torch.cuda.get_device_name(i)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU health check")
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="仅检查 nvidia-smi，不检查 torch",
    )
    args = parser.parse_args()

    try:
        _run_nvidia_smi()
        if not args.skip_torch:
            _run_torch_check()
    except Exception as e:
        print(f"[FAIL] GPU health check failed: {e}")
        return 1
    print("[PASS] GPU health check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
