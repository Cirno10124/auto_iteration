#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import traceback
from typing import List

import torch
from transformers import pipeline


def setup_logging():
    """控制台仅保留 WARNING 及以上，详细过程用 debug（需设置环境或调低级别查看）。"""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for name in ("transformers", "datasets", "torch", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def _print_progress(current: int, total: int) -> None:
    """控制台单行进度（不换行，用 \\r 刷新）。"""
    if total <= 0:
        return
    sys.stdout.write(f"\r标注进度 {current}/{total}")
    sys.stdout.flush()


def _collect_audio_paths(audio_dir: str, max_samples: int) -> List[str]:
    """按 os.walk 顺序收集待标注音频路径；max_samples>0 时最多收集该数量。"""
    paths: List[str] = []
    for root, _, files in os.walk(audio_dir):
        for fname in files:
            if not fname.lower().endswith((".wav", ".flac", ".mp3")):
                continue
            paths.append(os.path.join(root, fname))
            if max_samples > 0 and len(paths) >= max_samples:
                return paths
    return paths


def main():
    parser = argparse.ArgumentParser(description="Whisper 自动标注脚本")
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="音频文件目录"
    )
    parser.add_argument(
        "--labels_dir", type=str, required=True, help="标注文件输出目录"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Whisper 模型名称或路径。可以是 Hugging Face 模型ID（如 openai/whisper-large-v3-turbo）或本地微调后的模型路径（如 out/model）",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=(0 if torch.cuda.is_available() else -1),
        help="运行设备，-1 表示 CPU，>=0 表示 GPU 设备编号",
    )
    parser.add_argument(
        "--chunk_length_s",
        type=float,
        default=30.0,
        help="长音频分段长度（秒）",
    )
    parser.add_argument(
        "--no_speech_threshold",
        type=float,
        default=0.6,
        help="跳过无语音样本阈值",
    )
    parser.add_argument(
        "--compression_ratio_threshold",
        type=float,
        default=1.35,
        help="跳过高压缩率样本阈值",
    )
    parser.add_argument(
        "--logprob_threshold",
        type=float,
        default=-1.0,
        help="跳过低对数概率样本阈值",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="生成温度，必须大于0以免 None 引发错误，默认为1.0",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="测试模式下最大标注音频数，不提供则默认100，<=0 则不限制",
    )
    args = parser.parse_args()
    # 未提供 max_samples 时，默认限制为100条
    if args.max_samples is None:
        args.max_samples = 100

    logger = setup_logging()

    os.makedirs(args.labels_dir, exist_ok=True)

    logger.debug("=" * 60)
    logger.debug("Whisper 自动标注开始")
    logger.debug(f"音频目录: {args.audio_dir}")
    logger.debug(f"标签目录: {args.labels_dir}")
    logger.debug(f"使用模型: {args.model_name_or_path}")
    logger.debug(
        f"设备: {'CPU' if args.device == -1 else f'GPU {args.device}'}"
    )

    # 防止 temperature 为 None 导致生成过程报错，需指定温度值
    generate_kwargs = {
        "no_speech_threshold": args.no_speech_threshold,
        "compression_ratio_threshold": args.compression_ratio_threshold,
        "logprob_threshold": args.logprob_threshold,
        "temperature": args.temperature,
    }
    logger.debug(f"生成参数: {generate_kwargs}")

    try:
        logger.debug("正在加载模型...")
        asr = pipeline(
            "automatic-speech-recognition",
            model=args.model_name_or_path,
            device=args.device,
            chunk_length_s=args.chunk_length_s,
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        logger.debug("模型加载完成")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        raise

    audio_paths = _collect_audio_paths(args.audio_dir, args.max_samples)
    total = len(audio_paths)
    if args.max_samples > 0 and total >= args.max_samples:
        logger.debug(
            f"已达到最大标注数量上限 {args.max_samples}（本批共 {total} 个文件）"
        )

    for idx, audio_path in enumerate(audio_paths, start=1):
        root = os.path.dirname(audio_path)
        rel_dir = os.path.relpath(root, args.audio_dir)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        # 在 labels_dir 下保持相同子目录结构
        out_dir = os.path.join(args.labels_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        label_path = os.path.join(out_dir, base + ".txt")
        _print_progress(idx, total)
        # 检查是否已存在非空标注文件（仅限文件，排除目录）
        if os.path.isfile(label_path):
            try:
                size = os.path.getsize(label_path)
            except Exception as e:
                logger.warning(f"检查标注文件大小失败: {label_path}, 错误: {e}")
                size = 0
            if size and size > 0:
                logger.debug(f"跳过已存在标注: {label_path} | {audio_path}")
                continue
        try:
            result = asr(audio_path)
            # 置信度过滤
            # 处理可能为 None 的置信度值
            cr_val = result.get("compression_ratio")
            cr = float(cr_val) if cr_val is not None else 0.0
            lp_val = result.get("avg_logprob")
            if lp_val is None:
                lp_val = result.get("logprob")
            lp = float(lp_val) if lp_val is not None else 0.0
            try:
                if (
                    cr > args.compression_ratio_threshold
                    or lp < args.logprob_threshold
                ):
                    logger.debug(
                        f"低置信度，跳过: cr={cr:.2f}, lp={lp:.2f}, 文件: {audio_path}"
                    )
                    continue
            except TypeError as e:
                logger.debug(
                    f"置信度比较出错，跳过过滤: cr={cr}, lp={lp}, error={e}, 文件: {audio_path}"
                )
                logger.debug(f"错误堆栈:\n{traceback.format_exc()}")
            # 继续进行文本提取
            text = result.get("text", "").strip()
        except Exception as e:
            logger.error(f"标注失败: {audio_path}, 错误: {e}")
            logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
            continue

        try:
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            logger.debug(f"写入标注: {label_path}")
        except Exception as e:
            logger.error(f"写入标注文件失败: {label_path}, 错误: {e}")
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")
            continue

    if total > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
    logger.debug(f"标注完成，本批共 {total} 个音频文件")


if __name__ == "__main__":
    main()
