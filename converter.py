#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime


def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def backup_files(model_dir, files_to_backup, logger):
    """备份文件到备份目录"""
    import traceback

    backup_dir = os.path.join(model_dir, "backup")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = os.path.join(backup_dir, f"pre_conversion_{timestamp}")

    try:
        os.makedirs(backup_subdir, exist_ok=True)
        logger.info(f"创建备份目录: {backup_subdir}")

        backed_up_files = []
        for fname in files_to_backup:
            src_path = os.path.join(model_dir, fname)
            if os.path.exists(src_path):
                dst_path = os.path.join(backup_subdir, fname)
                shutil.copy2(src_path, dst_path)
                backed_up_files.append(fname)
                logger.info(f"已备份: {fname} -> {dst_path}")

        if backed_up_files:
            logger.info(
                f"备份完成，共备份 {len(backed_up_files)} 个文件到 {backup_subdir}"
            )
            return backup_subdir
        else:
            logger.info("没有需要备份的文件")
            return None

    except Exception as e:
        logger.error(f"备份文件失败: {e}")
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face Whisper model to ggml format"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="微调后模型目录，包含 pytorch_model.bin 或 model.safetensors",
    )
    parser.add_argument(
        "--hf_whisper_repo",
        type=str,
        default="../../3rd_party/whisper",
        help="Hugging Face Whisper 源码仓库路径，用于加载 mel_filters.npz 等资产",
    )
    parser.add_argument(
        "--whisper_cpp_repo",
        type=str,
        default="../../3rd_party/whisper.cpp",
        help="whisper.cpp 源码仓库路径，用于 locate 转换脚本",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=False,
        help="基预训练模型名称或路径，用于合并LoRA权重",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ggml_model",
        help="输出 GGML 文件目录",
    )
    parser.add_argument(
        "--use_f32",
        action="store_true",
        help="生成 f32 精度模型，默认使用 f16",
    )
    parser.add_argument(
        "--use_h5_to_ggml",
        action="store_true",
        help="使用 whisper.cpp 的 convert-h5-to-ggml.py 脚本",
    )
    args = parser.parse_args()

    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("模型转换开始")
    logger.info("=" * 60)
    logger.info(f"模型目录: {args.model_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 60)
    # 解析 repo 路径，兼容 whisper 或 whisper.cpp 目录名
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 验证 HF-whisper 仓库路径
    hf_repo = args.hf_whisper_repo
    if not os.path.exists(hf_repo):
        alt_hf = os.path.join(script_dir, args.hf_whisper_repo)
        if os.path.exists(alt_hf):
            hf_repo = alt_hf
    if not os.path.exists(hf_repo):
        logger.error(
            f"未找到 Hugging Face Whisper 仓库路径: {args.hf_whisper_repo} 或 {hf_repo}"
        )
        sys.exit(1)
    # 验证 whisper.cpp 仓库路径
    repo_dir = args.whisper_cpp_repo
    if not os.path.exists(repo_dir):
        alt_repo = os.path.join(script_dir, args.whisper_cpp_repo)
        if os.path.exists(alt_repo):
            repo_dir = alt_repo
    if not os.path.exists(repo_dir):
        logger.error(
            f"未找到 whisper.cpp 仓库路径: {args.whisper_cpp_repo} 或 {repo_dir}"
        )
        sys.exit(1)
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        alt_dir = os.path.join(script_dir, args.model_dir)
        if os.path.exists(alt_dir):
            model_dir = alt_dir

    if not os.path.exists(model_dir):
        logger.error(f"模型目录不存在: {args.model_dir}")
        sys.exit(1)

    # 备份需要删除或修改的文件
    files_to_backup = [
        "model.safetensors",
        "adapter_config.json",
        "pytorch_model.bin",
    ]
    backup_dir = backup_files(model_dir, files_to_backup, logger)

    # 合并 LoRA 权重到基础模型，并生成 pytorch_model.bin
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    base_name = (
        args.base_model_name_or_path or "openai/whisper-large-v3-turbo"
    )
    logger.info(f"合并 LoRA 权重到基础模型 {base_name} ...")
    try:
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_name
        )
        peft_model = PeftModel.from_pretrained(base_model, model_dir)
        merged_model = peft_model.merge_and_unload()
        # 关键：强制保存为单个 pytorch_model.bin（避免默认 safetensors / 分片导致后续找不到文件）
        merged_model.save_pretrained(
            model_dir, safe_serialization=False, max_shard_size="10GB"
        )
        model_bin = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_bin):
            # 兜底：某些环境仍可能保存成别名或分片，这里尽量定位并复制/合并为单文件
            cand = glob.glob(os.path.join(model_dir, "pytorch_model*.bin"))
            if cand:
                # 选最像主权重的那个
                cand.sort()
                shutil.copy2(cand[0], model_bin)
            else:
                # 最后兜底：直接 state_dict 导出
                import torch

                torch.save(merged_model.state_dict(), model_bin)
        if not os.path.exists(model_bin):
            raise FileNotFoundError(model_bin)
        logger.info(f"LoRA 合并完成，已生成 {model_bin}")
    except Exception as e:
        logger.error(f"合并 LoRA 权重失败: {e}")
        import traceback

        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        sys.exit(1)
    # 说明：merge_and_unload() 已完成合并。这里不再做“手动合并残留 LoRA”这类高风险操作，
    # 只做轻量自检：确保生成的权重文件可被加载（便于尽早暴露损坏/路径问题）。
    try:
        import torch

        _ = torch.load(model_bin, map_location="cpu")
    except Exception as e:
        logger.warning(
            f"合并后的权重文件无法加载（{model_bin}），后续转换很可能失败。错误: {e}"
        )

    # 选择转换脚本路径
    if args.use_h5_to_ggml:
        script = os.path.join(repo_dir, "models", "convert-h5-to-ggml.py")
        if not os.path.exists(script):
            logger.error(f"未找到 H5 转换脚本: {script}")
            sys.exit(1)
    else:
        script = os.path.join(repo_dir, "models", "convert-pt-to-ggml.py")
        if not os.path.exists(script):
            logger.error(f"未找到 PT 转换脚本: {script}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    # 构造命令行：将 hf_repo 作为第二参数传递给转换脚本
    if args.use_h5_to_ggml:
        cmd = [sys.executable, script, model_dir, hf_repo, args.output_dir]
    else:
        cmd = [sys.executable, script, model_bin, hf_repo, args.output_dir]
    if args.use_f32:
        cmd.append("f32")

    logger.info(f'运行转换命令: {" ".join(cmd)}')
    # 调用 whisper.cpp 转换脚本，捕获 stderr 以识别常见错误
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode == 0:
        logger.info(f"转换完成，GGML 模型输出至 {args.output_dir}")
        if backup_dir:
            logger.info(f"原始模型文件已备份到: {backup_dir}")
    else:
        out = result.stdout.decode(errors="ignore")
        err = result.stderr.decode(errors="ignore")
        logger.warning(
            f"convert-h5-to-ggml.py 脚本失败，错误代码 {result.returncode}，尝试回退到 convert-pt-to-ggml.py"
        )
        if out.strip():
            logger.debug(f"stdout:\n{out}")
        if err.strip():
            logger.debug(f"stderr:\n{err}")
        # 回退到 convert-pt-to-ggml.py
        fallback_script = os.path.join(
            repo_dir, "models", "convert-pt-to-ggml.py"
        )
        if not os.path.exists(fallback_script):
            logger.error(f"缺少回退脚本: {fallback_script}")
            sys.exit(1)
        if not os.path.exists(model_bin):
            logger.error(
                f"回退转换需要权重文件，但未找到 {model_bin}（请检查合并/保存步骤是否成功）"
            )
            sys.exit(1)
        fallback_cmd = [
            sys.executable,
            fallback_script,
            model_bin,
            hf_repo,
            args.output_dir,
        ]
        if args.use_f32:
            fallback_cmd.append("f32")
        logger.info(f"运行回退转换命令: {' '.join(fallback_cmd)}")
        fallback_result = subprocess.run(
            fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if fallback_result.returncode == 0:
            logger.info(f"回退转换完成，GGML 模型输出至 {args.output_dir}")
            if backup_dir:
                logger.info(f"原始模型文件已备份到: {backup_dir}")
        else:
            fout = fallback_result.stdout.decode(errors="ignore")
            ferr = fallback_result.stderr.decode(errors="ignore")
            logger.error(
                f"回退转换失败: return code {fallback_result.returncode}"
            )
            if fout.strip():
                logger.error(f"stdout:\n{fout}")
            if ferr.strip():
                logger.error(f"stderr:\n{ferr}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("模型转换完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
