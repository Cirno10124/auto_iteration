#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import jiwer
import peft
import soundfile as sf
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import disable_caching, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)

try:
    import opencc

    _opencc_converter = opencc.OpenCC("t2s")
except Exception:
    _opencc_converter = None

# Monkey-patch PeftModel to drop input_ids and inputs_embeds
_orig_peft_forward = peft.PeftModel.forward


def _patched_peft_forward(self, *args, **kwargs):
    kwargs.pop("input_ids", None)
    kwargs.pop("inputs_embeds", None)
    return _orig_peft_forward(self, *args, **kwargs)


peft.PeftModel.forward = _patched_peft_forward


def normalize_zh_text(text: str) -> str:
    """将文本统一为简体中文，并去除标点和空白，仅保留中英文和数字。"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""

    if _opencc_converter is not None:
        try:
            text = _opencc_converter.convert(text)
        except Exception:
            pass

    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", text)
    return text


def normalize_manifest_text(example: Dict[str, Any]) -> Dict[str, Any]:
    """仅修改入内存样本文本，不改动源 manifest 文件。"""
    if "text" in example:
        example["text"] = normalize_zh_text(example.get("text", ""))
    if "transcript" in example:
        example["transcript"] = normalize_zh_text(
            example.get("transcript", "")
        )
    return example


def has_non_empty_text(example: Dict[str, Any]) -> bool:
    """过滤规范化后为空文本的样本。"""
    text = (example.get("text") or "").strip()
    transcript = (example.get("transcript") or "").strip()
    return bool(text or transcript)


def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# Model wrapper to drop unexpected input_ids argument
class ModelWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, *args, **kwargs):
        kwargs.pop("input_ids", None)
        return self.base_model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


@dataclass
class SpeechSeq2SeqCollator:
    """Custom collator for speech-to-seq2seq models like Whisper"""

    processor: WhisperProcessor
    padding: Any = True

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [f["input_features"] for f in features]
        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features},
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True,
        )
        labels = [f["labels"] for f in features]
        label_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=self.padding,
            return_tensors="pt",
        )
        label_ids = label_batch["input_ids"].masked_fill(
            label_batch["input_ids"]
            == self.processor.tokenizer.pad_token_id,
            -100,
        )
        batch["labels"] = label_ids
        return batch


def prepare_dataset(batch, processor, sampling_rate=16000):
    """准备数据集，只支持 Whisper"""
    audio_fp = batch.get("audio_filepath") or batch.get("audio_file")
    batch["is_valid"] = True
    batch["skip_reason"] = ""
    if not audio_fp or not os.path.exists(audio_fp):
        batch["is_valid"] = False
        batch["skip_reason"] = f"audio not found: {audio_fp}"
        batch["input_features"] = []
        batch["labels"] = []
        return batch

    try:
        import numpy as np

        data, sr = sf.read(audio_fp, dtype="float32")
        # 多通道转单通道，避免后续特征提取维度异常
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if len(data) == 0 or sr <= 0:
            raise ValueError(
                f"empty audio or invalid sample rate: len={len(data)}, sr={sr}"
            )
        if sr != sampling_rate:
            data = np.interp(
                np.linspace(0, len(data), int(len(data) * sampling_rate / sr)),
                np.arange(len(data)),
                data,
            ).astype("float32")

        processed = processor(data, sampling_rate=sampling_rate)
        input_features = np.asarray(processed.input_features[0], dtype=np.float32)
        if input_features.ndim != 2:
            raise ValueError(
                f"processor returned invalid input_features shape: {input_features.shape}"
            )
        # 保持二维 mel 特征，Whisper 期望 [feature_bins, frames]（通常 80x3000）
        batch["input_features"] = input_features.tolist()
        # 使用 keyword 传参，兼容 transformers tokenizer 要求；并兼容可能的列名 transcript
        text = (batch.get("text") or batch.get("transcript") or "").strip()
        if not text:
            batch["is_valid"] = False
            batch["skip_reason"] = "empty normalized text"
            batch["input_features"] = [[0.0]]
            batch["labels"] = []
            return batch
        labels = processor.tokenizer(text=text).input_ids
        eos_id = processor.tokenizer.eos_token_id
        if labels and labels[-1] != eos_id:
            labels.append(eos_id)
        batch["labels"] = [int(x) for x in labels]
        return batch
    except Exception as e:
        batch["is_valid"] = False
        batch["skip_reason"] = f"{type(e).__name__}: {e}"
        # 占位值也保持同类型：list[list[float]]，与有效样本一致
        batch["input_features"] = [[0.0]]
        batch["labels"] = []
        return batch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    step,
    best_metric,
    output_dir,
    accelerator,
    logger,
):
    """保存检查点"""
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存模型状态（必须保存 PeftModel 适配器，否则恢复时会是 0 可训练参数）
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
    else:
        unwrapped_model = model
    # 若被 ModelWrapper 包装，保存内部的 PeftModel，否则保存的可能是错误结构
    to_save = (
        unwrapped_model.base_model
        if isinstance(unwrapped_model, ModelWrapper)
        else unwrapped_model
    )
    to_save.save_pretrained(checkpoint_dir)

    # 保存优化器和调度器状态（需要转换为可序列化的格式）
    # 注意：优化器和调度器的状态字典可能包含不可序列化的对象
    # 这里我们只保存必要的状态信息
    checkpoint_state = {
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
    }
    checkpoint_path = os.path.join(checkpoint_dir, "training_state.json")
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_state, f, indent=2)

    # 保存优化器和调度器状态到单独的文件
    if accelerator is not None:
        torch.save(
            optimizer.state_dict(),
            os.path.join(checkpoint_dir, "optimizer.pt"),
        )
        torch.save(
            scheduler.state_dict(),
            os.path.join(checkpoint_dir, "scheduler.pt"),
        )

    logger.info(f"检查点已保存到 {checkpoint_dir} (epoch {epoch}, step {step})")


def load_checkpoint(output_dir, logger):
    """加载检查点"""
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        return None, None, None, 0, 0, None

    checkpoint_state_path = os.path.join(
        checkpoint_dir, "training_state.json"
    )
    if not os.path.exists(checkpoint_state_path):
        return None, None, None, 0, 0, None

    try:
        with open(checkpoint_state_path, "r") as f:
            checkpoint_state = json.load(f)

        # 加载优化器和调度器状态
        opt_state_path = os.path.join(checkpoint_dir, "optimizer.pt")
        sched_state_path = os.path.join(checkpoint_dir, "scheduler.pt")
        opt_state = None
        sched_state = None
        if os.path.exists(opt_state_path):
            opt_state = torch.load(opt_state_path, map_location="cpu")
        if os.path.exists(sched_state_path):
            sched_state = torch.load(sched_state_path, map_location="cpu")

        logger.info(
            f"从检查点恢复训练: epoch {checkpoint_state['epoch']}, step {checkpoint_state['step']}"
        )
        return (
            checkpoint_dir,
            opt_state,
            sched_state,
            checkpoint_state["epoch"],
            checkpoint_state["step"],
            checkpoint_state.get("best_metric", None),
        )
    except Exception as e:
        logger.warning(f"加载检查点失败: {e}")
        return None, None, None, 0, 0, None


def merge_and_save_model(peft_model, base_model_path, output_dir, logger):
    """合并 LoRA 权重到基础模型并保存"""
    try:
        logger.info("开始合并 LoRA 权重到基础模型...")
        # 加载基础模型
        # _base_model = WhisperForConditionalGeneration.from_pretrained(
        #     base_model_path
        # )
        # 合并权重
        merged_model = peft_model.merge_and_unload()
        # 保存合并后的模型
        merged_output_dir = os.path.join(output_dir, "merged_model")
        os.makedirs(merged_output_dir, exist_ok=True)
        merged_model.save_pretrained(merged_output_dir)
        logger.info(f"合并后的模型已保存到 {merged_output_dir}")
        return merged_output_dir
    except Exception as e:
        logger.error(f"合并模型失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Whisper LoRA 微调训练脚本")
    parser.add_argument(
        "--train_manifest",
        type=str,
        required=True,
        help="训练数据 CSV/JSON 列表",
    )
    parser.add_argument(
        "--eval_manifest",
        type=str,
        required=True,
        help="验证数据 CSV/JSON 列表",
    )
    parser.add_argument(
        "--test_manifest",
        type=str,
        required=True,
        help="测试数据 CSV/JSON 列表",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Whisper 模型路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="模型输出目录"
    )
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--eval_metric", type=str, default="cer", help="评估指标: cer 或 wer"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--language", type=str, default="zh", help="语言代码")
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,v_proj",
        help="LoRA 目标模块",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.2)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=None,
        help="每N步保存检查点，None表示禁用",
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=float,
        default=0.25,
        help="每N个epoch保存检查点",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=3, help="早停耐心值"
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.001,
        help="早停阈值",
    )
    parser.add_argument(
        "--save_merged_model", action="store_true", help="保存合并后的完整模型"
    )
    parser.add_argument(
        "--no_resume_from_checkpoint",
        action="store_true",
        help="不恢复检查点，从头训练（重写标签/重建清单后建议使用）",
    )
    parser.add_argument(
        "--enable_hf_datasets_cache",
        action="store_true",
        help="启用 HuggingFace datasets 持久缓存（默认关闭以避免缓存膨胀）",
    )
    args = parser.parse_args()

    logger = setup_logging()
    if not args.enable_hf_datasets_cache:
        disable_caching()
        logger.info(
            "已关闭 HuggingFace datasets 持久缓存（如需启用可加 --enable_hf_datasets_cache）"
        )

    logger.info("=" * 60)
    logger.info("Whisper LoRA 微调训练开始")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model_name_or_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"训练轮数: {args.num_train_epochs}")
    logger.info(f"批次大小: {args.train_batch_size}")
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据集
    data_files = {
        "train": args.train_manifest,
        "validation": args.eval_manifest,
        "test": args.test_manifest,
    }
    datasets = load_dataset("csv", data_files=data_files)
    datasets = datasets.map(
        normalize_manifest_text, num_proc=1, load_from_cache_file=False
    )
    before_text_filter_sizes = {k: len(v) for k, v in datasets.items()}
    datasets = datasets.filter(
        has_non_empty_text, num_proc=1, load_from_cache_file=False
    )
    after_text_filter_sizes = {k: len(v) for k, v in datasets.items()}
    for split in before_text_filter_sizes:
        dropped = before_text_filter_sizes[split] - after_text_filter_sizes[split]
        if dropped > 0:
            logger.warning(
                f"{split} 集过滤掉 {dropped} 条规范化后空文本样本"
            )

    # 加载 Whisper 处理器和模型
    try:
        processor = WhisperProcessor.from_pretrained(
            args.model_name_or_path
        )
    except ValueError as e:
        logger.warning(f"WhisperProcessor 加载失败 ({e})，尝试手动构造...")
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        fe = WhisperFeatureExtractor.from_pretrained(
            args.model_name_or_path
        )
        tok = WhisperTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=False
        )
        processor = WhisperProcessor(feature_extractor=fe, tokenizer=tok)

    # 检查是否有检查点（可通过 --no_resume_from_checkpoint 强制从头训练）
    if args.no_resume_from_checkpoint:
        checkpoint_dir, opt_state, sched_state, start_epoch, start_step, best_metric = (
            None,
            None,
            None,
            0,
            0,
            None,
        )
        logger.info("已指定 --no_resume_from_checkpoint，将从头训练，不加载检查点")
    else:
        (
            checkpoint_dir,
            opt_state,
            sched_state,
            start_epoch,
            start_step,
            best_metric,
        ) = load_checkpoint(args.output_dir, logger)

    if checkpoint_dir:
        # 从检查点加载模型
        model = PeftModel.from_pretrained(
            WhisperForConditionalGeneration.from_pretrained(
                args.model_name_or_path
            ),
            checkpoint_dir,
            is_trainable=True,
        )
        logger.info("从检查点加载模型")
    else:
        # 加载基础模型并应用 LoRA
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_name_or_path
        )
        if hasattr(model, "peft_config"):
            # 避免重复注入 LoRA，导致 multiple adapters 警告与训练行为混乱
            logger.warning(
                "检测到输入模型已包含 peft_config，跳过再次注入 LoRA 适配器，将直接继续训练现有适配器。"
            )
        else:
            modules = [
                m.strip() for m in args.target_modules.split(",") if m.strip()
            ]
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            model = get_peft_model(model, lora_config)
            logger.info("创建新的 LoRA 模型")

    if hasattr(model, "peft_config"):
        peft_adapters = list(getattr(model, "peft_config", {}).keys())
        logger.info(
            f"当前模型已加载 PEFT 适配器: {peft_adapters if peft_adapters else ['default']}"
        )

    # 补丁 base_model.forward
    orig_base_forward = model.base_model.forward

    def patched_base_forward(self, *args, **kwargs):
        kwargs.pop("input_ids", None)
        kwargs.pop("inputs_embeds", None)
        return orig_base_forward(*args, **kwargs)

    model.base_model.forward = patched_base_forward.__get__(
        model.base_model, type(model.base_model)
    )

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"可训练参数: {trainable_params}/{total_params} ({100*trainable_params/total_params:.2f}%)"
    )
    if trainable_params == 0:
        logger.error(
            "可训练参数为 0，无法训练。若从检查点恢复，可能是检查点损坏或未正确保存 LoRA 适配器；"
            "请使用 --no_resume_from_checkpoint 从头训练，或检查 output_dir/checkpoint 下是否有 adapter_config.json 与 adapter_model.safetensors。"
        )
        raise RuntimeError("可训练参数为 0，训练已中止")

    model = ModelWrapper(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("已启用梯度检查点")

    # 数据预处理
    source_columns = set()
    for split in datasets.keys():
        source_columns.update(datasets[split].column_names)
    removable_columns = [
        c
        for c in ["audio_filepath", "audio_file", "text", "transcript"]
        if c in source_columns
    ]
    datasets = datasets.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=removable_columns,
        num_proc=1,
        load_from_cache_file=False,
    )
    if "is_valid" in datasets["train"].column_names:
        before_sizes = {k: len(v) for k, v in datasets.items()}
        datasets = datasets.filter(
            lambda b: b["is_valid"], num_proc=1, load_from_cache_file=False
        )
        after_sizes = {k: len(v) for k, v in datasets.items()}
        for split in before_sizes:
            dropped = before_sizes[split] - after_sizes[split]
            if dropped > 0:
                logger.warning(
                    f"{split} 集过滤掉 {dropped} 条无法读取/无效音频样本"
                )
        extra_cols = [
            c
            for c in ["is_valid", "skip_reason"]
            if c in datasets["train"].column_names
        ]
        if extra_cols:
            datasets = datasets.remove_columns(extra_cols)

    data_collator = SpeechSeq2SeqCollator(
        processor=processor, padding=True
    )

    # 初始化 Accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_loader = DataLoader(
        datasets["validation"],
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # 计算学习率调度器参数
    num_update_steps_per_epoch = (
        len(train_loader) // args.gradient_accumulation_steps
    )
    max_train_steps = num_update_steps_per_epoch * args.num_train_epochs
    num_warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else int(max_train_steps * 0.1)
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, max_train_steps
    )

    # 从检查点恢复优化器和调度器状态
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
        logger.info("已恢复优化器状态")
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)
        logger.info("已恢复调度器状态")

    torch.cuda.empty_cache()
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    model.train()

    # 早停相关变量
    best_metric_value = (
        best_metric
        if best_metric is not None
        else (
            float("inf")
            if args.eval_metric.lower() == "cer"
            else float("-inf")
        )
    )
    patience_counter = 0
    global_step = start_step
    first_no_grad_loss_logged = False

    # 训练循环
    for epoch in range(start_epoch, args.num_train_epochs):
        total_loss = 0.0
        model.train()

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}"
        )
        for step, batch in enumerate(progress_bar):
            if epoch == start_epoch and step < start_step:
                continue  # 跳过已训练的步骤

            inputs = {
                "input_features": batch["input_features"],
                "labels": batch["labels"],
            }
            # 跳过全为 padding 的 batch（labels 全为 -100），否则 loss 无 grad_fn 会报错
            labels = batch["labels"]
            if (labels == -100).all():
                logger.warning(
                    f"Epoch {epoch+1} step {step}: 跳过全为 padding 的 batch (labels 全为 -100)"
                )
                continue
            outputs = model(**inputs)
            loss = outputs.loss
            if not loss.requires_grad:
                if not first_no_grad_loss_logged:
                    first_no_grad_loss_logged = True
                    trainable_tensors = sum(
                        1 for p in model.parameters() if p.requires_grad
                    )
                    logger.warning(
                        "梯度诊断 | torch.is_grad_enabled=%s | loss.grad_fn=%s | "
                        "trainable_tensors=%s",
                        torch.is_grad_enabled(),
                        type(loss.grad_fn).__name__
                        if loss.grad_fn is not None
                        else "None",
                        trainable_tensors,
                    )
                logger.warning(
                    f"Epoch {epoch+1} step {step}: loss 未参与计算图 (requires_grad=False)，跳过本 step 避免 backward 报错"
                )
                continue
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                torch.cuda.empty_cache()

            total_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # 检查点保存（按步数）
            if (
                args.checkpoint_steps
                and global_step > 0
                and global_step % args.checkpoint_steps == 0
            ):
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    best_metric_value,
                    args.output_dir,
                    accelerator,
                    logger,
                )

        avg_loss = total_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1}/{args.num_train_epochs}, 平均 Loss: {avg_loss:.4f}"
        )

        # 验证集评估
        model.eval()
        torch.cuda.empty_cache()
        all_preds, all_labels = [], []

        for batch in tqdm(eval_loader, desc="评估中"):
            with torch.no_grad():
                gen_model = (
                    model.base_model
                    if hasattr(model, "base_model")
                    else model
                )
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=args.language, task=args.task
                )
                mask = batch.get("attention_mask")
                generated_ids = gen_model.generate(
                    input_features=batch["input_features"],
                    attention_mask=mask,
                    forced_decoder_ids=forced_decoder_ids,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    early_stopping=True,
                    max_new_tokens=args.max_new_tokens,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                raw_preds = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                preds = [
                    re.sub(r"<\|.*?\|>", "", p).strip() for p in raw_preds
                ]

                labels = batch["labels"]
                labels[labels == -100] = processor.tokenizer.pad_token_id
                refs = processor.batch_decode(labels, group_tokens=False)
                refs_clean = [
                    re.sub(r"<\|.*?\|>", "", r).strip() for r in refs
                ]
                all_preds.extend([normalize_zh_text(p) for p in preds])
                all_labels.extend(
                    [normalize_zh_text(r) for r in refs_clean]
                )

        # 计算评估指标
        if args.eval_metric.lower() == "wer":
            metric_value = jiwer.wer(all_labels, all_preds)
            is_better = metric_value < best_metric_value
        else:
            metric_value = jiwer.cer(all_labels, all_preds)
            is_better = metric_value < best_metric_value

        logger.info(
            f"Epoch {epoch+1}/{args.num_train_epochs}, {args.eval_metric.upper()}: {metric_value:.4f}"
        )

        # 打印样本输出
        logger.info("样本验证输出:")
        for i in range(min(3, len(all_preds))):
            logger.info(f"  {i+1}. 预测: {all_preds[i]}")
            logger.info(f"     参考: {all_labels[i]}")

        # 最佳模型保存和早停检查
        if is_better:
            improvement = abs(best_metric_value - metric_value)
            if improvement >= args.early_stopping_threshold:
                best_metric_value = metric_value
                patience_counter = 0
                # 保存最佳模型
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                best_model_dir = os.path.join(
                    args.output_dir, "best_model"
                )
                os.makedirs(best_model_dir, exist_ok=True)
                unwrapped_model.save_pretrained(best_model_dir)
                processor.save_pretrained(best_model_dir)
                logger.info(
                    f"保存最佳模型到 {best_model_dir} ({args.eval_metric.upper()}: {best_metric_value:.4f})"
                )
            else:
                logger.info(
                    f"改善幅度 {improvement:.6f} 小于阈值 {args.early_stopping_threshold}，不更新最佳模型"
                )
        else:
            patience_counter += 1
            logger.info(
                f"验证指标未改善，耐心计数: {patience_counter}/{args.early_stopping_patience}"
            )

        # 早停检查
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"早停触发：验证指标连续 {args.early_stopping_patience} 轮未改善")
            break

        # 检查点保存（按epoch）
        if (
            args.checkpoint_epochs > 0
            and (epoch + 1) % int(1 / args.checkpoint_epochs) == 0
        ):
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                best_metric_value,
                args.output_dir,
                accelerator,
                logger,
            )

        model.train()

    # 训练完成，保存最终模型
    logger.info("训练完成，保存最终模型...")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info(f"最终模型已保存到 {args.output_dir}")
    # 为模型打标签并保存元数据
    try:
        metadata = {
            "model_name_or_path": args.model_name_or_path,
            "train_manifest": args.train_manifest,
            "eval_manifest": args.eval_manifest,
            "test_manifest": args.test_manifest,
            "num_train_epochs": args.num_train_epochs,
            "train_batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        metadata_path = os.path.join(
            args.output_dir, "model_metadata.json"
        )
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, ensure_ascii=False, indent=2)
        logger.info(f"模型元数据已保存到 {metadata_path}")
    except Exception as e:
        logger.warning(f"保存模型元数据失败: {e}")

    # 保存合并后的模型
    if args.save_merged_model:
        merge_and_save_model(
            unwrapped_model,
            args.model_name_or_path,
            args.output_dir,
            logger,
        )

    logger.info("=" * 60)
    logger.info("训练完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
