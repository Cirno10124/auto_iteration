#!/usr/bin/env python3
import argparse
import logging
import os
import re
import traceback

import jiwer
import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def find_model_path(model_dir, logger):
    """查找可用的模型路径，优先级：merged_model > best_model > checkpoint > model_dir"""
    merged_model_dir = os.path.join(model_dir, "merged_model")
    best_model_dir = os.path.join(model_dir, "best_model")
    checkpoint_dir = os.path.join(model_dir, "checkpoint")

    if os.path.exists(merged_model_dir):
        logger.info(f"找到合并后的模型: {merged_model_dir}")
        return merged_model_dir, None, True

    if os.path.exists(best_model_dir):
        logger.info(f"找到最佳模型: {best_model_dir}")
        return best_model_dir, model_dir, False

    if os.path.exists(checkpoint_dir):
        logger.info(f"找到检查点模型: {checkpoint_dir}")
        return checkpoint_dir, model_dir, False

    if os.path.exists(model_dir):
        logger.info(f"使用模型目录: {model_dir}")
        return model_dir, None, False

    return None, None, False


def load_model_with_peft(model_path, base_model_path, device, logger):
    """使用 PEFT 加载模型，如果需要则合并权重"""
    try:
        # 检查是否是 PEFT 模型
        adapter_config_path = os.path.join(
            model_path, "adapter_config.json"
        )
        is_peft = os.path.exists(adapter_config_path)

        if is_peft:
            logger.info("检测到 LoRA 权重，使用 PEFT 加载...")
            if base_model_path is None:
                logger.error("LoRA 模型需要基础模型路径，但未提供")
                return None, None

            # 加载基础模型
            try:
                base_model = (
                    WhisperForConditionalGeneration.from_pretrained(
                        base_model_path
                    )
                )
                logger.info(f"基础模型加载成功: {base_model_path}")
            except Exception as e:
                logger.error(f"加载基础模型失败: {e}")
                return None, None

            # 加载 LoRA 权重
            try:
                model = PeftModel.from_pretrained(base_model, model_path)
                logger.info(f"LoRA 权重加载成功: {model_path}")
            except Exception as e:
                logger.error(f"加载 LoRA 权重失败: {e}")
                return None, None

            # 合并权重
            try:
                logger.info("合并 LoRA 权重到基础模型...")
                merged_model = model.merge_and_unload()
                logger.info("权重合并成功")
                model = merged_model
            except Exception as e:
                logger.error(f"合并权重失败: {e}")
                logger.error(f"错误堆栈:\n{traceback.format_exc()}")
                return None, None
        else:
            # 直接加载完整模型
            logger.info("加载完整模型...")
            try:
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_path
                )
                logger.info("模型加载成功")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                logger.error(f"错误堆栈:\n{traceback.format_exc()}")
                return None, None

        # 加载处理器
        try:
            processor = WhisperProcessor.from_pretrained(model_path)
            logger.info("处理器加载成功")
        except Exception as e:
            logger.warning("从模型目录加载处理器失败，尝试从基础模型加载...")
            if base_model_path:
                try:
                    processor = WhisperProcessor.from_pretrained(
                        base_model_path
                    )
                    logger.info("从基础模型加载处理器成功")
                except Exception as e2:
                    logger.error(f"从基础模型加载处理器也失败: {e2}")
                    return None, None
            else:
                logger.error(f"无法加载处理器: {e}")
                return None, None

        model = model.to(device)
        model.eval()
        return model, processor

    except Exception as e:
        logger.error(f"加载模型时出现未预期错误: {e}")
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        return None, None


def prepare_audio(
    path, processor, device, sampling_rate=16000, logger=None
):
    """准备音频数据，带错误处理"""
    try:
        if not os.path.exists(path):
            error_msg = f"音频文件不存在: {path}"
            if logger:
                logger.error(error_msg)
            else:
                print(f"错误: {error_msg}")
            return None, None

        data, sr = sf.read(path, dtype="float32")

        if len(data) == 0:
            error_msg = f"音频文件为空: {path}"
            if logger:
                logger.warning(error_msg)
            else:
                print(f"警告: {error_msg}")
            return None, None

        if sr != sampling_rate:
            if logger:
                logger.debug(
                    f"重采样音频: {sr}Hz -> {sampling_rate}Hz ({path})"
                )
            data = np.interp(
                np.linspace(
                    0, len(data), int(len(data) * sampling_rate / sr)
                ),
                np.arange(len(data)),
                data,
            )

        inputs = processor(
            data,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return inputs.input_features.to(device), inputs.attention_mask.to(
            device
        )

    except Exception as e:
        error_msg = f"处理音频文件失败 {path}: {e}"
        if logger:
            logger.error(error_msg)
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        else:
            print(f"错误: {error_msg}")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 Whisper 微调模型")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="微调后模型目录"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="基础模型路径（LoRA 模型需要）",
    )
    parser.add_argument(
        "--test_manifest",
        type=str,
        required=True,
        help="测试数据 CSV 清单，含 audio_filepath,text",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="eval_results.txt",
        help="评估结果输出文件",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批量大小，仅支持逐条或小批量",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cer",
        choices=["wer", "cer"],
        help="评估指标",
    )
    parser.add_argument(
        "--language", type=str, default="zh", help="识别语言，例如 zh 或 en"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="任务类型：转写或翻译",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="最大生成 token 数"
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=2,
        help="禁止重复 n-gram 大小",
    )
    args = parser.parse_args()

    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("模型评估开始")
    logger.info("=" * 60)
    logger.info(f"模型目录: {args.model_dir}")
    logger.info(f"测试清单: {args.test_manifest}")
    logger.info(f"评估指标: {args.metric.upper()}")
    logger.info("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(args.test_manifest):
        logger.error(f"测试清单文件不存在: {args.test_manifest}")
        exit(1)

    if not os.path.exists(args.model_dir):
        logger.error(f"模型目录不存在: {args.model_dir}")
        exit(1)

    # 优先使用上一轮最佳 LoRA 模型（best_model），如果不存在再依次查找
    best_model_dir = os.path.join(args.model_dir, "best_model")
    if os.path.isdir(best_model_dir):
        logger.info(f"检测到上一轮最佳模型: {best_model_dir}，将用于评估")
        # 保存原始 model_dir 作为 base_model_path
        args.base_model_path = args.model_dir
        args.model_dir = best_model_dir

    # 查找模型路径
    model_path, base_model_path, is_merged = find_model_path(
        args.model_dir, logger
    )
    if model_path is None:
        logger.error("未找到可用的模型")
        exit(1)

    # 如果未指定基础模型路径，尝试从配置推断
    if base_model_path is None and not is_merged:
        # 尝试从训练配置读取
        config_path = os.path.join(args.model_dir, "training_config.json")
        if os.path.exists(config_path):
            try:
                import json

                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "model_name_or_path" in config:
                        base_model_path = config["model_name_or_path"]
                        logger.info(f"从配置文件读取基础模型路径: {base_model_path}")
            except Exception as e:
                logger.warning(f"读取配置文件失败: {e}")

    # 如果仍然没有基础模型路径，使用命令行参数
    if base_model_path is None:
        base_model_path = args.base_model_path

    # 加载模型和处理器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    model, processor = load_model_with_peft(
        model_path, base_model_path, device, logger
    )
    if model is None or processor is None:
        logger.error("模型或处理器加载失败，退出")
        exit(1)

    # 加载测试集
    try:
        logger.info("加载测试数据集...")
        ds = load_dataset("csv", data_files={"test": args.test_manifest})[
            "test"
        ]
        logger.info(f"测试集大小: {len(ds)}")
    except Exception as e:
        logger.error(f"加载测试数据集失败: {e}")
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        exit(1)

    if len(ds) == 0:
        logger.error("测试集为空")
        exit(1)

    # 评估
    predictions = []
    references = []
    failed_count = 0

    logger.info("开始评估...")
    for idx, example in enumerate(ds):
        try:
            audio_fp = example.get("audio_filepath") or example.get(
                "audio_file"
            )
            if audio_fp is None:
                logger.warning(f"样本 {idx+1}: 缺少音频文件路径，跳过")
                failed_count += 1
                continue

            ref = example.get("text", "").strip()
            if not ref:
                logger.warning(f"样本 {idx+1}: 参考文本为空，跳过")
                failed_count += 1
                continue

            # 准备音频
            input_feats, attn_mask = prepare_audio(
                audio_fp, processor, device, logger=logger
            )
            if input_feats is None or attn_mask is None:
                logger.warning(f"样本 {idx+1}: 音频处理失败，跳过")
                failed_count += 1
                continue

            # 生成预测
            try:
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_features=input_feats,
                        attention_mask=attn_mask,
                        forced_decoder_ids=processor.get_decoder_prompt_ids(
                            language=args.language, task=args.task
                        ),
                        max_new_tokens=args.max_new_tokens,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )
                pred = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
                # 清理特殊 token
                pred = re.sub(r"<\|.*?\|>", "", pred).strip()
            except Exception as e:
                logger.error(f"样本 {idx+1}: 生成预测失败: {e}")
                logger.error(f"错误堆栈:\n{traceback.format_exc()}")
                failed_count += 1
                continue

            predictions.append(pred)
            references.append(ref)

            if (idx + 1) % 10 == 0:
                logger.info(f"已处理 {idx+1}/{len(ds)} 个样本")

            # 打印样本（前5个）
            if len(predictions) <= 5:
                logger.info(f"样本 {len(predictions)}:")
                logger.info(f"  参考: {ref}")
                logger.info(f"  预测: {pred}")

        except Exception as e:
            logger.error(f"样本 {idx+1}: 处理时出现未预期错误: {e}")
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")
            failed_count += 1
            continue

    logger.info(f"评估完成，成功: {len(predictions)}, 失败: {failed_count}")

    if len(predictions) == 0:
        logger.error("没有成功评估的样本，无法计算指标")
        exit(1)

    # 计算指标
    try:
        if args.metric.lower() == "wer":
            score = jiwer.wer(references, predictions)
        else:
            score = jiwer.cer(references, predictions)

        logger.info("=" * 60)
        logger.info(f"评估结果: {args.metric.upper()} = {score:.4f}")
        logger.info(f"成功评估样本数: {len(predictions)}/{len(ds)}")
        if failed_count > 0:
            logger.warning(f"失败样本数: {failed_count}")
        logger.info("=" * 60)

        # 保存结果
        try:
            os.makedirs(
                (
                    os.path.dirname(args.output_file)
                    if os.path.dirname(args.output_file)
                    else "."
                ),
                exist_ok=True,
            )
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(f"{args.metric.upper()}: {score:.4f}\n")
                f.write(f"成功评估样本数: {len(predictions)}/{len(ds)}\n")
                if failed_count > 0:
                    f.write(f"失败样本数: {failed_count}\n")
                f.write("\n详细结果:\n")
                for i, (ref, pred) in enumerate(
                    zip(references, predictions)
                ):
                    f.write(f"\n样本 {i+1}:\n")
                    f.write(f"参考: {ref}\n")
                    f.write(f"预测: {pred}\n")
            logger.info(f"评估结果已保存到: {args.output_file}")
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")

    except Exception as e:
        logger.error(f"计算评估指标失败: {e}")
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        exit(1)
