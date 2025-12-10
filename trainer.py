import argparse
import csv
import numpy as np
import evaluate
import jiwer
from transformers import (
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    WhisperProcessor, WhisperForConditionalGeneration, get_linear_schedule_with_warmup
)
import torch
import multiprocessing  # 新增
import peft  # 新增
from peft import LoraConfig, get_peft_model
import soundfile as sf
from dataclasses import dataclass
from typing import List, Dict, Any
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from torch.optim import AdamW

# 自定义 Trainer，跳过移除 unused columns
class CustomTrainer(Trainer):
    def _remove_unused_columns(self, dataset, description=''):
        return dataset
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels', None)
        inputs.pop('input_ids', None)
        if labels is not None:
            outputs = model(**inputs, labels=labels)
        else:
            outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        return (loss, outputs) if return_outputs else loss
    def _prepare_inputs(self, data):
        model_inputs = super()._prepare_inputs(data)
        model_inputs.pop('input_ids', None)
        return model_inputs

# ModelWrapper 丢弃多余输入
class ModelWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, *args, **kwargs):
        kwargs.pop('input_ids', None)
        return self.base_model(*args, **kwargs)
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

@dataclass
class SpeechSeq2SeqCollator:
    processor: WhisperProcessor
    padding: bool = True
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [f['input_features'] for f in features]
        batch = self.processor.feature_extractor.pad(
            {'input_features': input_features}, padding=self.padding,
            return_tensors='pt', return_attention_mask=True
        )
        labels = [f['labels'] for f in features]
        label_batch = self.processor.tokenizer.pad(
            {'input_ids': labels}, padding=self.padding, return_tensors='pt'
        )
        label_ids = label_batch['input_ids'].masked_fill(
            label_batch['input_ids'] == self.processor.tokenizer.pad_token_id,
            -100
        )
        batch['labels'] = label_ids
        return batch

class WhisperDataset(Dataset):
    """
    PyTorch Dataset that reads manifest CSV and loads audio & text.
    CSV must have columns: audio_filepath,text
    """
    def __init__(self, manifest_path, sampling_rate=16000):
        self.items = []
        self.sr = sampling_rate
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append((row['audio_filepath'], row['text']))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, text = self.items[idx]
        data, sr = sf.read(path, dtype='float32')
        if sr != self.sr:
            data = np.interp(
                np.linspace(0, len(data), int(len(data) * self.sr / sr)),
                np.arange(len(data)), data
            )
        return {'wav': data, 'text': text}


def main():
    parser = argparse.ArgumentParser(description='LoRA 微调 Trainer')
    parser.add_argument('--train_manifest', type=str, required=True)
    parser.add_argument('--eval_manifest', type=str, required=True)
    parser.add_argument('--test_manifest', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, default='openai/whisper-large-v3-turbo')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no','fp16','bf16'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_modules', type=str, default='q_proj,v_proj')
    args = parser.parse_args()

    # 加载处理器与模型
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    modules = [m.strip() for m in args.target_modules.split(',') if m.strip()]
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                             target_modules=modules, lora_dropout=args.lora_dropout,
                             bias='none', task_type='SEQ_2_SEQ_LM')
    model = get_peft_model(model, lora_config)
    # Patch base_model.forward to drop unexpected args
    orig_base_forward = model.base_model.forward
    def patched_base_forward(self, *args, **kwargs):
        kwargs.pop('input_ids', None)
        kwargs.pop('inputs_embeds', None)
        return orig_base_forward(*args, **kwargs)
    model.base_model.forward = patched_base_forward.__get__(model.base_model, type(model.base_model))
    # 再包装模型以移除多余输入
    model = ModelWrapper(model)  # 继续使用 wrapper 去除多余输入
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 固定随机种子
    torch.manual_seed(args.seed)

    # 构建 PyTorch Datasets
    train_ds = WhisperDataset(args.train_manifest, sampling_rate=processor.feature_extractor.sampling_rate)
    val_ds = WhisperDataset(args.eval_manifest, sampling_rate=processor.feature_extractor.sampling_rate)

    # 定义 Whisper 专用 Collator
    @dataclass
    class WhisperDataCollator:
        processor: WhisperProcessor
        sampling_rate: int = 16000
        max_frames: int = 3000  # 最大帧数，超出则截断
        padding: str = 'longest'

        def __call__(self, features: list) -> dict:
            # 将 wav 列表转为 torch.Tensor 列表以支持 processor
            wavs = [torch.tensor(f['wav'], dtype=torch.float32) for f in features]
            texts = [f['text'] for f in features]
            # 特征提取与动态 padding
            audio_batch = self.processor.feature_extractor(
                wavs,
                sampling_rate=self.sampling_rate,
                padding=self.padding,
                return_tensors='pt', return_attention_mask=True
            )
            # 标签编码与 mask
            with self.processor.as_target_processor():
                label_batch = self.processor.tokenizer(
                    texts,
                    padding=self.padding,
                    truncation=True,
                    return_tensors='pt'
                )
            labels = label_batch.input_ids.masked_fill(
                label_batch.input_ids == self.processor.tokenizer.pad_token_id, -100
            )
            return {
                'input_features': audio_batch.input_features,
                'attention_mask': audio_batch.attention_mask,
                'labels': labels
            }
    # 构造 DataLoader
    collate_fn = WhisperDataCollator(
        processor=processor,
        sampling_rate=processor.feature_extractor.sampling_rate,
        max_frames=processor.feature_extractor.chunk_length * processor.feature_extractor.sampling_rate // processor.feature_extractor.hop_length
    )
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size,
                              shuffle=True, num_workers=2, collate_fn=collate_fn)
    eval_loader = DataLoader(val_ds, batch_size=args.eval_batch_size,
                             shuffle=False, num_workers=2, collate_fn=collate_fn)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_update_steps = (len(train_loader) * args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=int(0.1 * num_update_steps), num_training_steps=num_update_steps)

    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    model.train()

    for epoch in range(args.num_train_epochs):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            # 构建 inputs，并移除多余参数
            inputs = {'input_features': batch['input_features'], 'labels': batch['labels']}
            inputs.pop('input_ids', None)
            inputs.pop('inputs_embeds', None)
            outputs = model(**inputs)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}")

        # 验证
        model.eval()
        val_loss = 0
        for batch in eval_loader:
            with torch.no_grad():
                # 移除多余参数
                eval_inputs = {'input_features': batch['input_features'], 'labels': batch['labels']}
                eval_inputs.pop('input_ids', None)
                eval_inputs.pop('inputs_embeds', None)
                outputs = model(**eval_inputs)
                val_loss += outputs.loss.item()
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(eval_loader):.4f}")
        model.train()

    # 保存 LoRA adapter 和模型
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == '__main__':
    main()
