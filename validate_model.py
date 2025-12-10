#!/usr/bin/env python3
import argparse
import csv
import torch
import soundfile as sf
import jiwer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import json
import time
import base64
import hashlib
import requests
import re
from opencc import OpenCC

# 文本预处理：简体化、数字转汉字、英文小写化、保留中英文及%
converter = OpenCC('t2s')
def preprocess_text(text):
    # 简体化
    text = converter.convert(text)
    # 英文小写化
    text = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), text)
    # 保留中英文、百分号与空格
    text = ''.join(ch for ch in text if re.match(r"[\u4e00-\u9fa5A-Za-z% ]", ch))
    # 去除空格
    text = text.replace(" ", "")
    return text.strip()

# 我将添加讯飞转录占位函数以支持后续实现

def transcribe_xunfei(audio_fp, config_path):
    """使用讯飞RaaS接口将音频转录为文本"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f).get('xunfei', {})
    app_id = cfg['app_id']
    api_key = cfg['api_key']
    host = cfg['host']
    upload_url = f"https://{host}{cfg['upload_url']}"
    result_url = f"https://{host}{cfg['result_url']}"
    timeout = cfg.get('timeout_seconds', 30)
    # 读取音频数据
    with open(audio_fp, 'rb') as fa:
        audio_data = fa.read()
    # 构造请求头
    cur_time = str(int(time.time()))
    # 默认识别参数，可按需修改
    param_dict = {'engine_type':'sms16k','aue':'raw'}
    param = base64.b64encode(json.dumps(param_dict).encode()).decode()
    checksum = hashlib.md5((api_key + cur_time + param).encode()).hexdigest()
    headers = {
        'Content-Type':'application/octet-stream',
        'X-Appid':app_id,
        'X-CurTime':cur_time,
        'X-Param':param,
        'X-CheckSum':checksum
    }
    # 上传音频
    resp = requests.post(upload_url, headers=headers, data=audio_data, timeout=timeout)
    rj = resp.json()
    if rj.get('code') != '0':
        return ''
    task_id = rj.get('data')
    # 轮询结果
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(2)
        # 构造查询头
        cur_time = str(int(time.time()))
        param2 = base64.b64encode(json.dumps({'task_id':task_id}).encode()).decode()
        checksum2 = hashlib.md5((api_key + cur_time + param2).encode()).hexdigest()
        headers2 = {
            'X-Appid':app_id,
            'X-CurTime':cur_time,
            'X-Param':param2,
            'X-CheckSum':checksum2
        }
        res = requests.get(result_url, headers=headers2, timeout=timeout)
        dj = res.json()
        if dj.get('code') == '0':
            return dj.get('data')
    return ''

def load_model(base_model_name_or_path, model_dir, use_whisper, device):
    # base_model_name_or_path is the pretrained base model; model_dir contains adapter weights
    base = base_model_name_or_path
    if use_whisper:
        processor = WhisperProcessor.from_pretrained(base)
        base_model = WhisperForConditionalGeneration.from_pretrained(base)
    else:
        processor = Wav2Vec2Processor.from_pretrained(base)
        base_model = Wav2Vec2ForCTC.from_pretrained(base)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.to(device)
    model.eval()
    return processor, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='LoRA 模型目录')
    parser.add_argument('--manifest', required=True, help='验证集 manifest CSV 路径')
    parser.add_argument('--base_model_name_or_path', required=True, help='预训练基模型名称或路径')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_metric', choices=['wer','cer'], default='cer')
    parser.add_argument('--language', default='zh')
    parser.add_argument('--task', choices=['transcribe','translate'], default='transcribe')
    parser.add_argument('--result_csv', required=True, help='结果输出CSV文件路径')
    parser.add_argument('--error_csv', required=True, help='错字统计CSV文件路径')
    parser.add_argument('--use_xunfei_compare', action='store_true', help='启用讯飞转录作为可选对照项')
    parser.add_argument('--xunfei_config', default=None, help='讯飞接口配置文件路径')
    # 生成配置，用于确保正常中断输出
    parser.add_argument('--num_beams', type=int, default=1, help='Beam 搜索大小， >1 时启用束搜索')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='生成长度惩罚')
    parser.add_argument('--early_stopping', action='store_true', help='启用 early_stopping 以在达到 eos 时提前结束')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='最大生成 token 数')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3, help='禁止重复 n-gram 大小')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='重复惩罚系数')
    parser.add_argument('--do_sample', action='store_true', help='启用采样策略')
    parser.add_argument('--top_k', type=int, default=50, help='top-k 采样参数')
    parser.add_argument('--top_p', type=float, default=0.95, help='top-p 采样参数')
    parser.add_argument('--segment_length_sec', type=int, default=30, help='超过此时长时拆分音频以防内存溢出')
    args = parser.parse_args()

    device = torch.device(args.device)
    # 判断是否为 Whisper 模型，根据 base_model_name_or_path 而非 adapter 目录
    use_whisper = True if 'whisper' in args.base_model_name_or_path.lower() else False
    processor, model = load_model(
        args.base_model_name_or_path,
        args.model_dir,
        use_whisper,
        device
    )
    # 加载原始基模型，用于速度对比
    if use_whisper:
        base_model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model_name_or_path).to(device).eval()
    else:
        base_model = Wav2Vec2ForCTC.from_pretrained(
            args.base_model_name_or_path).to(device).eval()
    # 初始化保存预测与时间列表
    base_preds, lora_preds, refs = [], [], []
    base_times, lora_times, xf_times = [], [], []
    xf_preds = []
    with open(args.manifest, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            audio_fp = row['audio_filepath']
            ref = row['text'].strip()
            # 流式分段读取音频，防止一次性加载整个文件导致内存溢出
            import numpy as np
            sf_obj = sf.SoundFile(audio_fp, 'r')
            sr = sf_obj.samplerate
            block_size = args.segment_length_sec * sr
            base_pred_segs, lora_pred_segs = [], []
            t_base_total, t_lora_total = 0.0, 0.0
            # 分段循环（使用流式blocks并在 no_grad 上下文中执行）
            with torch.no_grad(), sf.SoundFile(audio_fp) as f:
                for block in f.blocks(blocksize=block_size, dtype='float32'):
                    # 合并多声道为单声道
                    seg = block if block.ndim == 1 else block.mean(axis=1)
                    if use_whisper:
                        inputs = processor(seg, sampling_rate=sr, return_tensors='pt', return_attention_mask=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        mask = inputs['attention_mask']
                        forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
                        # 基准模型
                        t0 = time.time()
                        base_ids = base_model.generate(
                            input_features=inputs['input_features'],
                            attention_mask=mask,
                            eos_token_id=processor.tokenizer.eos_token_id,
                            early_stopping=args.early_stopping,
                            max_new_tokens=args.max_new_tokens,
                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                            repetition_penalty=args.repetition_penalty,
                            num_beams=args.num_beams,
                            length_penalty=args.length_penalty,
                            do_sample=args.do_sample,
                            top_k=args.top_k,
                            top_p=args.top_p
                        )
                        t_base_total += time.time() - t0
                        base_txt = processor.batch_decode(base_ids, skip_special_tokens=True)[0]
                        base_txt = preprocess_text(base_txt)
                        base_pred_segs.append(base_txt)
                        # LoRA 模型
                        t1 = time.time()
                        lora_ids = model.base_model.generate(
                            input_features=inputs['input_features'],
                            attention_mask=mask,
                            eos_token_id=processor.tokenizer.eos_token_id,
                            early_stopping=args.early_stopping,
                            max_new_tokens=args.max_new_tokens,
                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                            repetition_penalty=args.repetition_penalty,
                            num_beams=args.num_beams,
                            length_penalty=args.length_penalty,
                            do_sample=args.do_sample,
                            top_k=args.top_k,
                            top_p=args.top_p
                        )
                        t_lora_total += time.time() - t1
                        lora_txt = processor.batch_decode(lora_ids, skip_special_tokens=True)[0]
                        lora_txt = preprocess_text(lora_txt)
                        lora_pred_segs.append(lora_txt)
                        # 清理临时变量并释放显存
                        del inputs, base_ids, lora_ids
                        torch.cuda.empty_cache()
                        import gc; gc.collect()
                    else:
                        # 类似 logic for Wav2Vec2 CTC
                        inputs = processor(seg, sampling_rate=sr, return_tensors='pt')
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        t0 = time.time()
                        logits_base = base_model(**inputs).logits
                        ids_base = logits_base.argmax(-1)
                        t_base_total += time.time() - t0
                        base_txt = processor.batch_decode(ids_base, skip_special_tokens=True)[0]
                        base_txt = preprocess_text(base_txt)
                        base_pred_segs.append(base_txt)
                        t1 = time.time()
                        logits_lora = model.base_model(**inputs).logits
                        ids_lora = logits_lora.argmax(-1)
                        t_lora_total += time.time() - t1
                        lora_txt = processor.batch_decode(ids_lora, skip_special_tokens=True)[0]
                        lora_txt = preprocess_text(lora_txt)
                        lora_pred_segs.append(lora_txt)
                        # 清理临时变量并释放显存
                        del inputs, logits_base, logits_lora
                        torch.cuda.empty_cache()
                        import gc; gc.collect()
            # 清理 sf_obj 及释放缓存
            torch.cuda.empty_cache(); import gc; gc.collect()
            
            # 合并分段结果
            base_pred = ' '.join(base_pred_segs)
            lora_pred = ' '.join(lora_pred_segs)
            base_times.append(t_base_total)
            lora_times.append(t_lora_total)
            preds = lora_pred
            refs.append(ref)
            base_preds.append(base_pred)
            lora_preds.append(lora_pred)
            # 继续后续比对与评估流程
            continue
            # 原 xunfei 比对与评价逻辑以下开始
            # 可选讯飞对照及时间
            if args.use_xunfei_compare:
                t_xf0 = time.time()
                xf_pred = transcribe_xunfei(audio_fp, args.xunfei_config)
                xf_time = time.time() - t_xf0
            else:
                xf_pred, xf_time = None, 0.0
            xf_preds.append(xf_pred)
            xf_times.append(xf_time)
            if use_whisper:
                # 包含 attention_mask
                inputs = processor(data, sampling_rate=sr, return_tensors='pt', return_attention_mask=True)
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=args.language, task=args.task)
                mask = inputs.attention_mask.to(device)
                # 原始模型推理
                t0 = time.time()
                base_gen_ids = base_model.generate(
                    input_features=inputs.input_features.to(device),
                    attention_mask=mask,
                    forced_decoder_ids=forced_decoder_ids,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    early_stopping=args.early_stopping,
                    max_new_tokens=args.max_new_tokens,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                t_base = time.time() - t0
                base_pred = processor.batch_decode(base_gen_ids, skip_special_tokens=True)[0]
                # 标准化基准模型输出
                base_pred = preprocess_text(base_pred)
                base_preds.append(base_pred)
                base_times.append(t_base)
                # LoRA 模型推理
                t1 = time.time()
                lora_gen_ids = model.base_model.generate(
                    input_features=inputs.input_features.to(device),
                    attention_mask=mask,
                    forced_decoder_ids=forced_decoder_ids,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    early_stopping=args.early_stopping,
                    max_new_tokens=args.max_new_tokens,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                t_lora = time.time() - t1
                lora_pred = processor.batch_decode(lora_gen_ids, skip_special_tokens=True)[0]
                # 标准化LoRA模型输出
                lora_pred = preprocess_text(lora_pred)
                lora_preds.append(lora_pred)
                lora_times.append(t_lora)
                pred = lora_pred
            else:
                inputs = processor(data, sampling_rate=sr, return_tensors='pt')
                # 原始模型推理
                t0 = time.time()
                logits_base = base_model(inputs.input_values.to(device)).logits
                pred_ids_base = logits_base.argmax(-1)
                base_pred = processor.batch_decode(pred_ids_base)[0]
                # 标准化基准模型输出
                base_pred = preprocess_text(base_pred)
                base_preds.append(base_pred)
                base_times.append(t_base)
                # LoRA 模型推理
                t1 = time.time()
                logits_lora = model.base_model(inputs.input_values.to(device)).logits
                pred_ids_lora = logits_lora.argmax(-1)
                lora_pred = processor.batch_decode(pred_ids_lora)[0]
                # 标准化LoRA模型输出
                lora_pred = preprocess_text(lora_pred)
                lora_preds.append(lora_pred)
                lora_times.append(t_lora)
                pred = lora_pred
            # 清理特殊标记
            for tok in ['<|startoftranscript|>','<|endoftext|>','<|notimestamps|>']:
                pred = pred.replace(tok, '')
                ref = ref.replace(tok, '')
            # 文本预处理
            pred = preprocess_text(pred)
            ref = preprocess_text(ref)
            # 打印每个验证样本的预测与参考，方便检查
            print(f"Sample {idx+1}:")
            print(f"  Pred: {pred}")
            print(f"  Ref : {ref}")
            refs.append(ref)
    # 打印速度统计示例
    print(f"Avg base time: {sum(base_times)/len(base_times):.3f}s, Avg LoRA time: {sum(lora_times)/len(lora_times):.3f}s")
    # 输出结果CSV
    fieldnames = ['id','base_pred','lora_pred','ref','comparison','base_time','lora_time','base_cer','lora_cer','cer_delta']
    if args.use_xunfei_compare:
        fieldnames += ['xf_pred','xf_time']
    with open(args.result_csv,'w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(refs)):
            # 计算每条的 CER
            base_cer_val = jiwer.cer(refs[i], base_preds[i])
            lora_cer_val = jiwer.cer(refs[i], lora_preds[i])
            # 计算cer值百分数和差值
            base_cer_pct = base_cer_val * 100
            lora_cer_pct = lora_cer_val * 100
            delta_pct = base_cer_pct - lora_cer_pct
            row = {
                'id': i+1,
                'base_pred': base_preds[i],
                'lora_pred': lora_preds[i],
                'ref': refs[i],
                'comparison': f"{base_preds[i]}||{lora_preds[i]}",
                'base_time': f"{base_times[i]:.3f}",
                'lora_time': f"{lora_times[i]:.3f}",
                'base_cer': f"{base_cer_pct:.2f}%",
                'lora_cer': f"{lora_cer_pct:.2f}%",
                'cer_delta': f"{delta_pct:.2f}%"
            }
            if args.use_xunfei_compare:
                row['xf_pred'] = xf_preds[i]
                row['xf_time'] = f"{xf_times[i]:.3f}"
            writer.writerow(row)
    # 生成错误统计CSV
    import difflib
    from pypinyin import lazy_pinyin
    err_fieldnames = ['id','insertion_ratio','deletion_ratio','substitution_ratio','homophone_sub_ratio']
    with open(args.error_csv,'w',newline='',encoding='utf-8') as ef:
        err_writer = csv.DictWriter(ef, fieldnames=err_fieldnames)
        err_writer.writeheader()
        for i, (ref, pred) in enumerate(zip(refs, lora_preds)):
            # 计算插入、删除、替换比例
            matcher = difflib.SequenceMatcher(None, ref, pred)
            ins = dels = subs = homo = 0
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'insert':
                    ins += (j2 - j1)
                elif tag == 'delete':
                    dels += (i2 - i1)
                elif tag == 'replace':
                    length = min(i2 - i1, j2 - j1)
                    subs += length
                    # 同音替换
                    for k in range(length):
                        ref_char = ref[i1 + k]
                        pred_char = pred[j1 + k]
                        if lazy_pinyin(ref_char) == lazy_pinyin(pred_char):
                            homo += 1
            total_ref = len(ref) if len(ref) > 0 else 1
            ins_ratio = ins / total_ref
            del_ratio = dels / total_ref
            sub_ratio = subs / total_ref
            homo_ratio = homo / subs if subs > 0 else 0
            err_writer.writerow({
                'id': i+1,
                'insertion_ratio': f"{ins_ratio*100:.2f}%",
                'deletion_ratio': f"{del_ratio*100:.2f}%",
                'substitution_ratio': f"{sub_ratio*100:.2f}%",
                'homophone_sub_ratio': f"{homo_ratio*100:.2f}%"
            })

    if args.eval_metric == 'wer':
        score = jiwer.wer(refs, lora_preds)
        print(f"Validation WER: {score:.4f}")
    else:
        score = jiwer.cer(refs, lora_preds)
        print(f"Validation CER: {score:.4f}")

if __name__ == '__main__':
    main()
