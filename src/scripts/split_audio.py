#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad


def frame_generator(frame_duration_ms, audio_bytes, sample_rate):
    n = int(sample_rate * frame_duration_ms / 1000) * 2
    for i in range(0, len(audio_bytes), n):
        yield audio_bytes[i : i + n]


def vad_collector(sample_rate, frame_duration_ms, vad, frames):
    segments = []
    segment = b""
    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2
    for frame in frames:
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate):
            segment += frame
        else:
            if segment:
                segments.append(segment)
                segment = b""
    if segment:
        segments.append(segment)
    return segments


def _convert_to_wav_16k_mono(input_path, output_dir, sample_rate):
    """若输入不是 wav 或采样率不匹配，则转换为 16k 单声道 wav。"""
    data, sr = sf.read(input_path, dtype="float32")
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)

    suffix = Path(input_path).suffix.lower()
    needs_convert = suffix != ".wav" or sr != sample_rate
    if not needs_convert:
        # 保持与后续 VAD 一致的 int16 格式
        return input_path, data.astype(np.float32), sr

    # 线性重采样到目标采样率
    if sr != sample_rate:
        target_len = int(len(data) * sample_rate / sr)
        target_len = max(target_len, 1)
        data = np.interp(
            np.linspace(0, len(data), target_len, endpoint=False),
            np.arange(len(data)),
            data,
        ).astype(np.float32)

    converted_dir = os.path.join(output_dir, "_normalized")
    os.makedirs(converted_dir, exist_ok=True)
    converted_name = f"{Path(input_path).stem}_16k.wav"
    converted_path = os.path.join(converted_dir, converted_name)
    sf.write(converted_path, data, sample_rate, subtype="PCM_16")
    return converted_path, data, sample_rate


def vad_split(
    input_path,
    output_dir,
    sample_rate=16000,
    frame_duration=30,
    vad_aggressiveness=3,
    min_segment_duration=500,
    merge_threshold=15,
):
    """Split audio from input_path into segments using VAD and save to output_dir."""
    normalized_path, float_data, sr = _convert_to_wav_16k_mono(
        input_path, output_dir, sample_rate
    )
    int16_data = np.clip(float_data, -1.0, 1.0)
    data = (int16_data * np.iinfo(np.int16).max).astype(np.int16)
    raw_audio = data.tobytes()
    vad = webrtcvad.Vad(vad_aggressiveness)
    # 为每个输入文件创建独立子目录
    base = Path(input_path).stem
    out_dir = os.path.join(output_dir, base)
    os.makedirs(out_dir, exist_ok=True)
    # 分帧并收集语音段
    frames = frame_generator(frame_duration, raw_audio, sample_rate)
    segments = vad_collector(sample_rate, frame_duration, vad, frames)
    # 合并短于最小时长的片段，且不超过最大合并时长阈值
    merged = []
    min_samples = int(min_segment_duration * sample_rate / 1000)
    threshold_samples = int(merge_threshold * sample_rate)
    for seg in segments:
        arr = np.frombuffer(seg, dtype=np.int16)
        if arr.size < min_samples:
            if merged:
                prev_arr = np.frombuffer(merged[-1], dtype=np.int16)
                if prev_arr.size + arr.size <= threshold_samples:
                    merged[-1] += seg
                else:
                    merged.append(seg)
            else:
                merged.append(seg)
        else:
            merged.append(seg)
    segments = merged
    # 写入分割后的 wav 文件
    for idx, seg in enumerate(segments, start=1):
        arr = np.frombuffer(seg, dtype=np.int16)
        # 使用两位序号命名：01.wav, 02.wav...
        filename = f"{idx:02d}.wav"
        out_path = os.path.join(out_dir, filename)
        sf.write(out_path, arr, sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input audio file (PCM WAV/FLAC/MP3)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save audio segments",
    )
    parser.add_argument(
        "--min_segment_duration",
        type=int,
        default=500,
        help="Minimum segment duration in ms; shorter segments will be merged",
    )
    # 新增最大合并时长阈值参数（秒）
    parser.add_argument(
        "--merge_threshold",
        type=int,
        default=15,
        help="Maximum merged segment duration in seconds",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sampling rate for audio",
    )
    parser.add_argument(
        "--frame_duration",
        type=int,
        default=30,
        help="Frame duration in ms",
    )
    parser.add_argument(
        "--vad_aggressiveness",
        type=int,
        default=3,
        help="VAD aggressiveness 0-3",
    )
    args = parser.parse_args()
    vad_split(
        args.input,
        args.output_dir,
        args.sample_rate,
        args.frame_duration,
        args.vad_aggressiveness,
        args.min_segment_duration,
        args.merge_threshold,
    )
