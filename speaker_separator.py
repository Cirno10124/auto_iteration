#!/usr/bin/env python3
import json
import os

import librosa
import noisereduce
import numpy as np
import torch
from pyannote.audio import Pipeline


class SpeakerSeparator:
    """
    加载并封装说话人分离模型的类。

    用法：
        separator = SpeakerSeparator()
        diarization = separator.diarize(audio_file)
    """

    def __init__(
        self,
        model_revision: str = "pyannote/speaker-diarization",
        hf_token: str = None,
        device: str = "cuda",
        snr_threshold: float = 20,
    ):
        # 加载配置
        config_path = os.path.join(
            os.path.dirname(__file__), "speaker_separator_config.json"
        )
        if hf_token is None:
            try:
                with open(config_path, "r") as cf:
                    cfg = json.load(cf)
                model_revision = cfg.get("model_revision", model_revision)
                hf_token = cfg.get("hf_token", hf_token)
                device = cfg.get("device", device)
                snr_threshold = cfg.get("snr_threshold", snr_threshold)
            except Exception as e:
                print(f"警告：加载配置文件失败，将使用默认参数: {e}")

        # 转换设备类型
        if isinstance(device, str):
            device = torch.device(device)

        # 加载说话人分离模型
        print("正在加载 pyannote.audio 说话人分离模型...")
        try:
            self.pipeline = Pipeline.from_pretrained(
                model_revision, token=hf_token
            )
            self.pipeline.to(device)
            print("模型加载完成")
        except Exception as e:
            print(f"错误：无法加载 pyannote.audio 模型: {e}")
            print("请检查：1) HF_HUB_TOKEN 是否正确设置；2) 网络连接；3) CUDA 可用性")
            raise

        # 源分离功能暂不支持，注释此部分
        self.sep_pipeline = None

        # 设置 SNR 阈值
        self.snr_threshold = snr_threshold

    def _load_and_preprocess(self, filename: str):
        """
        加载并降采样到16kHz单声道，基于 SNR 决定是否降噪
        """
        waveform, sr = librosa.load(filename, sr=16000, mono=True)
        noise_pow = np.mean(waveform[: int(sr * 0.5)] ** 2)
        signal_pow = np.mean(waveform**2)
        snr = 10 * np.log10(signal_pow / (noise_pow + 1e-8))
        if snr < self.snr_threshold:
            waveform = noisereduce.reduce_noise(y=waveform, sr=sr)
        return waveform, sr

    def diarize(self, filename: str):
        """
        对音频文件进行说话人分离并返回 Annotation。
        """
        # 直接传路径给 pipeline
        return self.pipeline(filename)

    def separate_sources(self, filename: str):
        """
        源分离暂不支持，返回 None。
        """
        return None
