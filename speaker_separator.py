#!/usr/bin/env python3
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerSeparation


class SpeakerSeparator:
    """
    加载并封装说话人分离（发言人分离和源分离）模型的类。
    用法：
        separator = SpeakerSeparator()
        diarization = separator.diarize(audio_file)
        sources = separator.separate_sources(audio_file)
    """

    def __init__(self, revision="main", token=True, device="cuda"):
        # 加载说话人分离（diarization）模型
        print("正在加载 pyannote.audio 说话人分离模型...")
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                revision=revision,
                token=token,
            )
            self.pipeline.to(device)
            print("模型加载完成")
        except Exception as e:
            print(f"错误：无法加载 pyannote.audio 模型: {e}")
            print(
                "请检查：1) Hugging Face token 是否正确设置；2) 网络连接是否正常；3) CUDA 是否可用"
            )
            raise

        # 尝试加载源分离（source separation）模型
        try:
            self.sep_pipeline = SpeakerSeparation.from_pretrained(
                "pyannote/source-separation",
                revision=revision,
                token=token,
            )
            self.sep_pipeline.to(device)
            print("源分离模型加载完成")
        except Exception as e:
            print(f"警告：无法加载源分离模型: {e}")
            self.sep_pipeline = None

    def diarize(self, filename):
        """对音频文件进行说话人分离并返回分离结果。"""
        return self.pipeline(filename)

    def separate_sources(self, filename):
        """对音频文件进行源分离，返回通道字典或 None。"""
        if self.sep_pipeline is None:
            return None
        return self.sep_pipeline(filename)
