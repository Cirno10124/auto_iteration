import sys, os
sys.path.insert(0, os.getcwd())
import pytest
from speaker_separator import SpeakerSeparator


def test_diarize_one_speaker():
    # 使用 CPU 以避免 GPU 依赖
    separator = SpeakerSeparator(device="cuda")
    # 测试语料文件
    audio_path = os.path.join(os.path.dirname(__file__), "test2.wav")
    # 执行说话人分离（diarize）
    output = separator.diarize(audio_path)
    # 获取 Annotation 对象（DiarizeOutput 有 speaker_diarization 属性）
    annotation = output.speaker_diarization if hasattr(output, 'speaker_diarization') else output
    speakers = list(annotation.labels())
    assert len(speakers) == 1, f"检测到说话人数量为 {len(speakers)}, 期望 1 人"
