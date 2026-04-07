import os
import sys # noqa: E401, F401, F403, E0401  # type: ignores
from collections import namedtuple

import torch  

sys.path.insert(0, os.getcwd())
from speaker_separator import SpeakerSeparator


def test_diarize_one_speaker():
    # 使用 CPU 以避免 GPU 依赖
    separator = SpeakerSeparator(device="cuda")
    # 测试语料文件
    audio_path = os.path.join(os.path.dirname(__file__), "test2.wav")
    # 执行说话人分离（diarize）
    output = separator.diarize(audio_path)
    # 获取 Annotation 对象（DiarizeOutput 有 speaker_diarization 属性）
    annotation = (
        output.speaker_diarization
        if hasattr(output, "speaker_diarization")
        else output
    )
    speakers = list(annotation.labels())
    assert len(speakers) == 1, f"检测到说话人数量为 {len(speakers)}, 期望 1 人"


def test_save_speaker_segments(tmp_path):
    separator = SpeakerSeparator(device="cpu")
    audio_path = os.path.join(os.path.dirname(__file__), "test2.wav")
    # 指定输出目录
    out_dir = tmp_path / "speakers"
    returned = separator.save_speaker_segments(audio_path, str(out_dir))
    # 返回值应为传入的路径
    assert returned == str(out_dir)
    # 输出目录应存在并包含至少一个 wav 文件
    files = list(out_dir.glob("*.wav"))
    assert len(files) > 0, "未生成说话人分段文件"


def test_cluster_speakers(monkeypatch):
    # 创建 separator，使用 CPU
    separator = SpeakerSeparator(device="cpu")
    # 构造 DummyTurn
    Turn = namedtuple("Turn", ["start", "end", "speaker"])

    # 构造 Annotation 模拟
    class DummyAnnotation:
        def __init__(self, turns):
            self._turns = turns

        def itertracks(self, yield_label=True):
            for t in self._turns:
                yield t, None, t.speaker

    turns = [
        Turn(0.0, 1.0, "A"),
        Turn(1.0, 2.0, "A"),
        Turn(2.0, 3.0, "B"),
        Turn(3.0, 4.0, "B"),
    ]
    annotation = DummyAnnotation(turns)
    monkeypatch.setattr(separator, "diarize", lambda filename: annotation)

    # Stub embedding，根据 speaker 返回不同向量
    def fake_crop(file, turn, batch_size):
        if turn.speaker == "A":
            return torch.tensor([1.0, 0.0])
        return torch.tensor([0.0, 1.0])

    monkeypatch.setattr(separator.embedder, "crop", fake_crop)
    # 执行聚类
    clusters = separator.cluster_speakers("dummy.wav", threshold=0.5)
    # 应得到两个簇，每簇含两个片段
    assert set(clusters.keys()) == {0, 1}
    assert all(len(seg_list) == 2 for seg_list in clusters.values())


def test_cluster_speakers_same_audio(monkeypatch):
    """
    模拟同一音频两次聚类应合并到同一簇
    """

    # 初始化 separator
    separator = SpeakerSeparator(device="cpu")
    # 构造两个相同的分段
    Turn = namedtuple("Turn", ["start", "end", "speaker"])
    turns = [Turn(0.0, 1.0, "A"), Turn(0.0, 1.0, "A")]

    class DummyAnnotation:
        def __init__(self, turns):
            self._turns = turns

        def itertracks(self, yield_label=True):
            for t in self._turns:
                yield t, None, t.speaker

    annotation = DummyAnnotation(turns)
    monkeypatch.setattr(separator, "diarize", lambda filename: annotation)
    # stub embedding 始终返回相同向量
    monkeypatch.setattr(
        separator.embedder,
        "crop",
        lambda file, turn, batch_size: torch.tensor([1.0, 0.0]),
    )
    clusters = separator.cluster_speakers("test2.wav", threshold=0.5)
    # 只会有一个簇，包含两个分段
    assert set(clusters.keys()) == {0}
    assert len(clusters[0]) == 2


def test_cluster_speakers_simple(monkeypatch):
    """
    使用真实 embedder，模拟同一音频的两个相同分段，测试聚类合并功能
    """

    # 初始化 separator
    separator = SpeakerSeparator(device="cpu")
    # 测试音频路径
    audio_path = os.path.join(os.path.dirname(__file__), "test2.wav")
    # 构造两个相同的分段
    Turn = namedtuple("Turn", ["start", "end", "speaker"])
    turns = [Turn(0.0, 1.0, "A"), Turn(0.0, 1.0, "A")]

    class DummyAnnotation:
        def __init__(self, turns):
            self._turns = turns

        def itertracks(self, yield_label=True):
            for t in self._turns:
                yield t, None, t.speaker

    annotation = DummyAnnotation(turns)
    # Monkeypatch diarize
    monkeypatch.setattr(separator, "diarize", lambda filename: annotation)
    # stub embedding：避免依赖 HF token / 网络

    monkeypatch.setattr(
        separator.embedder,
        "crop",
        lambda file, turn, batch_size: torch.tensor([1.0, 0.0]),
    )
    # 执行聚类
    clusters = separator.cluster_speakers(audio_path, threshold=0.5)
    # 只会有一个簇，包含两个分段
    assert set(clusters.keys()) == {0}
    assert len(clusters[0]) == 2
