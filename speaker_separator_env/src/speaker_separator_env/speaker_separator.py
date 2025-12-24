#!/usr/bin/env python3
import json
import os

# 兼容补丁：speechbrain 在导入时会调用 torchaudio.list_audio_backends()
# 某些环境的 torchaudio 可能缺失该 API（版本不匹配或被同名模块覆盖），导致 pyannote diarization pipeline 无法加载。
try:
    import torchaudio  # noqa: E0401

    if not hasattr(torchaudio, "list_audio_backends"):

        def _list_audio_backends():
            # 尽量返回一个合理的后端列表；speechbrain 主要是用它做可用性检查
            return ["soundfile"]

        torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]

    # 部分库还会调用 torchaudio.get_audio_backend / set_audio_backend
    if not hasattr(torchaudio, "get_audio_backend"):

        def _get_audio_backend():
            return "soundfile"

        torchaudio.get_audio_backend = _get_audio_backend  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "set_audio_backend"):

        def _set_audio_backend(_backend: str):
            # torchaudio>=2.9 移除了旧后端选择机制，这里做兼容 no-op
            return None

        torchaudio.set_audio_backend = _set_audio_backend  # type: ignore[attr-defined]
except Exception:
    # 没有 torchaudio 或导入失败时不处理，让后续按原始错误提示
    pass

import librosa  # noqa: E0401
import noisereduce  # noqa: E0401
import numpy as np
import torch  # noqa: E0401
from pyannote.audio import Pipeline  # noqa: E0401
from sklearn.cluster import AgglomerativeClustering


class DummyEmbedder:
    """
    用于懒加载 embedding 的临时类，使用 pyannote Model+Inference 实现 crop。

    参考 HF 示例：
        from pyannote.audio import Model, Inference
        model = Model.from_pretrained("pyannote/embedding")
        inference = Inference(model)
        inference.crop("file.wav", Segment(start=2.0, end=5.0))
    """

    def __init__(
        self,
        hf_token=None,
        device=None,
        model_id: str = "pyannote/embedding",
        fallback_model_id: str = (
            "speechbrain/spkrec-ecapa-voxceleb@"
            "5c0be3875fda05e81f3c004ed8c7c06be308de1e"
        ),
    ):
        self.hf_token = hf_token
        self.device = device
        self.model_id = model_id
        # 直接使用 SpeechBrain ECAPA 作为真实 embedding，避免 pyannote/embedding 的额外依赖链
        self.fallback_model_id = fallback_model_id
        self._embedder = None

    def _lazy_init(self):
        if self._embedder is not None:
            return

        from pyannote.audio.pipelines.speaker_verification import (  # noqa: E0401
            PretrainedSpeakerEmbedding,
        )

        self._embedder = PretrainedSpeakerEmbedding(
            self.fallback_model_id,
            device=str(self.device) if self.device is not None else "cpu",
        )

    def crop(self, file, excerpt, batch_size=8):
        """
        对音频文件的一个片段提取 embedding。

        - file: 音频文件路径
        - excerpt: pyannote.core.Segment 或任意具有 start/end 属性的对象（例如 turn）
        """
        self._lazy_init()

        # SpeechBrain ECAPA 期望输入为 torch.Tensor: (batch_size, num_channels, num_samples)
        import numpy as _np
        import soundfile as _sf  # noqa: E0401

        audio, sr = _sf.read(file)
        if audio.ndim > 1:
            audio = _np.mean(audio, axis=1)

        start = max(0, int(float(excerpt.start) * sr))
        end = min(len(audio), int(float(excerpt.end) * sr))
        segment = audio[start:end]

        # SpeechBrain ECAPA 通常使用 16kHz。若采样率不同，尽量重采样以提升稳定性。
        if sr != 16000:
            segment = librosa.resample(
                segment.astype(_np.float32), orig_sr=sr, target_sr=16000
            )
            sr = 16000
        else:
            segment = segment.astype(_np.float32, copy=False)

        # 兼容极短音频：SpeechBrain 的 ECAPA 前向在 num_samples 太小时会在内部 squeeze 后触发 shape 错误。
        # 这里将片段扩展到一个合理的最小时长（默认 1.0s），优先用“重复填充”避免全零导致 embedding 不稳定。
        min_duration_s = 1.0
        min_samples = int(sr * min_duration_s)
        if segment.ndim != 1:
            segment = _np.asarray(segment, dtype=_np.float32).reshape(-1)
        if segment.size < min_samples:
            if segment.size <= 0:
                segment = _np.zeros(min_samples, dtype=_np.float32)
            else:
                reps = int(_np.ceil(min_samples / float(segment.size)))
                segment = _np.tile(segment, reps)[:min_samples]

        # (1, 1, num_samples)
        waveforms = (
            torch.from_numpy(segment.astype(_np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        embeddings = self._embedder(waveforms)
        # 返回 1D 向量，供上层 np.stack
        return _np.asarray(embeddings).reshape(-1)


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
        hf_token: str | None = None,
        device: str = "cuda",
        snr_threshold: float = 20,
    ):
        # 配置加载优先级：
        # 1) 显式参数
        # 2) SPEAKER_SEPARATOR_CONFIG 环境变量指向的 JSON
        # 3) 与本模块同目录的 speaker_separator_config.json
        # 4) 当前工作目录下的 speaker_separator_config.json
        config_candidates: list[str] = []
        env_cfg = os.environ.get("SPEAKER_SEPARATOR_CONFIG")
        if env_cfg:
            config_candidates.append(env_cfg)

        config_candidates.append(
            os.path.join(
                os.path.dirname(__file__), "speaker_separator_config.json"
            )
        )
        config_candidates.append(
            os.path.abspath("speaker_separator_config.json")
        )

        if hf_token is None:
            cfg = None
            for p in config_candidates:
                try:
                    if p and os.path.exists(p):
                        with open(p, "r", encoding="utf-8") as cf:
                            cfg = json.load(cf)
                        break
                except Exception as e:
                    print(f"警告：加载配置文件失败，将使用默认参数: {e}")
                    cfg = None
            if isinstance(cfg, dict):
                model_revision = cfg.get("model_revision", model_revision)
                hf_token = cfg.get("hf_token", hf_token)
                device = cfg.get("device", device)
                snr_threshold = cfg.get("snr_threshold", snr_threshold)

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.hf_token = hf_token
        # 延迟加载 embedding 推断器，cluster_speakers 时再初始化
        self.embedder = DummyEmbedder(
            hf_token=self.hf_token, device=self.device
        )

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

        # 源分离功能暂不支持
        self.sep_pipeline = None

        # 信噪比阈值
        self.snr_threshold = snr_threshold

    def _load_and_preprocess(self, filename: str):
        """
        加载并降采样到16kHz单声道，基于 SNR 判断是否降噪
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
        output = self.pipeline(filename)
        # 如果返回 DiarizeOutput，提取 speaker_diarization
        if hasattr(output, "speaker_diarization"):
            return output.speaker_diarization
        return output

    def separate_sources(self, filename: str):
        """
        源分离暂不支持，返回 None。
        """
        return None

    def cluster_speakers(self, filename: str, threshold: float = 0.75):
        """
        用 embedding 做层次聚类，threshold 控制合并阈值（余弦距离）。
        返回 {cluster_id: [(start,end), ...], ...}
        """
        # 1. 获取分段
        annotation = self.diarize(filename)
        # 2. 提取 embedding
        segments = []
        emb_list = []
        for turn, _, _ in annotation.itertracks(yield_label=True):
            segments.append(turn)
            emb = self.embedder.crop(filename, turn, batch_size=8)
            # 兼容多种返回类型：torch.Tensor / numpy / list
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            elif hasattr(emb, "numpy"):
                # 某些对象实现 numpy() 方法
                emb = emb.numpy()
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim != 1:
                emb = emb.reshape(-1)
            emb_list.append(emb)
        X = np.stack(emb_list, axis=0).astype(np.float32, copy=False)
        # 3. 聚类
        # sklearn 新版本用 metric，老版本用 affinity；这里做兼容
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=threshold,
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                affinity="cosine",
                linkage="average",
                distance_threshold=threshold,
            )
        labels = clustering.fit_predict(X)
        # 4. 构建结果
        clusters = {}
        for turn, label in zip(segments, labels):
            clusters.setdefault(int(label), []).append(
                (float(turn.start), float(turn.end))
            )
        return clusters

    def save_speaker_segments(
        self, audio_file: str, out_dir: str | None = None
    ):
        """
        将说话人分段保存到指定目录（默认: 相对于模块目录的 out/speakers）。
        """
        # 确定输出目录
        if out_dir is None:
            base = os.path.dirname(__file__)
            out_dir = os.path.join(base, "out", "speakers")
        os.makedirs(out_dir, exist_ok=True)

        # 获取 Annotation 对象
        annotation = self.diarize(audio_file)

        # 读取原始波形
        waveform, sr = librosa.load(audio_file, sr=None, mono=True)

        # 按说话人片段保存
        import soundfile as sf  # noqa: E0401

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = int(turn.start * sr)
            end = int(turn.end * sr)
            segment = waveform[start:end]
            filename = f"{speaker}_{start}_{end}.wav"
            path = os.path.join(out_dir, filename)
            sf.write(path, segment, sr)

        return out_dir
