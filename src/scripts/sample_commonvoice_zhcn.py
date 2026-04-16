#!/usr/bin/env python3
"""
从 Hugging Face Common Voice (zh-CN) 按需采样一个小数据集（例如 4 个说话人 * 每人 10 条）。

特点：
- 使用 datasets streaming=True，只会下载被选中的少量音频样本
- 输出 16kHz 单声道 wav + manifest.jsonl（含 speaker_id/client_id）

用法示例：
  python auto_iteration/sample_commonvoice_zhcn.py --out_dir out/cv_zhcn_4x10 --num_speakers 4 --per_speaker 10

可选：指定数据集版本/切分
  python auto_iteration/sample_commonvoice_zhcn.py --dataset mozilla-foundation/common_voice_17_0 --lang zh-CN --split train
"""

import argparse
import json
import os
import re
from collections import defaultdict

import librosa  # noqa: E0401
import numpy as np
import soundfile as sf  # noqa: E0401
from datasets import load_dataset  # noqa: E0401
from datasets.data_files import EmptyDatasetError  # noqa: E0401
from datasets.exceptions import DatasetNotFoundError  # noqa: E0401
from huggingface_hub import HfApi, HfFileSystem  # noqa: E0401


def _unwrap_audio_array(obj):
    """
    将 datasets/torchcodec/speechbrain 等可能返回的“音频样本对象”解包为 numpy 数组。
    """
    if obj is None:
        return None

    # torch.Tensor
    try:
        import torch  # noqa: E0401

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except Exception:
        pass

    # numpy array
    if isinstance(obj, np.ndarray):
        return obj

    # 常见封装：AudioSamples/类似对象，可能有 data/samples/tensor 等字段
    for attr in ["data", "samples", "tensor", "waveform", "array"]:
        if hasattr(obj, attr):
            return _unwrap_audio_array(getattr(obj, attr))

    # 常见方法：to_numpy/numpy
    for m in ["to_numpy", "numpy"]:
        if hasattr(obj, m) and callable(getattr(obj, m)):
            return _unwrap_audio_array(getattr(obj, m)())

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj)

    # 最后兜底：尽量转成 numpy
    return np.asarray(obj)


def _get_audio(example):
    """
    兼容 datasets 的 Audio 特征：
    - example["audio"] 可能是 {"array": np.ndarray, "sampling_rate": int, ...}
    - 也可能是其他字段名或格式
    """
    if "audio" not in example:
        # 兼容不同数据集的音频字段命名
        for k in ["speech", "wav", "waveform"]:
            if k in example:
                audio = example[k]
                # datasets Audio: {"array": ..., "sampling_rate": ...}
                if (
                    isinstance(audio, dict)
                    and "array" in audio
                    and "sampling_rate" in audio
                ):
                    return audio["array"], int(audio["sampling_rate"])
                # webdataset 可能直接给 wav bytes
                if isinstance(audio, (bytes, bytearray, memoryview)):
                    import io

                    import soundfile as _sf  # noqa: E0401

                    data, sr = _sf.read(io.BytesIO(bytes(audio)))
                    return data, int(sr)
                # webdataset 也可能给 dict 里包含 bytes/path
                if isinstance(audio, dict):
                    # 常见替代字段名
                    if "data" in audio and isinstance(
                        audio["data"], (bytes, bytearray, memoryview)
                    ):
                        import io

                        import soundfile as _sf  # noqa: E0401

                        data, sr = _sf.read(
                            io.BytesIO(bytes(audio["data"]))
                        )
                        return data, int(sr)
                    if "sample_rate" in audio and "array" in audio:
                        return audio["array"], int(audio["sample_rate"])
                    if "sr" in audio and "array" in audio:
                        return audio["array"], int(audio["sr"])
                    if "bytes" in audio and isinstance(
                        audio["bytes"], (bytes, bytearray)
                    ):
                        import io

                        import soundfile as _sf  # noqa: E0401

                        data, sr = _sf.read(io.BytesIO(audio["bytes"]))
                        return data, int(sr)
                    if "bytes" in audio and isinstance(
                        audio["bytes"], memoryview
                    ):
                        import io

                        import soundfile as _sf  # noqa: E0401

                        data, sr = _sf.read(
                            io.BytesIO(bytes(audio["bytes"]))
                        )
                        return data, int(sr)
                    if (
                        "path" in audio
                        and isinstance(audio["path"], str)
                        and audio["path"]
                    ):
                        import soundfile as _sf  # noqa: E0401

                        data, sr = _sf.read(audio["path"])
                        return data, int(sr)
                    if (
                        "filename" in audio
                        and isinstance(audio["filename"], str)
                        and audio["filename"]
                    ):
                        import soundfile as _sf  # noqa: E0401

                        data, sr = _sf.read(audio["filename"])
                        return data, int(sr)
                # 也可能直接给本地/缓存路径字符串
                if isinstance(audio, str) and audio:
                    import soundfile as _sf  # noqa: E0401

                    data, sr = _sf.read(audio)
                    return data, int(sr)
                # 也可能是 (array, sr) 的二元组/列表
                if isinstance(audio, (tuple, list)) and len(audio) == 2:
                    arr, sr = audio
                    if isinstance(sr, (int, np.integer)):
                        return arr, int(sr)
                # datasets + torchcodec: AudioDecoder
                # 兼容 datasets.features._torchcodec.AudioDecoder（不同版本方法名可能不同）
                if (
                    type(audio).__name__ == "AudioDecoder"
                    and "datasets.features._torchcodec"
                    in type(audio).__module__
                ):
                    # 优先尝试从其暴露的 path/bytes 读取
                    for attr in [
                        "path",
                        "_path",
                        "uri",
                        "_uri",
                        "filename",
                        "_filename",
                    ]:
                        if hasattr(audio, attr):
                            p = getattr(audio, attr)
                            if isinstance(p, str) and p:
                                import soundfile as _sf  # noqa: E0401

                                data, sr = _sf.read(p)
                                return data, int(sr)
                    for attr in [
                        "bytes",
                        "_bytes",
                        "data",
                        "_data",
                        "buffer",
                        "_buffer",
                    ]:
                        if hasattr(audio, attr):
                            b = getattr(audio, attr)
                            if isinstance(
                                b, (bytes, bytearray, memoryview)
                            ):
                                import io

                                import soundfile as _sf  # noqa: E0401

                                data, sr = _sf.read(io.BytesIO(bytes(b)))
                                return data, int(sr)
                    # 再尝试可能的方法名
                    for m in [
                        "decode",
                        "get_all_samples",
                        "get_samples",
                        "to_numpy",
                        "to_array",
                    ]:
                        if hasattr(audio, m):
                            fn = getattr(audio, m)
                            if callable(fn):
                                decoded = fn()
                                # 兼容：可能返回 (arr, sr)
                                if (
                                    isinstance(decoded, (tuple, list))
                                    and len(decoded) == 2
                                ):
                                    arr, sr = decoded
                                    return arr, int(sr)
                                if isinstance(decoded, dict):
                                    if "array" in decoded and (
                                        "sampling_rate" in decoded
                                        or "sample_rate" in decoded
                                    ):
                                        sr = decoded.get(
                                            "sampling_rate",
                                            decoded.get("sample_rate"),
                                        )
                                        arr = _unwrap_audio_array(
                                            decoded["array"]
                                        )
                                        return arr, int(sr)
                                # 兼容：可能只返回 samples（torch.Tensor / np.ndarray / list）
                                sr = getattr(
                                    audio, "_desired_sample_rate", None
                                )
                                if sr is None:
                                    md = getattr(audio, "metadata", None)
                                    # metadata 可能是 dict 或对象
                                    if isinstance(md, dict):
                                        sr = md.get(
                                            "sample_rate"
                                        ) or md.get("sampling_rate")
                                    else:
                                        sr = getattr(
                                            md, "sample_rate", None
                                        ) or getattr(
                                            md, "sampling_rate", None
                                        )
                                if sr is None:
                                    # 最后兜底：常见情况下 AISHELL 采样率为 16000
                                    sr = 16000
                                arr = _unwrap_audio_array(decoded)
                                return arr, int(sr)
                    # 最后给出可诊断信息
                    attrs = [
                        a for a in dir(audio) if not a.startswith("__")
                    ]
                    raise TypeError(
                        "检测到 datasets.features._torchcodec.AudioDecoder，但无法解码出音频。"
                        f"可用属性/方法示例(前50): {attrs[:50]}"
                    )

                if hasattr(audio, "decode"):
                    decoded = audio.decode()
                    # 可能返回 (waveform, sr) 或 dict
                    if (
                        isinstance(decoded, (tuple, list))
                        and len(decoded) == 2
                    ):
                        arr, sr = decoded
                        return arr, int(sr)
                    if isinstance(decoded, dict):
                        if "array" in decoded and (
                            "sampling_rate" in decoded
                            or "sample_rate" in decoded
                        ):
                            sr = decoded.get(
                                "sampling_rate", decoded.get("sample_rate")
                            )
                            return decoded["array"], int(sr)
                if hasattr(audio, "get_all_samples"):
                    decoded = audio.get_all_samples()
                    if (
                        isinstance(decoded, (tuple, list))
                        and len(decoded) == 2
                    ):
                        arr, sr = decoded
                        return arr, int(sr)
                # 有些实现把采样率暴露为属性
                if hasattr(audio, "sampling_rate") and hasattr(
                    audio, "__array__"
                ):
                    return np.asarray(audio), int(
                        getattr(audio, "sampling_rate")
                    )

                # 落到这里说明我们识别不了该 wav 的形态
                raise TypeError(
                    f"不支持的 {k} 字段格式: type={type(audio)}"
                    + (
                        f", keys={list(audio.keys())}"
                        if isinstance(audio, dict)
                        else ""
                    )
                )
        raise KeyError(
            f"样本中缺少 audio 字段（可用 keys: {list(example.keys())[:50]}）"
        )
    audio = example["audio"]
    if (
        isinstance(audio, dict)
        and "array" in audio
        and "sampling_rate" in audio
    ):
        return audio["array"], int(audio["sampling_rate"])
    raise TypeError(
        f"不支持的 audio 格式: {type(audio)} {list(audio.keys()) if isinstance(audio, dict) else ''}"
    )


def _to_wav_16k_mono(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if x is None:
        raise ValueError("audio array is None")
    x = np.asarray(x)
    # (num_samples,) or 2D:
    # - datasets/torchaudio/torchcodec 可能返回 (num_samples, channels)
    # - 也常见返回 (channels, num_samples)（尤其是单声道时形如 (1, N)）
    if x.ndim == 2:
        # 启发式判断哪个维度是 channels：通常 channels 很小（1/2），samples 很大（>1e3）
        # - 若形如 (1, N)/(2, N)：认为是 (channels, num_samples)，沿 axis=0 做平均
        # - 否则认为是 (num_samples, channels)，沿 axis=1 做平均
        if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
            x = np.mean(x, axis=0)
        else:
            x = np.mean(x, axis=1)
    x = x.astype(np.float32, copy=False)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000).astype(
            np.float32, copy=False
        )
        sr = 16000
    # 极短音频保护：若出现长度异常（例如只有 1 个采样点），大概率是上游字段解析/维度判断出了问题
    if x.ndim != 1 or x.shape[0] < 160:  # 10ms @ 16kHz
        raise ValueError(
            f"decoded audio too short or wrong shape: shape={x.shape}, sr={sr}"
        )
    return x, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mozilla-foundation/common_voice_17_0",
        help="HF dataset name，例如 mozilla-foundation/common_voice_17_0",
    )
    parser.add_argument(
        "--lang", type=str, default="zh-CN", help="语言配置，例如 zh-CN"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="datasets 的 config 名称（可选）。Common Voice 用 zh-CN；其他数据集可能不需要。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="split，例如 train/validation/test",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出目录，例如 out/cv_zhcn_4x10",
    )
    parser.add_argument(
        "--num_speakers", type=int, default=4, help="说话人数量"
    )
    parser.add_argument(
        "--per_speaker", type=int, default=10, help="每个说话人采样条数"
    )
    parser.add_argument(
        "--max_scan",
        type=int,
        default=5000,
        help="最多扫描多少条流式样本以找到足够的说话人",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="某些数据集需要启用 trust_remote_code 才能正确加载",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="仅搜索数据集并退出（例如 --search aishell / --search chinese / --search mandarin）",
    )
    parser.add_argument(
        "--speaker_field",
        type=str,
        default=None,
        help="强制使用某个字段作为 speaker_id（例如 AISHELL 可能是 speaker/spk_id/SpeakerID 等）",
    )
    parser.add_argument(
        "--debug_keys",
        type=int,
        default=0,
        help="调试：打印前 N 条样本的 keys，帮助定位 speaker/audio 字段",
    )
    args = parser.parse_args()

    if args.search:
        api = HfApi()
        print(f"正在搜索 datasets: {args.search!r}（只展示前 30 个）")
        try:
            for i, ds in enumerate(api.list_datasets(search=args.search)):
                if i >= 30:
                    break
                # ds.id 是 dataset repo id
                print(ds.id)
        except Exception as e:
            raise RuntimeError(
                f"搜索失败（可能无法访问 Hub/镜像的 datasets 列表 API）: {e}"
            )
        return

    os.makedirs(args.out_dir, exist_ok=True)
    wav_dir = os.path.join(args.out_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")

    def _try_load(dataset_name: str):
        # 兼容：有些数据集需要 config（如 Common Voice 的 zh-CN），有些不需要
        config = args.config
        if config is None:
            # 允许用户用 --lang default/none 来显式表示“不要 config”
            if args.lang and args.lang.lower() not in (
                "default",
                "none",
                "",
            ):
                config = args.lang

        if config:
            try:
                return load_dataset(
                    dataset_name,
                    config,
                    split=args.split,
                    streaming=True,
                    trust_remote_code=args.trust_remote_code,
                )
            except (TypeError, ValueError, DatasetNotFoundError):
                pass

        return load_dataset(
            dataset_name,
            split=args.split,
            streaming=True,
            trust_remote_code=args.trust_remote_code,
        )

    ds = None
    tried = []
    # 1) 先尝试用户指定的版本
    try:
        ds = _try_load(args.dataset)
    except (EmptyDatasetError, DatasetNotFoundError) as e:
        tried.append((args.dataset, str(e)))

    # 2) 常见 Common Voice 版本回退（有些环境的镜像/索引可能缺某些版本的数据文件）
    if ds is None and args.dataset.startswith(
        "mozilla-foundation/common_voice_"
    ):
        for ver in [
            "17_0",
            "16_1",
            "15_0",
            "14_0",
            "13_0",
            "12_0",
            "11_0",
        ]:
            candidate = f"mozilla-foundation/common_voice_{ver}"
            if candidate == args.dataset:
                continue
            try:
                ds = _try_load(candidate)
                args.dataset = candidate
                break
            except (EmptyDatasetError, DatasetNotFoundError) as e:
                tried.append((candidate, str(e)))

    # 3) 仍失败：尝试直接在 repo 内按文件模式加载（parquet/json/csv），只取 zh-CN + split 的分片
    if ds is None:
        fs = HfFileSystem()
        base = f"hf://datasets/{args.dataset}"

        def _pick_files(ext: str):
            files = fs.glob(f"{base}/**/*.{ext}")
            # 尽量过滤出包含语言与 split 的分片（Common Voice 常见命名/目录结构）
            lang = (
                args.lang.lower()
                if args.lang
                and args.lang.lower() not in ("default", "none", "")
                else ""
            )
            split = args.split.lower()
            filtered = []
            for p in files:
                pl = p.lower()
                if (not lang or lang in pl) and split in pl:
                    filtered.append(p)
            # 若过滤过严导致空，则退化到只过滤语言
            if not filtered:
                for p in files:
                    if not lang or lang in p.lower():
                        filtered.append(p)
            return sorted(filtered)

        parquet_files = _pick_files("parquet")
        json_files = _pick_files("json")
        csv_files = _pick_files("csv")

        if parquet_files:
            # parquet builder 的 split 名默认是 train
            ds = load_dataset(
                "parquet",
                data_files=parquet_files,
                split="train",
                streaming=True,
            )
        elif json_files:
            ds = load_dataset(
                "json",
                data_files=json_files,
                split="train",
                streaming=True,
            )
        elif csv_files:
            ds = load_dataset(
                "csv",
                data_files=csv_files,
                split="train",
                streaming=True,
            )
        else:
            msg = "\n".join(
                [f"- {name}: {err}" for name, err in tried[-5:]]
            )
            raise RuntimeError(
                f"无法以 streaming 方式加载数据集：{args.dataset}（lang={args.lang}, split={args.split}）。\n"
                f"已尝试的版本/错误（最后 5 条）：\n{msg}\n"
                f"也没有在仓库中找到 parquet/json/csv 数据文件。"
            )

    # 先收集：每个 speaker 保留前 per_speaker 条
    picked = defaultdict(list)  # speaker_id -> [example, ...]
    scanned = 0

    def _parse_aishell_speaker_id(s: str):
        """
        尝试从 AISHELL 常见 utterance id / 路径中解析 speaker id：
        - BAC009S0002W0122 -> BAC009S0002
        - .../S0002/xxx.wav -> S0002
        """
        if not s:
            return None
        # 1) 从 utterance id 中抓取形如 XXX000S0000 的 speaker
        m = re.search(r"([A-Za-z]{3}\d{3}S\d{4})", s)
        if m:
            return m.group(1)
        # 2) 从路径中抓取目录名 S0002
        m = re.search(r"(S\d{4})", s)
        if m:
            return m.group(1)
        return None

    def _extract_speaker_id(ex: dict):
        if args.speaker_field:
            v = ex.get(args.speaker_field)
            if v:
                v = str(v)
                # 若用户指定的是 __key__/utt_id 之类，优先尝试解析出 speaker 子串
                sid = _parse_aishell_speaker_id(v)
                return sid or v

        # 常见字段候选
        for k in [
            "client_id",
            "speaker_id",
            "speaker",
            "spk_id",
            "spkid",
            "spk",
            "speakerid",
            "user_id",
            "client",
        ]:
            v = ex.get(k)
            if v:
                return str(v)

        # 尝试从 utterance/id/path 等字段解析
        for k in [
            "utt_id",
            "utterance_id",
            "utterance",
            "id",
            "file",
            "path",
            "audio_path",
            "__key__",
            "__url__",
        ]:
            v = ex.get(k)
            if isinstance(v, str) and v:
                sid = _parse_aishell_speaker_id(v)
                if sid:
                    return sid

        # 尝试从 audio dict 的 path 解析
        a = ex.get("audio")
        if isinstance(a, dict):
            p = a.get("path") or a.get("filename")
            if isinstance(p, str) and p:
                sid = _parse_aishell_speaker_id(p)
                if sid:
                    return sid

        return None

    for ex in ds:
        scanned += 1
        if scanned > args.max_scan:
            break

        if args.debug_keys and scanned <= args.debug_keys:
            keys = list(ex.keys())
            print(f"[debug] sample#{scanned} keys={keys}")
            if "wav" in ex:
                w = ex["wav"]
                if isinstance(w, dict):
                    print(
                        f"[debug] sample#{scanned} wav=dict keys={list(w.keys())}"
                    )
                else:
                    print(f"[debug] sample#{scanned} wav=type={type(w)}")

        speaker_id = _extract_speaker_id(ex)
        if not speaker_id:
            continue
        if (
            speaker_id in picked
            and len(picked[speaker_id]) >= args.per_speaker
        ):
            continue

        # 如果还没凑够 speakers，就允许加入新 speaker；否则只补齐已选 speaker
        if speaker_id not in picked and len(picked) >= args.num_speakers:
            continue

        picked[speaker_id].append(ex)

        done = len(picked) >= args.num_speakers and all(
            len(v) >= args.per_speaker for v in picked.values()
        )
        if done:
            break

    # 过滤掉不足 per_speaker 的 speaker（极端情况下会发生）
    picked = {
        k: v for k, v in picked.items() if len(v) >= args.per_speaker
    }
    # 若超过 num_speakers，则截断
    speaker_ids = list(picked.keys())[: args.num_speakers]

    if len(speaker_ids) < args.num_speakers:
        raise RuntimeError(
            f"只找到 {len(speaker_ids)}/{args.num_speakers} 个满足每人 {args.per_speaker} 条的说话人。"
            f"可尝试增大 --max_scan 或换 split/dataset 版本。"
        )

    # 写出 wav + manifest
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for spk in speaker_ids:
            for i, ex in enumerate(picked[spk][: args.per_speaker]):
                audio, sr = _get_audio(ex)
                audio16, sr16 = _to_wav_16k_mono(audio, sr)
                out_wav = os.path.join(wav_dir, f"{spk}_{i:03d}.wav")
                sf.write(out_wav, audio16, sr16)

                text = ex.get("sentence") or ex.get("text") or ""
                row = {
                    "audio_filepath": out_wav,
                    "speaker_id": spk,
                    "text": text,
                    "source": args.dataset,
                    "lang": args.lang,
                    "split": args.split,
                    "sampling_rate": sr16,
                    "num_samples": int(audio16.shape[0]),
                    "duration": float(audio16.shape[0] / sr16),
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"完成：{args.num_speakers} speakers * {args.per_speaker} 条，共 {args.num_speakers * args.per_speaker} 条"
    )
    print(f"输出目录: {args.out_dir}")
    print(f"manifest: {manifest_path}")
    print(f"扫描样本数: {scanned}")


if __name__ == "__main__":
    main()
