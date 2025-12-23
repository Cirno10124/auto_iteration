#!/usr/bin/env python3
import argparse
import json

from .speaker_separator import SpeakerSeparator


def main():
    parser = argparse.ArgumentParser(
        description="Run pyannote speaker diarization (and optional clustering)."
    )
    parser.add_argument("--audio", type=str, required=True, help="输入音频路径")
    parser.add_argument("--out_dir", type=str, default=None, help="保存分段 wav 的输出目录（默认不保存）")
    parser.add_argument("--cluster", action="store_true", help="是否额外做一次聚类（embedding + 层次聚类）")
    parser.add_argument("--threshold", type=float, default=0.75, help="聚类距离阈值（cosine distance）")
    args = parser.parse_args()

    sep = SpeakerSeparator()
    ann = sep.diarize(args.audio)

    print("[diarization] segments:")
    for turn, _, speaker in ann.itertracks(yield_label=True):
        print(f"{float(turn.start):.3f}\t{float(turn.end):.3f}\t{speaker}")

    if args.out_dir:
        out_dir = sep.save_speaker_segments(args.audio, out_dir=args.out_dir)
        print(f"[save] wrote speaker segments to: {out_dir}")

    if args.cluster:
        clusters = sep.cluster_speakers(args.audio, threshold=args.threshold)
        print("[cluster] clusters (json):")
        print(json.dumps(clusters, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


