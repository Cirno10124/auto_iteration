#!/usr/bin/env python3
import argparse
import os
import signal
import sys

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd

from auto_iteration.speaker_separator import SpeakerSeparator

# 全局标志变量，用于控制循环退出
should_exit = False


def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    global should_exit
    print("\n收到退出信号 (Ctrl+C)，正在退出...")
    should_exit = True


def collect_audio(chunk_duration, output_dir, sample_rate, channels):
    # 全局退出标志在 signal_handler 中声明，此处无需再次声明

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except (OSError, IOError) as e:
        print(f"错误：无法创建输出目录 '{output_dir}': {e}")
        sys.exit(1)

    separator = SpeakerSeparator()

    speaker_counters = {}
    index = 1

    while not should_exit:
        try:
            filename = os.path.join(output_dir, f"{index:02d}.wav")
            print(f"录制片段 {index:02d}，时长 {chunk_duration}秒 -> {filename}")

            # 录制音频
            try:
                recording = sd.rec(
                    int(chunk_duration * sample_rate),
                    samplerate=sample_rate,
                    channels=channels,
                )
                sd.wait()
            except Exception as e:
                print(f"错误：录制音频失败: {e}")
                print("跳过当前片段，继续下一个...")
                index += 1
                continue

            # 写入主音频文件
            try:
                wavfile.write(
                    filename,
                    sample_rate,
                    (recording * np.iinfo(np.int16).max).astype(np.int16),
                )
            except (IOError, OSError) as e:
                print(f"错误：无法写入文件 '{filename}': {e}")
                print("跳过当前片段，继续下一个...")
                index += 1
                continue

            # 说话人分离（源分离）
            sources = separator.separate_sources(filename)
            if sources:
                try:
                    separated = sources
                    for channel, audio in separated.items():
                        sep_dir = os.path.join(output_dir, "separated")
                        os.makedirs(sep_dir, exist_ok=True)
                        sep_filename = os.path.join(
                            sep_dir, f"{index:02d}_sep{channel}.wav"
                        )
                        wavfile.write(
                            sep_filename,
                            sample_rate,
                            (
                                audio.numpy() * np.iinfo(np.int16).max
                            ).astype(np.int16),
                        )
                except Exception as e:
                    print(f"警告：分离失败: {e}")

            # 说话人分离
            try:
                diarization = separator.diarize(filename)
            except Exception as e:
                print(f"错误：说话人分离失败: {e}")
                print("跳过当前片段的说话人分离，继续下一个...")
                index += 1
                continue

            # 处理分离结果并写入文件
            try:
                for turn, _, speaker in diarization.itertracks(
                    yield_label=True
                ):
                    speaker_folder = os.path.join(
                        output_dir, f"speaker_{int(speaker)+1:02d}"
                    )
                    try:
                        os.makedirs(speaker_folder, exist_ok=True)
                    except (OSError, IOError) as e:
                        print(f"警告：无法创建说话人文件夹 '{speaker_folder}': {e}")
                        continue

                    count = speaker_counters.get(speaker, 0) + 1
                    speaker_counters[speaker] = count
                    start_sample = int(turn.start * sample_rate)
                    end_sample = int(turn.end * sample_rate)
                    segment = recording[start_sample:end_sample]

                    segment_filename = os.path.join(
                        speaker_folder, f"{count:02d}.wav"
                    )
                    try:
                        wavfile.write(
                            segment_filename,
                            sample_rate,
                            (segment * np.iinfo(np.int16).max).astype(
                                np.int16
                            ),
                        )
                    except (IOError, OSError) as e:
                        print(f"警告：无法写入说话人片段文件 '{segment_filename}': {e}")
                        continue
            except Exception as e:
                print(f"错误：处理说话人分离结果时出错: {e}")
                print("继续下一个片段...")

            index += 1

        except KeyboardInterrupt:
            # 额外的 KeyboardInterrupt 处理（虽然已经有信号处理器）
            print("\n收到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"未预期的错误: {e}")
            print("继续下一个片段...")
            index += 1
            continue

    print(f"\n录制结束。共录制 {index - 1} 个片段。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Microphone audio collector"
    )
    parser.add_argument(
        "--chunk_duration",
        type=int,
        default=30,
        help="Chunk duration in seconds",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="audio_chunks",
        help="Directory to save audio chunks",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Audio sample rate"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of audio channels"
    )
    args = parser.parse_args()
    collect_audio(
        args.chunk_duration,
        args.output_dir,
        args.sample_rate,
        args.channels,
    )
