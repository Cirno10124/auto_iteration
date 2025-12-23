# Speaker Separator 独立环境（最小化依赖）

此目录用于将 `auto_iteration/speaker_separator.py` 相关依赖从 auto_iteration 的训练/评测环境中**分离**出来，方便在单独的 venv/conda 环境中运行说话人分离与简单聚类。

## 安装

- **建议 Python**：3.11

1) 创建并激活虚拟环境（示例）

```bash
python -m venv .venv
source .venv/bin/activate
```

2) 安装依赖

```bash
pip install -r requirements.txt
```

> 说明：`torch/torchaudio` 在不同平台（CPU/CUDA/ROCm）需要选择对应 wheel。
> 若你在服务器上有 CUDA，请优先按 PyTorch 官方说明安装对应版本，然后再安装本目录其余依赖。

3) 安装本包（推荐）

```bash
pip install -e .
```

## 运行示例

安装后可直接使用命令行入口：

```bash
speaker-separator --audio /path/to/audio.wav --out_dir ./out_speakers
```

或用模块方式运行：

```bash
python -m speaker_separator_env --audio /path/to/audio.wav --out_dir ./out_speakers
```

本目录也保留了兼容脚本 `run_speaker_separator.py`（等价于上面两种方式）：

```bash
python run_speaker_separator.py --audio /path/to/audio.wav --out_dir ./out_speakers
```

可选聚类（基于 embedding + 层次聚类）：

```bash
python run_speaker_separator.py --audio /path/to/audio.wav --cluster --threshold 0.75
```

## 配置

本包会按以下优先级读取配置（JSON）：

- 1) 显式参数（`SpeakerSeparator(...)`）
- 2) 环境变量 `SPEAKER_SEPARATOR_CONFIG` 指向的 JSON 文件
- 3) 包目录内的 `speaker_separator_config.json`
- 4) 当前工作目录下的 `speaker_separator_config.json`

- `model_revision`: `pyannote/speaker-diarization@...`
- `hf_token`: Hugging Face token（访问受限模型需要）
- `device`: `cpu` / `cuda`
- `snr_threshold`: 降噪触发阈值


