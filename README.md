# 自动化迭代 (auto_iteration)

本模块提供一套端到端自动化迭代流水线，用于：

1. 从音频源获取原始音频并分离说话人存储
2. 对原始长音频进行 VAD 分段（split_audio.py）  
3. 调用 Whisper 模型(或训练后的模型)自动标注（labeler.py）  
4. 构建训练/验证/测试清单（dataset_manager.py）  
5. 对模型进行 LoRA 微调训练（train_lora.py）  
6. 评估模型性能并保存最佳权重（evaluator.py）  
7. 可选将模型转换为 GGML 格式（converter.py）  

所有步骤由 `orchestrator.py` 统一调用，可配置、可覆盖、可多说话人并行（需要硬件支持）。

---

## 目录结构
```
auto_iteration/
├─ split_audio.py       # 音频分段脚本
├─ labeler.py           # Whisper 自动标注脚本
├─ dataset_manager.py   # 清单构建脚本
├─ train_lora.py        # LoRA 微调训练脚本
├─ evaluator.py         # 模型评估脚本
├─ converter.py         # 模型转 GGML 脚本
├─ orchestrator.py      # 总控入口（薄封装，逻辑在 orchestrator_core/）
├─ orchestrator_core/   # 配置加载、步骤执行、主流程编排
├─ orchestrator_config.json  # 示例配置（可注释 JSONC）
└─ tests/               # 端到端测试脚本及测试数据
```

## 前置要求

- Python 3.8+（3.11版本最佳）  
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```
- CUDA GPU（训练必需）
- 可选：安装 `ffmpeg`、`sox` 等工具以支持更多音频格式

## 配置说明

优先使用环境分层配置（均支持 JSON 注释）：

- `orchestrator_config.dev.json`
- `orchestrator_config.test.json`
- `orchestrator_config.prod.json`

启动时可使用：

```bash
python orchestrator.py --env dev
python orchestrator.py --env test
python orchestrator.py --env prod
```

或显式指定配置文件（兼容旧用法）：

```bash
python orchestrator.py --config orchestrator_config.json
```

配置会在启动前做基础校验（必填项、类型、范围），不合法配置会被拦截并报错。

编辑配置文件时重点关注：
- `paths`: 各阶段目录设置  
- `audio_split`: 分段参数  
- `labeling`: 标注模型、阈值、设备、最大样本数  
- `training`: 训练超参、early stopping  
- `iteration`: 循环控制（once、skip_manifest、stop_after_labels 等）  
- `speakers`: 多说话人映射（可选）  
- `logging`: 日志输出设置  

敏感信息（如 HF token）建议通过环境变量注入，例如：

```bash
set HF_HUB_TOKEN=your_token   # Windows PowerShell 可用 $env:HF_HUB_TOKEN=\"...\" 方式
```

## 使用方法

1. **一次性运行**  
   ```bash
   python orchestrator.py --config orchestrator_config.json --override iteration.once=true
   ```
2. **持续迭代（多说话人）**  
   ```bash
   python orchestrator.py --config orchestrator_config.json \
     --speakers speaker_01 speaker_02
   ```
3. **端到端测试**  
   ```bash
   pytest -s tests/end_to_end_test.py
   ```

## 测试分层与 CI 策略

- `unit`：纯逻辑单元测试，无模型/网络依赖
- `integration_light`：轻量集成测试，基于 `tests/stubs` 跑主流程
- `heavy`：重型测试（真实模型、网络、大数据、GPU），CI 默认跳过
- 主 CI 必过项：`flake8`、`pytest -m "not heavy"`、GPU 脚本 smoke、`sign_tools` 的 `lint/test/build`

### 在 CI 中执行（默认）

```bash
pytest -m "not heavy" -q
```

CI 还会在 GitHub Actions 中生成覆盖率（XML 报告以 artifact 形式上传）。本地带覆盖率：

```bash
pytest -m "not heavy" -q --cov=. --cov-config=.coveragerc --cov-report=term-missing
```

### 本地执行全量测试（包含 heavy）

```bash
pytest -q
```

### 常见排查建议

- 若 CI 失败，先本地执行 `pytest -m "not heavy" -q` 复现
- 若 heavy 相关失败，优先检查模型下载、HF token、GPU 与数据路径
- 使用 `-k <关键字>` 聚焦单测，例如 `pytest -k orchestrator -q`

## GPU 容器化运行

前提：
- 已安装 Docker
- 已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- 宿主机可用 `nvidia-smi`

构建镜像：

```bash
docker compose -f docker-compose.gpu.yml build
```

运行（使用 `prod` 配置）：

```bash
docker compose -f docker-compose.gpu.yml up
```

注入 HF token（示例）：

```bash
set HF_HUB_TOKEN=your_token
docker compose -f docker-compose.gpu.yml up
```

启动前可先做 GPU 预检：

```bash
docker compose -f docker-compose.gpu.yml run --rm orchestrator-gpu python3 scripts/gpu_health_check.py
```

仅做驱动层检查（跳过 torch）：

```bash
docker compose -f docker-compose.gpu.yml run --rm orchestrator-gpu python3 scripts/gpu_health_check.py --skip-torch
```

## 依赖安全扫描

仓库已接入 `Security Scan` 工作流（`.github/workflows/security-scan.yml`），包含：

- Python：`pip-audit -r requirements.txt`
- Node（`sign_tools`）：`npm audit --json`
- 触发方式：PR、手动触发、每月定时（1号）

本地可执行：

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

```bash
cd sign_tools
npm ci
npm audit
```

## 脚本概要

- **audio_collector.py**：实时音频采集并进行说话人分离
- **split_audio.py**：基于 VAD 切分长音频到指定目录
- **labeler.py**：调用 Hugging Face Whisper 进行自动标注
- **dataset_manager.py**：扫描标注结果，生成 CSV 清单并支持抽样
- **train_lora.py**：使用 PEFT 和 LoRA 对模型微调，支持 early stopping
- **evaluator.py**：基于 `jiwer` 计算 WER/CER，并生成报告
- **converter.py**：将训练后的模型转为 GGML 格式

## 日志与模型管理

- 日志文件输出到配置指定目录  
- 控制台与文件使用统一格式，便于检索：`speaker=`、`iter=`、`step=`、`model=`（中控在切分/标注/训练等步骤会自动填充；子脚本需自行 `logging` 配置后才显示）

- 每轮训练会为该说话人保存 `best_model/xxx`，可作为下一轮增量训练输入
- `model_metadata.json` 会记录超参和训练信息，方便追踪

---

---

## 说话人聚类 (cluster_speakers)

`SpeakerSeparator` 提供 `cluster_speakers(audio_file: str, threshold: float = 0.75)` 方法：

- `audio_file`：音频文件路径
- `threshold`：距离阈值（余弦距离），越小聚类越保守
- 返回：字典 `{cluster_id: [(start, end), ...], ...}`

示例：

```python
from auto_iteration.speaker_separator import SpeakerSeparator

sep = SpeakerSeparator(device="cpu")
clusters = sep.cluster_speakers("path/to/audio.wav", threshold=0.6)
for cid, segments in clusters.items():
    print(f"簇{cid}: {segments}")
```

---

欢迎提 issue 和 PR，以优化功能和增强可用性。
