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
├─ orchestrator.py      # 总控流水线脚本
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

编辑 `orchestrator_config.json`（支持 JSON 注释）:
- `paths`: 各阶段目录设置  
- `audio_split`: 分段参数  
- `labeling`: 标注模型、阈值、设备、最大样本数  
- `training`: 训练超参、early stopping  
- `iteration`: 循环控制（once、skip_manifest、stop_after_labels 等）  
- `speakers`: 多说话人映射（可选）  
- `logging`: 日志输出设置  

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
