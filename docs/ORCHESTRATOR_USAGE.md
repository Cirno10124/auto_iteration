# orchestrator.py 用法说明

`orchestrator.py` 是自动化迭代总控脚本，用于音频切分、自动标注、清单构建、训练、评估及模型转换。

## 依赖环境

- Python 3.11
- 已安装依赖：见项目根目录 `requirements.txt`。

## 运行方式

```bash
python orchestrator.py [选项]
```

## 参数说明

| 参数                  | 类型     | 默认值      | 含义                                                                                 |
|-----------------------|----------|-------------|--------------------------------------------------------------------------------------|
| `--raw_audio_dir`     | str      | None        | 原始长音频目录，若提供则先进行 VAD 切分。                                              |
| `--split_script`      | str      | 同脚本目录  | VAD 切分脚本路径（`src/scripts/split_audio.py`）。                                        |
| `--audio_dir`         | str      | `audio_chunks` | 音频切片输出目录。                                                                    |
| `--labels_dir`        | str      | `labels`    | 自动标注结果输出目录。                                                                |
| `--manifest_dir`      | str      | `manifests` | 数据清单（manifest）输出目录。                                                        |
| `--model_dir`         | str      | `out/model` | 模型训练输出目录。                                                                    |
| `--ggml_dir`          | str      | `ggml_model`| GGML 模型输出目录。                                                                  |
| `--annotation_ratio`  | float    | 0.0         | 从已标注数据中抽取比例用于人工标注（伪标签抽样）。取值区间 (0,1]，0 表示不启用。         |
| `--skip_manifest`     | flag     | False       | 跳过清单构建，仅进行标注和抽样操作。                                                  |
| `--stop_after_labels` | flag     | False       | 仅运行标注和清单构建后停止执行。                                                      |
| `--test_size`         | int      | 0           | 测试模式时每个数据 split 限制的样本数，0 表示不截断。                                   |
| `--interval`          | int      | 86400      | 迭代周期（秒），默认一天。                                                            |
| `--once`              | flag     | False       | 仅执行一次迭代后退出。                                                                |

## 使用示例

```bash
python orchestrator.py \
  --raw_audio_dir /data/raw_audio \
  --split_script /path/to/src/scripts/split_audio.py \
  --audio_dir chunks \
  --labels_dir labels \
  --manifest_dir manifests \
  --model_dir out/model \
  --ggml_dir ggml_model \
  --annotation_ratio 0.1 \
  --interval 86400 
```

以上示例中，脚本将：
1. 对 `/data/raw_audio` 中的长音频进行 VAD 切分；
2. 在 `chunks` 目录下保存切片；
3. 生成伪标签并输出到 `labels`；
4. 构建 `manifests` 清单；
5. 使用 10% 伪标签样本进行人工标注；
6. 每 24 小时执行一次迭代。






