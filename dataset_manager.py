import argparse
import csv
import logging
import os
import random


def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def read_existing_csv(csv_path):
    """读取现有的CSV文件，返回音频路径集合和完整数据列表"""
    if not os.path.exists(csv_path):
        return set(), []

    audio_paths = set()
    data = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = row["audio_filepath"]
                text = row.get("text", "").strip()
                audio_paths.add(audio_path)
                data.append((audio_path, text))
    except Exception as e:
        logging.warning(f"读取现有CSV文件失败 {csv_path}: {e}")
        return set(), []
    return audio_paths, data


def write_csv(path, data, logger):
    """写入CSV文件
    data: 列表，每个元素是 (audio_path, text) 元组
    """
    written_count = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写入列：音频路径和文本内容
        writer.writerow(["audio_filepath", "text"])
        for audio_path, text in data:
            writer.writerow([audio_path, text])
            written_count += 1
    logger.info(f"写入 {path} 共 {written_count} 条记录")


def check_data_size(data_size, dataset_name, logger):
    """检查数据集大小，如果过小则警告"""
    if data_size == 0:
        logger.warning(f"警告：{dataset_name} 数据集为空（0条记录），这可能导致训练或评估失败")
    elif data_size < 10:
        logger.warning(
            f"警告：{dataset_name} 数据集数据量很少（仅{data_size}条记录），建议至少10条以上"
        )


def main():
    parser = argparse.ArgumentParser(description="数据管理：构建训练/验证集清单")
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="音频文件目录"
    )
    parser.add_argument(
        "--labels_dir", type=str, required=True, help="标注文件目录"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="输出清单目录"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="训练集比例"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="测试集比例"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("数据清单构建开始")
    logger.info("=" * 60)
    logger.info(f"音频目录: {args.audio_dir}")
    logger.info(f"标签目录: {args.labels_dir}")
    logger.info(f"输出目录: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_csv = os.path.join(args.output_dir, "train.csv")
    val_csv = os.path.join(args.output_dir, "val.csv")
    test_csv = os.path.join(args.output_dir, "test.csv")

    # 检查是否已存在清单文件（判断是否为第一次迭代）
    is_first_iteration = not (
        os.path.exists(train_csv)
        and os.path.exists(val_csv)
        and os.path.exists(test_csv)
    )

    if is_first_iteration:
        logger.info("首次迭代：进行初始数据集分割")
        # 首次迭代：扫描所有数据并分割
        pairs = []
        for root, _, files in os.walk(args.audio_dir):
            for fname in files:
                if not fname.lower().endswith((".wav", ".flac", ".mp3")):
                    continue
                audio_path = os.path.join(root, fname)
                rel_path = os.path.relpath(audio_path, args.audio_dir)
                base, _ = os.path.splitext(rel_path)
                label_rel = base + ".txt"
                label_path = os.path.join(args.labels_dir, label_rel)
                if not os.path.exists(label_path):
                    logger.warning(f"缺少标注文件 {label_path}, 跳过")
                    continue
                if os.path.getsize(label_path) == 0:
                    logger.warning(f"空标注文件 {label_path}, 跳过")
                    continue
                pairs.append((audio_path, label_path))

        if len(pairs) == 0:
            logger.error("未找到任何有效的音频-标签对，无法构建清单")
            return

        random.seed(args.seed)
        random.shuffle(pairs)
        total = len(pairs)
        # 计算各数据集大小，确保验证集和测试集至少包含1条
        train_size = int(total * args.train_ratio)
        test_size = int(total * args.test_ratio)
        val_size = total - train_size - test_size
        # 强制至少1条给验证集和测试集
        if val_size < 1:
            val_size = 1
        if test_size < 1:
            test_size = 1
        # 重新计算训练集大小保证总和
        train_size = total - val_size - test_size
        if train_size < 0:
            train_size = 0
        # 划分数据
        train_pairs = pairs[:train_size]
        val_pairs = pairs[train_size:train_size + val_size]
        test_pairs = pairs[train_size + val_size:]

        # 将 (audio_path, label_path) 转换为 (audio_path, text)
        def load_text_from_pairs(pair_list):
            result = []
            for audio_path, label_path in pair_list:
                try:
                    with open(label_path, "r", encoding="utf-8") as lf:
                        text = lf.read().strip()
                    result.append((audio_path, text))
                except Exception as e:
                    logger.warning(f"读取标签文件失败 {label_path}: {e}")
            return result

        train_data = load_text_from_pairs(train_pairs)
        val_data = load_text_from_pairs(val_pairs)
        test_data = load_text_from_pairs(test_pairs)

        # 检查数据量
        check_data_size(len(train_data), "训练集", logger)
        check_data_size(len(val_data), "验证集", logger)
        check_data_size(len(test_data), "测试集", logger)

        write_csv(train_csv, train_data, logger)
        write_csv(val_csv, val_data, logger)
        write_csv(test_csv, test_data, logger)

        logger.info(
            f"初始分割完成：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条，测试集 {len(test_data)} 条"
        )
    else:
        logger.info("非首次迭代：将新数据添加到训练集")
        # 非首次迭代：读取现有清单，找出新数据并添加到训练集
        existing_train_paths, existing_train_data = read_existing_csv(
            train_csv
        )
        existing_val_paths, existing_val_data = read_existing_csv(val_csv)
        existing_test_paths, existing_test_data = read_existing_csv(
            test_csv
        )

        all_existing_paths = (
            existing_train_paths | existing_val_paths | existing_test_paths
        )

        logger.info(
            f"现有数据：训练集 {len(existing_train_data)} 条，验证集 {len(existing_val_data)} 条，测试集 {len(existing_test_data)} 条"
        )

        # 扫描新数据
        new_pairs = []
        for root, _, files in os.walk(args.audio_dir):
            for fname in files:
                if not fname.lower().endswith((".wav", ".flac", ".mp3")):
                    continue
                audio_path = os.path.join(root, fname)
                if audio_path in all_existing_paths:
                    continue  # 跳过已存在的数据

                rel_path = os.path.relpath(audio_path, args.audio_dir)
                base, _ = os.path.splitext(rel_path)
                label_rel = base + ".txt"
                label_path = os.path.join(args.labels_dir, label_rel)
                if not os.path.exists(label_path):
                    logger.warning(f"缺少标注文件 {label_path}, 跳过")
                    continue
                if os.path.getsize(label_path) == 0:
                    logger.warning(f"空标注文件 {label_path}, 跳过")
                    continue
                new_pairs.append((audio_path, label_path))

        if len(new_pairs) == 0:
            logger.info("未发现新数据，保持现有清单不变")
            # 检查现有数据量
            check_data_size(len(existing_train_data), "训练集", logger)
            check_data_size(len(existing_val_data), "验证集", logger)
            check_data_size(len(existing_test_data), "测试集", logger)
            return

        logger.info(f"发现 {len(new_pairs)} 条新数据，将添加到训练集")

        # 将新数据添加到训练集
        updated_train_data = existing_train_data.copy()
        for audio_path, label_path in new_pairs:
            try:
                with open(label_path, "r", encoding="utf-8") as lf:
                    text = lf.read().strip()
                updated_train_data.append((audio_path, text))
            except Exception as e:
                logger.warning(f"读取新标签文件失败 {label_path}: {e}")
                continue

        # 检查更新后的数据量
        check_data_size(len(updated_train_data), "训练集", logger)
        check_data_size(len(existing_val_data), "验证集", logger)
        check_data_size(len(existing_test_data), "测试集", logger)

        # 写入更新后的清单（训练集更新，验证集和测试集保持不变）
        write_csv(train_csv, updated_train_data, logger)
        write_csv(val_csv, existing_val_data, logger)
        write_csv(test_csv, existing_test_data, logger)

        logger.info(
            f"更新完成：训练集 {len(updated_train_data)} 条（新增 {len(new_pairs)} 条），验证集 {len(existing_val_data)} 条，测试集 {len(existing_test_data)} 条"
        )

    logger.info("=" * 60)
    logger.info("数据清单构建完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
