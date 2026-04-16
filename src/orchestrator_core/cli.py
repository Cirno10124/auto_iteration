"""命令行入口。"""

import argparse
import logging
import os
from typing import Optional

from logging_utils import ensure_early_console_logging, setup_logging

from orchestrator_core.config_loader import apply_cli_overrides, load_config
from orchestrator_core.config_validation import validate_config
from orchestrator_core.context import build_speaker_list
from orchestrator_core.paths import PROJECT_ROOT
from orchestrator_core.pipeline import run_orchestrator_loop


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="自动化迭代总控脚本")
    parser.add_argument(
        "--config", type=str, required=False, help="配置文件路径（JSON格式）"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        choices=["dev", "test", "prod"],
        help="环境配置名；当未传 --config 时，默认读取 orchestrator_config.<env>.json",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="覆盖配置项，格式: key1=value1 key2=value2",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="*",
        help="指定要训练的说话人ID列表，空或未指定则训练 config中所有 speakers 或原始目录",
    )
    args = parser.parse_args(argv)

    ensure_early_console_logging()
    root_log = logging.getLogger()

    config_path = args.config or os.path.join(
        PROJECT_ROOT, f"orchestrator_config.{args.env}.json"
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}（可通过 --config 指定）"
        )

    config = load_config(config_path, logger=root_log)

    apply_cli_overrides(config, args.override, logger=root_log)
    errors = validate_config(config)
    if errors:
        for err in errors:
            root_log.error("配置校验失败: %s", err)
        raise ValueError("配置校验失败，请修正后重试。")

    log_config = config.get("logging", {})
    log_dir = log_config.get("log_dir", "logs")
    script_dir = PROJECT_ROOT
    log_dir = os.path.join(script_dir, log_dir)
    log_level = log_config.get("log_level", "INFO")
    log_file_prefix = log_config.get("log_file_prefix", "orchestrator")
    logger = setup_logging(log_dir, log_level, log_file_prefix)

    logger.info(f"成功加载配置文件: {config_path}")

    speaker_list = build_speaker_list(config, args.speakers, logger=logger)

    run_orchestrator_loop(logger, config, config_path, speaker_list)


if __name__ == "__main__":
    main()
