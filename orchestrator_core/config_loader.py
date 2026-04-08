"""JSONC 配置加载与 CLI 覆盖。"""

import json
import logging
from typing import Any, Dict, List, Optional

from logging_utils import ensure_early_console_logging


def _config_logger(logger: Any = None) -> Any:
    if logger is not None:
        return logger
    ensure_early_console_logging()
    return logging.getLogger("orchestrator.config")


def load_config(config_path: str, logger: Any = None) -> Dict[str, Any]:
    """加载配置文件，支持 JSONC 格式（带注释的 JSON）"""

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        cleaned_lines = []
        in_string = False
        escape_next = False

        for line in lines:
            cleaned_line = ""
            i = 0
            while i < len(line):
                char = line[i]

                if escape_next:
                    cleaned_line += char
                    escape_next = False
                    i += 1
                    continue

                if char == "\\":
                    escape_next = True
                    cleaned_line += char
                    i += 1
                    continue

                if char == '"':
                    in_string = not in_string
                    cleaned_line += char
                    i += 1
                    continue

                if not in_string:
                    if i < len(line) - 1 and line[i : i + 2] == "//":
                        break
                    if i < len(line) - 1 and line[i : i + 2] == "/*":
                        j = line.find("*/", i + 2)
                        if j != -1:
                            i = j + 2
                            continue
                        else:
                            i += 2
                            continue
                    cleaned_line += char
                    i += 1
                else:
                    cleaned_line += char
                    i += 1

            cleaned_lines.append(cleaned_line)

        cleaned_content = "\n".join(cleaned_lines)

        def remove_block_comments(text: str) -> str:
            result = []
            i = 0
            in_string = False
            escape_next = False

            while i < len(text):
                char = text[i]

                if escape_next:
                    result.append(char)
                    escape_next = False
                    i += 1
                    continue

                if char == "\\":
                    escape_next = True
                    result.append(char)
                    i += 1
                    continue

                if char == '"':
                    in_string = not in_string
                    result.append(char)
                    i += 1
                    continue

                if (
                    not in_string
                    and i < len(text) - 1
                    and text[i : i + 2] == "/*"
                ):
                    j = text.find("*/", i + 2)
                    if j != -1:
                        i = j + 2
                        continue
                    else:
                        result.append(char)
                        i += 1
                        continue

                result.append(char)
                i += 1

            return "".join(result)

        cleaned_content = remove_block_comments(cleaned_content)

        def remove_trailing_commas(text: str) -> str:
            out = []
            i = 0
            in_string = False
            escape_next = False
            n = len(text)

            while i < n:
                ch = text[i]

                if escape_next:
                    out.append(ch)
                    escape_next = False
                    i += 1
                    continue

                if ch == "\\":
                    out.append(ch)
                    escape_next = True
                    i += 1
                    continue

                if ch == '"':
                    out.append(ch)
                    in_string = not in_string
                    i += 1
                    continue

                if not in_string and ch == ",":
                    j = i + 1
                    while j < n and text[j] in " \t\r\n":
                        j += 1
                    if j < n and text[j] in "}]":
                        i += 1
                        continue

                out.append(ch)
                i += 1

            return "".join(out)

        cleaned_content = remove_trailing_commas(cleaned_content)

        config = json.loads(cleaned_content)

        log = _config_logger(logger)
        log.info(f"成功加载配置文件: {config_path}")
        return config
    except FileNotFoundError:
        error_msg = f"配置文件不存在: {config_path}"
        _config_logger(logger).error(error_msg)
        raise
    except json.JSONDecodeError as e:
        error_msg = f"配置文件格式错误: {e}"
        _config_logger(logger).error(error_msg)
        raise
    except Exception as e:
        error_msg = f"加载配置文件时出错: {e}"
        _config_logger(logger).error(error_msg)
        raise


def apply_cli_overrides(
    config: Dict[str, Any],
    overrides: Optional[List[str]],
    logger: Any = None,
) -> None:
    """将 key=value 覆盖写入 config（原地修改）。"""
    if not overrides:
        return
    log = _config_logger(logger)
    for override in overrides:
        if "=" not in override:
            log.warning(
                f"忽略无效的覆盖项 '{override}'（格式应为 key=value）"
            )
            continue
        key, value = override.split("=", 1)
        keys = key.split(".")
        target = config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        try:
            if value.lower() == "true":
                value = True  # type: ignore[assignment]
            elif value.lower() == "false":
                value = False  # type: ignore[assignment]
            elif value.isdigit():
                value = int(value)  # type: ignore[assignment]
            else:
                try:
                    value = float(value)  # type: ignore[assignment]
                except ValueError:
                    pass
        except Exception:
            pass
        target[keys[-1]] = value
        log.info(f"已覆盖配置项 {key} = {value}")
