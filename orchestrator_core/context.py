"""说话人列表与路径上下文。"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from logging_utils import ensure_early_console_logging


def build_speaker_list(
    config: Dict[str, Any],
    speakers_arg: Optional[List[str]],
    logger: Any = None,
) -> List[Tuple[Optional[str], Any]]:
    """构造 [(speaker_id, raw_audio_dir), ...]；无 mapping 时为 [(None, raw_audio_dir)]。"""
    if logger is None:
        ensure_early_console_logging()
        logger = logging.getLogger("orchestrator.context")
    speakers_map = config.get("speakers", {})
    if speakers_arg:
        selected = set(speakers_arg)
        speakers_map = {
            k: v for k, v in speakers_map.items() if k in selected
        }
        missing = selected - speakers_map.keys()
        for m in missing:
            logger.warning(f"配置中不存在说话人 {m}，已忽略")
    if speakers_map:
        return list(speakers_map.items())
    return [(None, config.get("paths", {}).get("raw_audio_dir"))]
