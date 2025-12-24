"""
兼容模块（legacy shim）

你已经把 `speaker_separator_env` 作为一个独立包（src layout）引入后，
推荐改用：

  from speaker_separator_env import SpeakerSeparator, DummyEmbedder

这个文件仅为兼容旧代码 `from speaker_separator import ...` 的导入路径。
"""

from speaker_separator_env.speaker_separator import (  # noqa: F401
    DummyEmbedder,
    SpeakerSeparator,
)
