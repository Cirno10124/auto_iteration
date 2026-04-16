import os

import pytest

from orchestrator import load_config  # noqa: E402
from orchestrator_core.config_loader import (  # noqa: E402
    apply_cli_overrides,
    normalize_config_types,
)
from orchestrator_core.context import build_speaker_list  # noqa: E402
from orchestrator_core.paths import PROJECT_ROOT  # noqa: E402

pytestmark = pytest.mark.unit


def test_load_config_supports_jsonc_and_trailing_commas(tmp_path):
    """配置加载应支持 JSONC 注释与尾逗号。"""
    config_file = tmp_path / "config.jsonc"
    config_file.write_text(
        """
{
  // line comment
  "paths": {
    "audio_dir": "audio", // trailing comment
  },
  "iteration": {
    "once": true,
  },
}
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(str(config_file))
    assert cfg["paths"]["audio_dir"] == "audio"
    assert cfg["iteration"]["once"] is True


def test_load_config_invalid_json_raises_decode_error(tmp_path):
    """非法配置内容应触发解析异常。"""
    config_file = tmp_path / "bad.jsonc"
    config_file.write_text('{"paths": {"audio_dir": "x",}', encoding="utf-8")

    with pytest.raises(Exception):
        load_config(str(config_file))


def test_apply_cli_overrides_nested_and_types():
    """CLI 覆盖参数应正确应用嵌套键和类型转换。"""
    cfg = {"paths": {"audio_dir": "a"}, "iteration": {"once": False}}
    apply_cli_overrides(
        cfg,
        [
            "paths.audio_dir=/tmp/x",
            "iteration.once=true",
            "iteration.max_iterations=3",
        ],
    )
    assert cfg["paths"]["audio_dir"] == "/tmp/x"
    assert cfg["iteration"]["once"] is True
    assert cfg["iteration"]["max_iterations"] == 3


def test_normalize_config_types_iteration_bool_strings():
    """字符串布尔值应在校验前被标准化。"""
    cfg = {
        "iteration": {
            "once": "false",
            "skip_manifest": "1",
            "stop_after_labels": "0",
            "skip_labeling": "true",
        }
    }
    normalize_config_types(cfg)
    assert cfg["iteration"]["once"] is False
    assert cfg["iteration"]["skip_manifest"] is True
    assert cfg["iteration"]["stop_after_labels"] is False
    assert cfg["iteration"]["skip_labeling"] is True


def test_build_speaker_list_empty_mapping_uses_raw_dir():
    """未配置说话人映射时应回退到 raw_audio_dir 单说话人模式。"""
    cfg = {"paths": {"raw_audio_dir": "/raw"}, "speakers": {}}
    lst = build_speaker_list(cfg, None)
    assert lst == [(None, "/raw")]


def test_build_speaker_list_filters_by_arg():
    """传入 speakers 参数时应仅保留指定说话人。"""
    cfg = {
        "speakers": {"sp1": "/a", "sp2": "/b"},
    }
    lst = build_speaker_list(cfg, ["sp1"])
    assert lst == [("sp1", "/a")]


def test_project_root_is_repo_auto_iteration():
    """PROJECT_ROOT 应定位到 src 目录并包含 orchestrator_core。"""
    # orchestrator_core 已迁移到 src/orchestrator_core
    assert os.path.isfile(os.path.join(PROJECT_ROOT, "orchestrator_core", "cli.py"))
