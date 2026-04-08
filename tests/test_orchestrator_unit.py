import os
import sys

import pytest

sys.path.insert(0, os.getcwd())
from orchestrator import load_config
from orchestrator_core.config_loader import apply_cli_overrides
from orchestrator_core.context import build_speaker_list
from orchestrator_core.paths import PROJECT_ROOT

pytestmark = pytest.mark.unit


def test_load_config_supports_jsonc_and_trailing_commas(tmp_path):
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
    config_file = tmp_path / "bad.jsonc"
    config_file.write_text('{"paths": {"audio_dir": "x",}', encoding="utf-8")

    with pytest.raises(Exception):
        load_config(str(config_file))


def test_apply_cli_overrides_nested_and_types():
    cfg = {"paths": {"audio_dir": "a"}, "iteration": {"once": False}}
    apply_cli_overrides(
        cfg,
        ["paths.audio_dir=/tmp/x", "iteration.once=true", "iteration.max_iterations=3"],
    )
    assert cfg["paths"]["audio_dir"] == "/tmp/x"
    assert cfg["iteration"]["once"] is True
    assert cfg["iteration"]["max_iterations"] == 3


def test_build_speaker_list_empty_mapping_uses_raw_dir():
    cfg = {"paths": {"raw_audio_dir": "/raw"}, "speakers": {}}
    lst = build_speaker_list(cfg, None)
    assert lst == [(None, "/raw")]


def test_build_speaker_list_filters_by_arg():
    cfg = {
        "speakers": {"sp1": "/a", "sp2": "/b"},
    }
    lst = build_speaker_list(cfg, ["sp1"])
    assert lst == [("sp1", "/a")]


def test_project_root_is_repo_auto_iteration():
    # orchestrator_core 位于 auto_iteration/orchestrator_core
    assert os.path.isfile(os.path.join(PROJECT_ROOT, "orchestrator.py"))
