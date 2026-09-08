from __future__ import annotations

from core.config import load_config


def test_shared_config_contains_only_integrated_runtime_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        """{
          "runtime": {"device": "cpu", "allow_cpu_fallback": true},
          "separation": {"num_steps": 8},
          "benchmark": {"output_dir": "reports"},
          "cholimex": {"overlap_padding": 0.2}
        }"""
    )

    config = load_config(path)

    assert config.runtime_device == "cpu"
    assert config.separation_num_steps == 8
    assert config.cholimex_overlap_padding == 0.2


def test_shared_config_rejects_removed_crawler_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"source": {"youtube_only": true}}')

    try:
        load_config(path)
    except ValueError as error:
        assert "unknown config key" in str(error)
    else:
        raise AssertionError("crawler configuration must not be accepted")
