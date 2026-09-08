from __future__ import annotations

import tomllib
from pathlib import Path

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


def test_sommelier_runtime_declares_vendor_import_dependencies():
    """The original entry point imports these before processing CLI flags."""
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pipeline" / "sommelier" / "pyproject.toml").read_text())
    dependencies = set(config["project"]["dependencies"])
    names = {dependency.split("[", 1)[0].split("=", 1)[0] for dependency in dependencies}
    assert {"openai", "onnxruntime", "faster-whisper", "whisperx"} <= names


def test_sommelier_runtime_pins_upstream_torch_compatibility_set():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pipeline" / "sommelier" / "pyproject.toml").read_text())
    dependencies = set(config["project"]["dependencies"])
    assert {"torch==2.7.1", "torchaudio==2.7.1", "torchmetrics==1.7.4"} <= dependencies


def test_sommelier_runtime_pins_hub_version_that_accepts_use_auth_token():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pipeline" / "sommelier" / "pyproject.toml").read_text())
    assert "huggingface-hub==0.33.4" in set(config["project"]["dependencies"])


def test_sommelier_runtime_pins_speechbrain_before_the_token_api_change():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pipeline" / "sommelier" / "pyproject.toml").read_text())
    assert "speechbrain==0.5.16" in set(config["project"]["dependencies"])


def test_sommelier_defers_salm_import_when_asr_moe_is_disabled():
    root = Path(__file__).resolve().parents[1]
    source = (root / "pipeline" / "sommelier" / "vendor" / "podcast_pipeline" / "main_original_ASR_MoE.py").read_text()
    import_statement = "from nemo.collections.speechlm2.models import SALM"
    assert source.count(import_statement) == 1
    assert source.index(import_statement) > source.index("if args.ASRMoE:")


def test_sommelier_lightning_load_compatibility_wrapper_accepts_weights_only():
    root = Path(__file__).resolve().parents[1]
    source = (root / "pipeline" / "sommelier" / "vendor" / "podcast_pipeline" / "main_original_ASR_MoE.py").read_text()
    assert "def _patched_load(path_or_url: Union[IO, str, Path], map_location=None, weights_only=None)" in source
    assert "torch.load(path_or_url, map_location=map_location, weights_only=False)" in source
