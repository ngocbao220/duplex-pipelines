"""Contract tests for the standalone separation-model smoke runner."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "test_separation_models.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_separation_models", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_selection_accepts_all_and_rejects_unknown_models():
    module = _load_module()

    assert set(module.select_models("all")) == set(module.MODELS)
    assert module.select_models("sepformer-wsj02mix,sepreformer-base-wsj0") == [
        "sepformer-wsj02mix",
        "sepreformer-base-wsj0",
    ]

    try:
        module.select_models("not-a-model")
    except ValueError as exc:
        assert "Unknown model" in str(exc)
    else:
        raise AssertionError("unknown model must be rejected")


def test_normalize_sources_preserves_source_count_and_length():
    module = _load_module()
    sources = np.arange(30, dtype=np.float32).reshape(3, 10)

    normalized = module.normalize_sources(sources, expected_length=16)

    assert normalized.shape == (3, 16)
    assert np.array_equal(normalized[:, :10], sources)
    assert np.all(normalized[:, 10:] == 0)


def test_complete_report_writes_two_track_aliases_only_for_two_source_model(tmp_path):
    module = _load_module()
    paths = module.write_outputs(
        tmp_path,
        np.zeros(16, dtype=np.float32),
        16_000,
        np.zeros((3, 16), dtype=np.float32),
    )
    report = module.build_report("rahma89-voice-separation-model", paths, 3, 0.2)
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")

    assert not (tmp_path / "speakerA.wav").exists()
    assert not (tmp_path / "speakerB.wav").exists()
    assert len(list(tmp_path.glob("source_*.wav"))) == 3
    assert report["two_track_output"] is False
