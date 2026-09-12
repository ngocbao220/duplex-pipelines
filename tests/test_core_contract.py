import json
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from core.orchestration.contract import run_sample, validate_conversation_collection, validate_stereo
from core.orchestration import worker
from core.orchestration.process import stream_process


def _wav(path, frames=1600):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(frames), 16000)
    return path


def _stereo_wav(path, frames=1600):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((frames, 2)), 16000)
    return path


def test_core_contract_requires_one_timeline_aligned_stereo_output(tmp_path):
    mixture = _wav(tmp_path / "mixture.wav")
    stereo = _stereo_wav(tmp_path / "audio.stereo.wav")
    assert validate_stereo(mixture, stereo) == 0.1
    _stereo_wav(stereo, frames=800)
    try:
        validate_stereo(mixture, stereo)
    except ValueError as error:
        assert "timeline" in str(error)
    else:
        raise AssertionError("short track must fail validation")


def test_duplexchat_collection_contract_allows_zero_conversations(tmp_path):
    manifest = tmp_path / "conversations" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text('{"conversation_count": 0, "conversations": []}')

    assert validate_conversation_collection(manifest) == 0


def test_duplexchat_worker_returns_a_conversation_collection(monkeypatch, tmp_path):
    captured = {}
    runner_module = types.ModuleType("duplexchat.runner")

    def run_single_audio(*_args, **kwargs):
        captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True)
        return {"collection": tmp_path / "output" / "conversations" / "manifest.json", "devices": ["cpu"]}

    runner_module.run_single_audio = run_single_audio
    package = types.ModuleType("duplexchat")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "duplexchat", package)
    monkeypatch.setitem(sys.modules, "duplexchat.runner", runner_module)
    collection, metadata = worker.duplexchat(tmp_path / "input.wav", tmp_path / "output", {"debug": True})

    assert collection is None
    assert metadata == {"collection": str(tmp_path / "output" / "conversations" / "manifest.json"), "devices": ["cpu"]}


def test_failed_sample_emits_one_concise_console_error_and_persists_traceback(tmp_path, capsys):
    mixture = _wav(tmp_path / "mixture.wav")

    def failing_adapter(source, output, config):
        raise FileNotFoundError("checkpoint missing")

    result = run_sample("vilier", {"key": "sample", "mixture": str(mixture)}, tmp_path / "output", {}, "code", failing_adapter)

    captured = capsys.readouterr().out
    assert result["status"] == "failed"
    assert "FileNotFoundError: checkpoint missing" in result["traceback"]
    assert " - vilier - [ERROR] - Sample sample failed: FileNotFoundError: checkpoint missing" in captured
    assert "Traceback" not in captured


def test_worker_forces_headless_matplotlib_backend_over_notebook_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    log = tmp_path / "worker.log"

    assert stream_process(
        [sys.executable, "-c", "import os; print(os.environ['MPLBACKEND'])"], tmp_path, log
    ) == 0

    assert log.read_text(encoding="utf-8").strip() == "Agg"
