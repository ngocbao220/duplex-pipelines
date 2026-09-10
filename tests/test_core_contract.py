import json
import sys

import numpy as np
import soundfile as sf

from core.orchestration.contract import run_sample, validate_conversation_collection, validate_stereo
from core.orchestration.process import stream_process
from core.orchestration.report import comparison_report


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


def test_conversation_collection_requires_one_stereo_output_per_dialogue(tmp_path):
    mixture = _wav(tmp_path / "conversation" / "mixture.wav", frames=3200)
    stereo = _stereo_wav(tmp_path / "conversation" / "audio.stereo.wav", frames=3200)
    collection = [{"mixture": str(mixture), "stereo": str(stereo)}]

    assert validate_conversation_collection(collection) == 0.2
    _stereo_wav(stereo, frames=1600)
    try:
        validate_conversation_collection(collection)
    except ValueError as error:
        assert "timeline" in str(error)
    else:
        raise AssertionError("short conversation track must fail validation")


def test_collection_run_does_not_create_invalid_full_input_tracks(tmp_path):
    source = _wav(tmp_path / "source.wav")

    def collection_adapter(_source, output, _config):
        mixture = _wav(output / "conversations" / "conversation_00000" / "mixture.wav")
        stereo = _stereo_wav(output / "conversations" / "conversation_00000" / "audio.stereo.wav")
        return None, {"output_kind": "conversation_collection", "conversations": [
            {"mixture": str(mixture), "stereo": str(stereo)}
        ]}

    result = run_sample("duplexchat", {"key": "sample", "mixture": str(source)}, tmp_path / "output", {}, "code", collection_adapter)

    assert result["status"] == "complete"
    assert result["audio_sha256"] is None
    assert not (tmp_path / "output" / "audio.stereo.wav").exists()


def test_core_report_uses_only_common_successes(tmp_path):
    rows = {"vilier": [{"key": "a", "status": "ok", "all": {"pit_si_sdr": 1.0}}],
            "cholimex": [{"key": "a", "status": "ok", "all": {"pit_si_sdr": 2.0}}]}
    runtime = {name: [{"status": "complete", "duration_sec": 1, "inference_seconds": 1}] for name in rows}
    result = comparison_report(rows, runtime, tmp_path / "comparison")
    assert result["common_keys"] == ["a"]
    assert json.loads((tmp_path / "comparison.json").read_text())["common_samples"] == 1


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
