import json
import sys

import numpy as np
import soundfile as sf

from core.orchestration.contract import run_sample, validate_conversation_collection, validate_tracks
from core.orchestration.process import stream_process
from core.orchestration.report import comparison_report


def _wav(path, frames=1600):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(frames), 16000)
    return path


def test_core_contract_requires_two_full_tracks(tmp_path):
    mixture = _wav(tmp_path / "mixture.wav")
    tracks = [_wav(tmp_path / "a.wav"), _wav(tmp_path / "b.wav")]
    assert validate_tracks(mixture, tracks) == 0.1
    _wav(tracks[1], frames=800)
    try:
        validate_tracks(mixture, tracks)
    except ValueError as error:
        assert "timeline" in str(error)
    else:
        raise AssertionError("short track must fail validation")


def test_conversation_collection_requires_two_timeline_aligned_tracks_per_dialogue(tmp_path):
    mixture = _wav(tmp_path / "conversation" / "mixture.wav", frames=3200)
    first = _wav(tmp_path / "conversation" / "speakerA.wav", frames=3200)
    second = _wav(tmp_path / "conversation" / "speakerB.wav", frames=3200)
    collection = [{"mixture": str(mixture), "tracks": [str(first), str(second)]}]

    assert validate_conversation_collection(collection) == 0.2
    _wav(second, frames=1600)
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
        first = _wav(output / "conversations" / "conversation_00000" / "speakerA.wav")
        second = _wav(output / "conversations" / "conversation_00000" / "speakerB.wav")
        return [], {"output_kind": "conversation_collection", "conversations": [
            {"mixture": str(mixture), "tracks": [str(first), str(second)]}
        ]}

    result = run_sample("duplexchat", {"key": "sample", "mixture": str(source)}, tmp_path / "output", {}, "code", collection_adapter)

    assert result["status"] == "complete"
    assert result["track_sha256"] == []
    assert not (tmp_path / "output" / "speakerA.wav").exists()


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
