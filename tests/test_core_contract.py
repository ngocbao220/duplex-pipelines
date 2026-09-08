import json

import numpy as np
import soundfile as sf

from core.orchestration.contract import run_sample, validate_tracks
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


def test_core_report_uses_only_common_successes(tmp_path):
    rows = {"vilier": [{"key": "a", "status": "ok", "all": {"pit_si_sdr": 1.0}}],
            "cholimex": [{"key": "a", "status": "ok", "all": {"pit_si_sdr": 2.0}}]}
    runtime = {name: [{"status": "complete", "duration_sec": 1, "inference_seconds": 1}] for name in rows}
    result = comparison_report(rows, runtime, tmp_path / "comparison")
    assert result["common_keys"] == ["a"]
    assert json.loads((tmp_path / "comparison.json").read_text())["common_samples"] == 1
