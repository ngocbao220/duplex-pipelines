import numpy as np
import pytest
import soundfile as sf

from core.stereo_benchmark.activity import activity_summary
from core.stereo_benchmark.audio import load_stereo
from core.stereo_benchmark.dynamics import analyze_turns, cosine_distinctiveness
from core.stereo_benchmark.models import acoustic_metrics


def test_load_stereo_rejects_mono(tmp_path):
    path = tmp_path / "mono.wav"
    sf.write(path, np.zeros(160), 16000)

    with pytest.raises(ValueError, match="exactly two channels"):
        load_stereo(path)


def test_load_stereo_preserves_left_and_right(tmp_path):
    path = tmp_path / "stereo.wav"
    data = np.column_stack([np.full(160, 0.25), np.full(160, -0.5)])
    sf.write(path, data, 16000, subtype="FLOAT")

    audio = load_stereo(path)

    assert audio.sample_rate == 16000
    assert audio.left[0] == pytest.approx(0.25)
    assert audio.right[0] == pytest.approx(-0.5)


def test_activity_summary_partitions_all_frames():
    left = np.array([1, 1, 0, 0], dtype=bool)
    right = np.array([0, 1, 1, 0], dtype=bool)

    result = activity_summary(left, right, frame_sec=0.1)

    assert result["left_only"]["duration_sec"] == pytest.approx(0.1)
    assert result["right_only"]["duration_sec"] == pytest.approx(0.1)
    assert result["overlap"]["duration_sec"] == pytest.approx(0.1)
    assert result["silence"]["duration_sec"] == pytest.approx(0.1)


def test_itd_is_one_minus_cosine_similarity():
    assert cosine_distinctiveness(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert cosine_distinctiveness(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_turns_detect_transition_and_overlap_transition():
    left = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    right = np.array([0, 0, 1, 1, 1, 1], dtype=bool)

    result = analyze_turns(left, right, frame_sec=0.5)

    assert result["transition_count"] == 1
    assert result["overlapping_transition_count"] == 1


def test_short_spurt_inside_other_turn_is_backchannel_candidate():
    left = np.array([1] * 10, dtype=bool)
    right = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)

    result = analyze_turns(left, right, frame_sec=0.2)

    assert len(result["backchannel_candidates"]) == 1
    assert result["backchannel_candidates"][0]["speaker"] == "right"


def test_optional_model_failure_is_reported_without_crashing(monkeypatch):
    import core.stereo_benchmark.models as models

    monkeypatch.setattr(models, "_squim_model", lambda _device: (_ for _ in ()).throw(RuntimeError("model unavailable")))

    result = acoustic_metrics(np.zeros(160), np.zeros(160), 16000, "cpu")

    assert result["squim"]["left"]["status"] == "unavailable"
    assert "model unavailable" in result["squim"]["left"]["reason"]
