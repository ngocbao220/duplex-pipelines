from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.benchmark import score_reference_audio_pair, score_reference_sample


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate, subtype="FLOAT")
    return path


def _reference_audio(sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    return 0.1 * np.sin(2 * math.pi * 220 * time), 0.1 * np.sin(2 * math.pi * 440 * time)


def test_score_reference_sample_reads_the_two_stereo_channels(tmp_path):
    first, second = _reference_audio()
    gt_first = _write_wav(tmp_path / "gt1.wav", first)
    gt_second = _write_wav(tmp_path / "gt2.wav", second)
    sample_dir = tmp_path / "predictions" / "sample"
    prediction = _write_wav(sample_dir / "audio.stereo.wav", np.column_stack([first, second]))

    row = score_reference_sample(
        {"key": "sample", "gt_speaker_1": str(gt_first), "gt_speaker_2": str(gt_second)},
        tmp_path / "predictions",
    )

    assert row["status"] == "ok"
    assert row["prediction"] == [str(prediction)]
    assert row["all"]["pit_si_sdr"] is not None


def test_score_reference_audio_pair_scores_explicit_five_wav_inputs(tmp_path):
    first, second = _reference_audio()
    gt_first = _write_wav(tmp_path / "gt1.wav", first)
    gt_second = _write_wav(tmp_path / "gt2.wav", second)
    mixed = _write_wav(tmp_path / "mixed.wav", first + second)
    prediction_first = _write_wav(tmp_path / "predict1.wav", first)
    prediction_second = _write_wav(tmp_path / "predict2.wav", second)

    row = score_reference_audio_pair(gt_first, gt_second, mixed, prediction_first, prediction_second, key="local")

    assert row["status"] == "ok"
    assert row["key"] == "local"
    assert row["all"]["pit_si_sdr"] is not None
    assert row["prediction"] == [str(prediction_first), str(prediction_second)]


def test_score_reference_audio_pair_rejects_a_mixed_input_that_does_not_match_gt(tmp_path):
    first, second = _reference_audio()
    gt_first = _write_wav(tmp_path / "gt1.wav", first)
    gt_second = _write_wav(tmp_path / "gt2.wav", second)
    mixed = _write_wav(tmp_path / "mixed.wav", first)
    prediction_first = _write_wav(tmp_path / "predict1.wav", first)
    prediction_second = _write_wav(tmp_path / "predict2.wav", second)

    with pytest.raises(ValueError, match=r"does not match gt1 \+ gt2"):
        score_reference_audio_pair(gt_first, gt_second, mixed, prediction_first, prediction_second)


def test_score_reference_audio_pair_accepts_a_pcm_clipped_gt_sum(tmp_path):
    time = np.arange(16000, dtype=np.float32) / 16000
    first = 0.7 * np.sin(2 * math.pi * 220 * time)
    second = 0.7 * np.sin(2 * math.pi * 220 * time)
    gt_first = _write_wav(tmp_path / "gt1.wav", first)
    gt_second = _write_wav(tmp_path / "gt2.wav", second)
    mixed = _write_wav(tmp_path / "mixed.wav", np.clip(first + second, -1.0, 1.0))
    prediction_first = _write_wav(tmp_path / "predict1.wav", first)
    prediction_second = _write_wav(tmp_path / "predict2.wav", second)

    row = score_reference_audio_pair(gt_first, gt_second, mixed, prediction_first, prediction_second)

    assert row["status"] == "ok"
    assert row["all"]["delta_si_sdr"] is not None


def test_local_reference_benchmark_notebook_contains_compilable_code_cells():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks" / "local_reference_benchmark.ipynb").read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert any("score_reference_audio_pair" in cell for cell in code_cells)
    assert any("Final summary" in cell for cell in code_cells)
    for index, source in enumerate(code_cells):
        compile(source, f"local_reference_benchmark.ipynb cell {index}", "exec")
