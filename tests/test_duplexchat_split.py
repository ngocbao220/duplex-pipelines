from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat import runner  # noqa: E402


def _segment(speaker: str, start: float, end: float) -> dict:
    return {"speaker": speaker, "start": start, "end": end}


def test_split_runner_separates_each_dialogue_with_one_model_load(monkeypatch, tmp_path):
    output_root = tmp_path / "output"
    phase_dir = output_root / "phases"
    source_wav = tmp_path / "source.wav"
    source_wav.write_bytes(b"placeholder")
    segments = [
        _segment("A", 0.0, 6.0),
        _segment("B", 6.0, 12.0),
        _segment("A", 17.0, 23.0),
        _segment("B", 23.0, 29.0),
    ]
    calls = {"diarize": 0, "load": 0, "separate": 0}

    monkeypatch.setattr(runner, "prepare_input", lambda *_args: source_wav)

    def fake_diarize(*_args, **_kwargs):
        calls["diarize"] += 1
        return object(), segments

    monkeypatch.setattr(runner, "diarize", fake_diarize)
    monkeypatch.setattr(
        runner,
        "load_wav_tensor",
        lambda *_args: (torch.zeros(1, 29 * 16_000), 16_000),
    )

    def fake_load(*_args, **_kwargs):
        calls["load"] += 1
        return object()

    def fake_separate(waveform, sample_rate, *_args):
        calls["separate"] += 1
        output_length = round(waveform.shape[-1] * 24_000 / sample_rate)
        output = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=output_length, mode="linear", align_corners=False,
        ).squeeze(0)
        return output.clone(), output.clone() + 1, 24_000

    monkeypatch.setattr(runner, "load_separation_models", fake_load)
    monkeypatch.setattr(runner, "separate_waveform", fake_separate)

    result = runner.run_single_audio(
        str(source_wav),
        output_prefix=str(output_root / "speaker"),
        output_dir=str(phase_dir),
        runtime_device="cpu",
        split_conversation=True,
    )

    manifest = json.loads((output_root / "conversations" / "manifest.json").read_text())
    assert calls == {"diarize": 1, "load": 1, "separate": 2}
    assert result["stereo"] == output_root / "audio.stereo.wav"
    assert manifest["sample_rate"] == 24_000
    assert len(manifest["conversations"]) == 2
    assert all(
        (output_root / "conversations" / f"conversation_{index:05d}" / "audio.stereo.wav").exists()
        for index in range(2)
    )