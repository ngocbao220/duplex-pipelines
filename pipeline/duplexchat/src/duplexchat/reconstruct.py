"""Purpose: Persist DuplexChat separated tracks and phase metadata.

Inputs: Two separated waveforms, output naming and model metadata.
Outputs: One stereo WAV file and separation manifest artifacts.
"""
from pathlib import Path

import torch

from core.outputs import save_stereo_wav, save_wav, write_json


OUTPUT_SAMPLE_RATE = 24_000
INPUT_SAMPLE_RATE = 16_000


def _to_output_rate(waveform, sample_rate):
    if sample_rate == OUTPUT_SAMPLE_RATE:
        return waveform
    output_length = max(1, round(waveform.shape[-1] * OUTPUT_SAMPLE_RATE / sample_rate))
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=output_length, mode="linear", align_corners=False,
    ).squeeze(0)


def write_stereo(output_prefix, phase_dir, first_channel, second_channel, sample_rate, backend, model):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stereo = prefix.parent / "audio.stereo.wav"
    first_channel = _to_output_rate(first_channel, sample_rate)
    second_channel = _to_output_rate(second_channel, sample_rate)
    save_stereo_wav(stereo, first_channel, second_channel, OUTPUT_SAMPLE_RATE)
    phase_stereo = phase_dir / "phase_04_separation" / "audio.stereo.wav"
    save_stereo_wav(phase_stereo, first_channel, second_channel, OUTPUT_SAMPLE_RATE)
    write_json(
        phase_dir / "phase_04_separation" / "separation.json",
        {
            "backend": backend,
            "model": model,
            "sample_rate": OUTPUT_SAMPLE_RATE,
            "input_sample_rate": INPUT_SAMPLE_RATE,
            "output_sample_rate": OUTPUT_SAMPLE_RATE,
            "stereo": str(stereo),
        },
    )
    return stereo


def write_conversation_stereo(
    conversation_dir,
    mixture,
    first_channel,
    second_channel,
    sample_rate,
    metadata,
):
    """Write one cropped conversation and its separator output."""
    conversation_dir = Path(conversation_dir)
    mixture = _to_output_rate(mixture, sample_rate)
    first_channel = _to_output_rate(first_channel, sample_rate)
    second_channel = _to_output_rate(second_channel, sample_rate)
    save_wav(conversation_dir / "mixture.wav", mixture, OUTPUT_SAMPLE_RATE)
    save_stereo_wav(conversation_dir / "audio.stereo.wav", first_channel, second_channel, OUTPUT_SAMPLE_RATE)
    write_json(conversation_dir / "metadata.json", {**metadata, "sample_rate": OUTPUT_SAMPLE_RATE})
    return conversation_dir / "audio.stereo.wav"
