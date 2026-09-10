"""Purpose: Persist DuplexChat separated tracks and phase metadata.

Inputs: Two separated waveforms, output naming and model metadata.
Outputs: One stereo WAV file and separation manifest artifacts.
"""
from pathlib import Path

from core.outputs import save_stereo_wav, write_json


def write_stereo(output_prefix, phase_dir, first_channel, second_channel, sample_rate, backend, model):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stereo = prefix.parent / "audio.stereo.wav"
    save_stereo_wav(stereo, first_channel, second_channel, sample_rate)
    phase_stereo = phase_dir / "phase_04_separation" / "audio.stereo.wav"
    save_stereo_wav(phase_stereo, first_channel, second_channel, sample_rate)
    write_json(phase_dir / "phase_04_separation" / "separation.json", {"backend": backend, "model": model, "sample_rate": sample_rate, "stereo": str(stereo)})
    return stereo
