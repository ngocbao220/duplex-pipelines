"""Purpose: Normalize a DuplexChat input into the phase audio artifact.

Inputs: Arbitrary source audio path and phase output directory.
Outputs: 24 kHz mono WAV and input metadata JSON.
"""
from pathlib import Path
import subprocess

from core.outputs import write_json


def prepare_input(audio_path: Path, output_dir: Path) -> Path:
    write_json(output_dir / "phase_00_input" / "input.json", {"audio_path": str(audio_path), "audio_name": audio_path.name})
    target = output_dir / "phase_01_preprocess" / "audio_24k_mono.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(audio_path), "-ar", "24000", "-ac", "1", str(target)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target
