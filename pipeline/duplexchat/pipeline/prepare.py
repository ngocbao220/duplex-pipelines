from pathlib import Path
import subprocess

from duplexchat_pipe.outputs import write_json


def prepare_input(audio_path: Path, output_dir: Path) -> Path:
    write_json(output_dir / "phase_00_input" / "input.json", {"audio_path": str(audio_path), "audio_name": audio_path.name})
    target = output_dir / "phase_01_preprocess" / "audio_16k_mono.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(target)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target
