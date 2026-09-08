"""Purpose: Bridge vendored Sommelier artifacts to the common two-track contract.

Inputs: One mixture path, output directory and fixed Sommelier profile.
Outputs: Native Sommelier JSON/MP3 artifacts and two full-duration WAV tracks.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(source: Path, output: Path, config: dict):
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Sommelier requires HUGGINGFACE_TOKEN for pyannote diarization and embedding")
    _stage_sepreformer_checkpoint()
    native = output / "native"
    input_dir = native / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied = input_dir / source.name
    shutil.copy2(source, copied)
    cfg = {
        "huggingface_token": token,
        "language": {"multilingual": False, "supported": ["en", "ko", "ja", "zh", "es", "fr", "de", "it", "pt", "ru", "ar", "hi"]},
        "entrypoint": {"input_folder_path": str(input_dir), "SAMPLE_RATE": int(config.get("sample_rate", 16000))},
        "separate": {"step1": {"model_path": "", "denoise": True, "margin": 44100, "chunks": 15, "n_fft": 6144, "dim_t": 8, "dim_f": 3072}},
    }
    vendor = Path(__file__).resolve().parents[2] / "vendor" / "podcast_pipeline"
    with tempfile.TemporaryDirectory(prefix="sommelier-config-") as temporary:
        config_path = Path(temporary) / "config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        command = [sys.executable, str(vendor / "main_original_ASR_MoE.py"), "--input_folder_path", str(input_dir), "--config_path", str(config_path), "--sepreformer", "--no-demucs", "--no-ASRMoE", "--no-qwen3omni", "--LLM", "case_0", "--overlap_threshold", str(config.get("overlap_threshold", 1.0)), "--speaker-link-threshold", str(config.get("speaker_link_threshold", 0.75))]
        subprocess.run(command, cwd=vendor, check=True)
    manifests = sorted(input_dir.rglob(f"{source.stem}.json"), key=lambda path: path.stat().st_mtime)
    if not manifests:
        raise RuntimeError("Sommelier completed without its JSON manifest")
    tracks = _reconstruct_tracks(source, manifests[-1], output)
    return tracks, {"native_manifest": str(manifests[-1])}


def _stage_sepreformer_checkpoint() -> None:
    """Expose the shared checkpoint through original Sommelier's fixed lookup path."""
    configured = os.environ.get("VILIER_SEPREFORMER_CHECKPOINT", "")
    if not configured:
        raise RuntimeError("Set VILIER_SEPREFORMER_CHECKPOINT to a trusted SepReformer .pt/.pth file")
    checkpoint = Path(configured).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() not in {".pt", ".pth"}:
        raise RuntimeError(f"Invalid VILIER_SEPREFORMER_CHECKPOINT: {checkpoint}")
    weights_dir = Path(__file__).resolve().parents[2] / "vendor" / "SepReformer" / "models" / "SepReformer_Base_WSJ0" / "log" / "pretrain_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    link = weights_dir / checkpoint.name
    if link.exists() or link.is_symlink():
        if link.resolve() != checkpoint:
            raise RuntimeError(f"Sommelier checkpoint link already targets another file: {link}")
        return
    link.symlink_to(checkpoint)


def _reconstruct_tracks(source: Path, manifest_path: Path, output: Path) -> list[Path]:
    import numpy as np
    import soundfile as sf
    from pydub import AudioSegment

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rate = int(payload.get("metadata", {}).get("sample_rate", 16000))
    original = AudioSegment.from_file(source).set_channels(1).set_frame_rate(rate)
    total = len(original.get_array_of_samples())
    tracks = {}
    segment_dir = manifest_path.parent / source.stem
    for index, segment in enumerate(payload.get("segments", [])):
        speaker = str(segment.get("speaker", ""))
        if not speaker:
            continue
        path = segment_dir / f"{segment.get('index', f'{index:05d}')}_{speaker}.mp3"
        if not path.exists():
            continue
        audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(rate)
        samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        start = max(0, int(round(float(segment["start"]) * rate)))
        end = min(total, start + len(samples))
        track = tracks.setdefault(speaker, np.zeros(total, dtype=np.float32))
        track[start:end] += samples[: end - start]
    if len(tracks) != 2:
        raise ValueError(f"Sommelier must produce exactly two speakers; got {len(tracks)}")
    paths = []
    for index, (_, audio) in enumerate(sorted(tracks.items())):
        path = output / f"speaker{'AB'[index]}.wav"
        sf.write(path, np.clip(audio, -1.0, 1.0), rate, subtype="PCM_16")
        paths.append(path)
    return paths
