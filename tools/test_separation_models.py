#!/usr/bin/env python3
"""Run comparable source-separation smoke tests on one mixed WAV.

Purpose: Compare supported separation checkpoints without changing any pipeline.
Inputs: One mono/stereo mixture WAV and optional local SepReformer checkpoint.
Outputs: Per-model source WAVs, report.json, and an aggregate summary.json.

This is an inference smoke runner, not an objective benchmark: it has no
reference sources and therefore reports elapsed time/RTF only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np


MODELS: dict[str, dict[str, Any]] = {
    "sepformer-wsj02mix": {
        "id": "speechbrain/sepformer-wsj02mix",
        "adapter": "speechbrain",
        "sample_rate": 8_000,
        "expected_sources": 2,
    },
    "mossformer2-librimix-2spk": {
        "id": "alibabasglab/mossformer2-librimix-2spk",
        "adapter": "transformers_mossformer2",
        "sample_rate": 8_000,
        "expected_sources": 2,
    },
    "dialoguesidon": {
        "id": "sarulab-speech/DialogueSidon",
        "adapter": "dialoguesidon",
        "sample_rate": 16_000,
        "expected_sources": 2,
    },
    "rahma89-voice-separation-model": {
        "id": "Rahma89/voice-separation-model",
        "adapter": "rahma89_asteroid",
        "sample_rate": 16_000,
        "expected_sources": 3,
    },
    "sepformer-whamr": {
        "id": "speechbrain/sepformer-whamr",
        "adapter": "speechbrain",
        "sample_rate": 8_000,
        "expected_sources": 2,
    },
    "sepreformer-base-wsj0": {
        "id": "SepReformer_Base_WSJ0",
        "adapter": "sepreformer",
        "sample_rate": 8_000,
        "expected_sources": 2,
    },
}


def select_models(raw: str) -> list[str]:
    """Resolve ``all`` or a comma-separated list of stable model names."""
    selected = list(MODELS) if raw.strip().lower() == "all" else [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in selected if name not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Choices: {', '.join(MODELS)}")
    return selected


def _resample(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    import torch
    import torchaudio.functional as audio_functional

    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
    return audio_functional.resample(tensor, original_sr, target_sr).squeeze(0).numpy()


def normalize_sources(sources: Any, expected_length: int) -> np.ndarray:
    """Normalize model-specific output layouts to ``[sources, samples]``.

    Source count is deliberately preserved. In particular, Rahma89 is a
    three-source model, so this tool never silently drops a source to claim a
    two-track result.
    """
    if hasattr(sources, "detach"):
        sources = sources.detach().float().cpu().numpy()
    array = np.asarray(sources, dtype=np.float32).squeeze()
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise RuntimeError(f"Expected source audio shaped [sources, samples], got {array.shape}")
    # Common SpeechBrain layout is [samples, sources].
    if array.shape[0] > array.shape[1] and array.shape[1] <= 8:
        array = array.T
    if array.shape[1] > expected_length:
        array = array[:, :expected_length]
    elif array.shape[1] < expected_length:
        array = np.pad(array, ((0, 0), (0, expected_length - array.shape[1])))
    return array.astype(np.float32, copy=False)


def write_outputs(output_dir: Path, mixture: np.ndarray, sample_rate: int, sources: np.ndarray) -> dict[str, Any]:
    """Write inspectable source artifacts, including aliases for exactly 2 tracks."""
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    mixture_path = output_dir / "mixture.wav"
    sf.write(mixture_path, mixture, sample_rate)
    source_paths = []
    for index, source in enumerate(sources, start=1):
        path = output_dir / f"source_{index:02d}.wav"
        sf.write(path, source, sample_rate)
        source_paths.append(path.name)
    result: dict[str, Any] = {"mixture": mixture_path.name, "sources": source_paths}
    if len(sources) == 2:
        for alias, source in zip(("speakerA.wav", "speakerB.wav"), sources, strict=True):
            sf.write(output_dir / alias, source, sample_rate)
        result["speaker_a"] = "speakerA.wav"
        result["speaker_b"] = "speakerB.wav"
    return result


def build_report(model_name: str, paths: dict[str, Any], source_count: int, elapsed_seconds: float, **extra: Any) -> dict[str, Any]:
    report = {
        "status": "complete",
        "model": model_name,
        "source_count": source_count,
        "two_track_output": source_count == 2,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "artifacts": paths,
    }
    report.update(extra)
    return report


def _speechbrain_sources(model_id: str, mixture: np.ndarray, sample_rate: int, device: str) -> tuple[np.ndarray, int]:
    import torch
    from speechbrain.inference.separation import SepformerSeparation

    model_sr = 8_000
    model = SepformerSeparation.from_hparams(
        source=model_id,
        savedir=str(Path.home() / ".cache" / "speechbrain" / model_id.replace("/", "_")),
        run_opts={"device": device},
    )
    model_input = _resample(mixture, sample_rate, model_sr)
    output = model.separate_batch(torch.from_numpy(model_input).unsqueeze(0).to(device))
    return normalize_sources(output, len(model_input)), model_sr


def _dialoguesidon_sources(mixture: np.ndarray, sample_rate: int, device: str, steps: int) -> tuple[np.ndarray, int]:
    source_root = Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"
    sys.path.insert(0, str(source_root))
    try:
        from duplexchat.separation_backend import load_separation_models, run_separation

        models = load_separation_models(device=device, backend="dialoguesidon", model_id=MODELS["dialoguesidon"]["id"])
        import torch

        first, second, output_sr = run_separation(
            torch.from_numpy(mixture).unsqueeze(0), sample_rate, num_steps=steps,
            models=models, chunk_seconds=max(30.0, len(mixture) / sample_rate), overlap_seconds=0.0,
        )
        return normalize_sources(np.vstack((first.numpy(), second.numpy())), int(first.shape[-1])), output_sr
    finally:
        sys.path.remove(str(source_root))


def _mossformer2_sources(model_id: str, mixture: np.ndarray, sample_rate: int, device: str) -> tuple[np.ndarray, int]:
    """Use the official Transformers AutoModel implementation for LibriMix."""
    import torch
    from transformers import AutoModel

    model_sr = 8_000
    audio = _resample(mixture, sample_rate, model_sr)
    # This public checkpoint ships a standard Transformers ``mossformer2``
    # config. Do not execute arbitrary repository code while benchmarking it.
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    batch = torch.from_numpy(audio).unsqueeze(0).to(device)
    with torch.inference_mode():
        try:
            output = model(batch)
        except TypeError:
            output = model(input_values=batch)
    for attribute in ("separated_audio", "sources", "audio_values", "logits"):
        if hasattr(output, attribute):
            output = getattr(output, attribute)
            break
    if isinstance(output, (tuple, list)):
        output = output[0]
    return normalize_sources(output, len(audio)), model_sr


def _rahma89_sources(mixture: np.ndarray, sample_rate: int, device: str) -> tuple[np.ndarray, int]:
    """Load Rahma89 exactly as its published Asteroid project specifies."""
    import torch
    import yaml
    from huggingface_hub import snapshot_download

    repo_dir = Path(snapshot_download(
        repo_id=MODELS["rahma89-voice-separation-model"]["id"],
        allow_patterns=["best.ckpt", "configs/*", "src/*"],
    ))
    sys.path.insert(0, str(repo_dir))
    try:
        from src.model import build_model, load_checkpoint
        train = yaml.safe_load((repo_dir / "configs" / "train.yaml").read_text(encoding="utf-8"))
        data = yaml.safe_load((repo_dir / "configs" / "data.yaml").read_text(encoding="utf-8"))
        configuration, dataset = dict(train["model"]), data["dataset"]
        # The published training YAML has this key, but inference deliberately
        # disables gradient checkpointing and passes it explicitly below.
        configuration.pop("gradient_checkpointing", None)
        model_sr = int(dataset["sample_rate"])
        model = build_model(n_src=dataset["n_src"], sample_rate=model_sr, **configuration, use_gradient_checkpointing=False)
        load_checkpoint(model, repo_dir / "best.ckpt", device)
        audio = _resample(mixture, sample_rate, model_sr)
        with torch.inference_mode():
            output = model(torch.from_numpy(audio).unsqueeze(0).to(device))
        return normalize_sources(output, len(audio)), model_sr
    finally:
        sys.path.remove(str(repo_dir))


def _sepreformer_sources(mixture: np.ndarray, sample_rate: int, device: str, checkpoint: str | None) -> tuple[np.ndarray, int]:
    source_root = Path(__file__).resolve().parents[1] / "pipeline" / "vilier" / "src"
    sys.path.insert(0, str(source_root))
    try:
        from vilier.separation import SepReformerSeparator

        configured_checkpoint = checkpoint or os.environ.get("VILIER_SEPREFORMER_CHECKPOINT")
        if not configured_checkpoint:
            raise RuntimeError("SepReformer requires --sepreformer-checkpoint or VILIER_SEPREFORMER_CHECKPOINT")
        separator = SepReformerSeparator(
            Path(__file__).resolve().parents[1] / "pipeline" / "vilier" / "SepReformer",
            device, "SepReformer_Base_WSJ0", checkpoint_path=configured_checkpoint,
        )
        first, second = separator.separate(mixture, sample_rate)
        return normalize_sources(np.vstack((first, second)), len(mixture)), sample_rate
    finally:
        sys.path.remove(str(source_root))


def run_one(name: str, mixture: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[np.ndarray, int]:
    spec = MODELS[name]
    adapter = spec["adapter"]
    if adapter == "speechbrain":
        return _speechbrain_sources(spec["id"], mixture, sample_rate, args.device)
    if adapter == "dialoguesidon":
        return _dialoguesidon_sources(mixture, sample_rate, args.device, args.dialoguesidon_steps)
    if adapter == "transformers_mossformer2":
        return _mossformer2_sources(spec["id"], mixture, sample_rate, args.device)
    if adapter == "rahma89_asteroid":
        return _rahma89_sources(mixture, sample_rate, args.device)
    if adapter == "sepreformer":
        return _sepreformer_sources(mixture, sample_rate, args.device, args.sepreformer_checkpoint)
    raise AssertionError(f"No adapter for {adapter}")


def _load_mixture(path: Path, max_seconds: float | None) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mixture = audio.mean(axis=1, dtype=np.float32)
    if max_seconds is not None:
        mixture = mixture[: round(max_seconds * sample_rate)]
    if mixture.size == 0:
        raise ValueError("Input contains no audio after --max-seconds")
    return mixture, int(sample_rate)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One overlap/mixed WAV file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="all", help=f"all or comma-separated: {', '.join(MODELS)}")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-seconds", type=float, default=None, help="Bound the input duration for a smoke run")
    parser.add_argument("--dialoguesidon-steps", type=int, default=10)
    parser.add_argument("--sepreformer-checkpoint", default=None)
    return parser.parse_args(argv)


def _resolve_device(raw: str) -> str:
    import torch

    if raw == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    if raw.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using cpu", file=sys.stderr)
        return "cpu"
    return raw


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.device = _resolve_device(args.device)
    names = select_models(args.models)
    mixture, sample_rate = _load_mixture(args.input, args.max_seconds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    print(f"[INFO] mixed input={args.input} duration={len(mixture) / sample_rate:.2f}s device={args.device}")
    for index, name in enumerate(names, start=1):
        model_dir = args.output_dir / name
        print(f"[INFO] [{index}/{len(names)}] {name}: starting")
        started = time.perf_counter()
        try:
            sources, output_sr = run_one(name, mixture, sample_rate, args)
            sources = np.vstack([_resample(source, output_sr, sample_rate) for source in sources])
            sources = normalize_sources(sources, len(mixture))
            artifacts = write_outputs(model_dir, mixture, sample_rate, sources)
            elapsed = time.perf_counter() - started
            report = build_report(name, artifacts, len(sources), elapsed,
                                  model_id=MODELS[name]["id"], adapter=MODELS[name]["adapter"],
                                  device=args.device, input_sample_rate=sample_rate, output_sample_rate=sample_rate,
                                  duration_seconds=round(len(mixture) / sample_rate, 3),
                                  real_time_factor=round(elapsed / (len(mixture) / sample_rate), 4))
            print(f"[INFO] [{index}/{len(names)}] {name}: complete sources={len(sources)} rtf={report['real_time_factor']}")
        except Exception as exc:  # One missing optional runtime must not stop comparisons.
            elapsed = time.perf_counter() - started
            model_dir.mkdir(parents=True, exist_ok=True)
            report = {"status": "failed", "model": name, "model_id": MODELS[name]["id"],
                      "adapter": MODELS[name]["adapter"], "elapsed_seconds": round(elapsed, 3),
                      "error": f"{type(exc).__name__}: {exc}"}
            if os.environ.get("SEPARATION_SMOKE_TRACEBACK") == "1":
                report["traceback"] = traceback.format_exc()
            print(f"[ERROR] [{index}/{len(names)}] {name}: {report['error']}", file=sys.stderr)
        (model_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary.append(report)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    complete = sum(row["status"] == "complete" for row in summary)
    print(f"[INFO] complete={complete}/{len(summary)} summary={args.output_dir / 'summary.json'}")
    return 0 if complete == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
