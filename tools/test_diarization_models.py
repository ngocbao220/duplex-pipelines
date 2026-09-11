#!/usr/bin/env python3
"""Run a diarization smoke test on one audio file.

Inputs: One audio path and a Hugging Face diarization model.
Outputs: Speaker-labelled segments, labels, linking diagnostics, and timing report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "pipeline" / "duplexchat" / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from duplexchat.audio import transcode_to_wav_16k_mono  # noqa: E402
from duplexchat.diarization_backend import (  # noqa: E402
    load_diarization_pipeline,
    run_diarization,
)
from core.outputs import write_diarization_phase, write_json  # noqa: E402


DEFAULT_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input WAV/audio file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face diarization model or alias")
    parser.add_argument("--backend", default="auto", choices=("auto", "pyannote", "sortformer", "diarizen"))
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N")
    parser.add_argument("--chunk-seconds", type=float, default=90.0)
    parser.add_argument("--max-seconds", type=float, default=None, help="Only process the first N seconds")
    return parser.parse_args(argv)


def _speaker_summary(segments: list[dict]) -> dict[str, dict[str, float | int]]:
    durations: dict[str, float] = defaultdict(float)
    for segment in segments:
        durations[str(segment["speaker"])] += max(0.0, float(segment["end"]) - float(segment["start"]))
    total = sum(durations.values())
    return {
        speaker: {
            "speech_seconds": round(duration, 3),
            "speech_percent": round(100.0 * duration / total, 3) if total else 0.0,
            "segments": sum(1 for segment in segments if str(segment["speaker"]) == speaker),
        }
        for speaker, duration in sorted(durations.items())
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"Input audio file does not exist: {args.input}")
    if args.chunk_seconds <= 0:
        raise SystemExit("--chunk-seconds must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise SystemExit("--max-seconds must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalized = args.output_dir / "audio_16k_mono.wav"
    transcode_to_wav_16k_mono(args.input, normalized)
    if args.max_seconds is not None:
        import soundfile as sf

        audio, sample_rate = sf.read(normalized, dtype="float32", always_2d=True)
        audio = audio[: max(1, round(args.max_seconds * sample_rate))]
        sf.write(normalized, audio, sample_rate, subtype="FLOAT")

    started = time.perf_counter()
    model = load_diarization_pipeline(args.model, device=args.device, backend=args.backend)
    diagnostics: dict = {}
    try:
        segments = run_diarization(
            model,
            normalized,
            max_chunk_dur=args.chunk_seconds,
            diagnostics=diagnostics,
        )
    finally:
        if hasattr(model, "to"):
            model.to("cpu")
    elapsed = time.perf_counter() - started

    phase_dir = write_diarization_phase(
        args.output_dir,
        segments,
        duration_sec=max((float(item["end"]) for item in segments), default=0.0),
        model=args.model,
        backend=args.backend,
    )
    write_json(phase_dir / "linking.json", diagnostics)
    report = {
        "status": "complete",
        "model": args.model,
        "backend": args.backend,
        "device": args.device,
        "input": str(args.input),
        "normalized_input": str(normalized),
        "segment_count": len(segments),
        "speaker_count": len({str(item["speaker"]) for item in segments}),
        "speakers": _speaker_summary(segments),
        "elapsed_seconds": round(elapsed, 3),
        "max_end_seconds": round(max((float(item["end"]) for item in segments), default=0.0), 3),
        "artifacts": {
            "segments": str(phase_dir / "diarization.json"),
            "labels": str(phase_dir / "labels"),
            "linking": str(phase_dir / "linking.json"),
        },
    }
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
