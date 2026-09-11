"""Orchestrate the one-file, reference-free stereo benchmark and artifact output."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .activity import ActivityConfig, activity_summary, config_dict, energy_vad, mask_segments
from .audio import load_stereo
from .dynamics import (
    MAX_BACKCHANNEL_DURATION_SEC, MERGE_GAP_SEC, MIN_TURN_DURATION_SEC,
    analyze_turns, inactive_channel_energy_ratio_db,
)
from .models import acoustic_metrics, speaker_metrics


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_benchmark(audio_path: Path, output_dir: Path, device: str = "auto", debug: bool = False) -> tuple[dict, Path]:
    """Run reference-free diagnostics for a single, two-channel WAV/audio file."""
    audio = load_stereo(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_device = resolve_device(device)
    activity_config = ActivityConfig()
    left_mask, left_threshold = energy_vad(audio.left, audio.sample_rate, activity_config)
    right_mask, right_threshold = energy_vad(audio.right, audio.sample_rate, activity_config)
    # Frame count can differ only on malformed input; trim defensively for every paired statistic.
    frame_count = min(len(left_mask), len(right_mask))
    left_mask, right_mask = left_mask[:frame_count], right_mask[:frame_count]
    frame_sec = activity_config.frame_sec
    activity = activity_summary(left_mask, right_mask, frame_sec)
    dynamics = analyze_turns(left_mask, right_mask, frame_sec)
    report = {
        "input": {
            "path": str(audio.path.resolve()), "sha256": _sha256(audio.path), "sample_rate": audio.sample_rate,
            "duration_sec": audio.duration_sec, "samples": audio.frames, "channels": 2,
            "left": "estimated speaker track 1", "right": "estimated speaker track 2",
        },
        "acoustic_quality": acoustic_metrics(audio.left, audio.right, audio.sample_rate, selected_device),
        "speaker_identity": speaker_metrics(audio.left, audio.right, audio.sample_rate, left_mask, right_mask, frame_sec, selected_device),
        "speech_activity": {"vad": "energy_vad", "left_threshold_dbfs": left_threshold, "right_threshold_dbfs": right_threshold, **activity},
        "turn_taking": {key: value for key, value in dynamics.items() if key not in {"turns", "all_talk_spurts", "transitions", "backchannel_candidates"}},
        "backchannel": {"method": "VAD-based candidate; no ASR/linguistic criterion", "count": len(dynamics["backchannel_candidates"]), "per_minute": dynamics["backchannels_per_min"], "mean_duration_sec": dynamics["mean_backchannel_duration_sec"]},
        "leakage_proxy": {"name": "inactive_channel_energy_ratio_db", **inactive_channel_energy_ratio_db(audio.left, audio.right, left_mask, right_mask, frame_sec, audio.sample_rate)},
        "runtime": {"timestamp_utc": datetime.now(UTC).isoformat(), "device": selected_device},
        "versions": {"benchmark": "0.1.0", "python": platform.python_version(), "torch": torch.__version__, "torchaudio": torchaudio.__version__},
        "config": {"activity": config_dict(activity_config), "turn_analysis": {"merge_gap_sec": MERGE_GAP_SEC, "min_turn_duration_sec": MIN_TURN_DURATION_SEC, "max_backchannel_duration_sec": MAX_BACKCHANNEL_DURATION_SEC}},
    }
    timeline = {"left_vad": mask_segments(left_mask, frame_sec, "left"), "right_vad": mask_segments(right_mask, frame_sec, "right"), "turns": dynamics["turns"], "overlaps": mask_segments(left_mask & right_mask, frame_sec), "backchannel_candidates": dynamics["backchannel_candidates"], "transitions": dynamics["transitions"]}
    report_path = output_dir / "report.json"
    timeline_path = output_dir / "timeline.json"
    _write_json(report_path, report)
    _write_json(timeline_path, timeline)
    if debug:
        sf.write(output_dir / "left.wav", audio.left, audio.sample_rate)
        sf.write(output_dir / "right.wav", audio.right, audio.sample_rate)
        _write_json(output_dir / "vad_left.json", timeline["left_vad"])
        _write_json(output_dir / "vad_right.json", timeline["right_vad"])
        _write_json(output_dir / "turns.json", timeline["turns"])
        _write_json(output_dir / "overlap.json", timeline["overlaps"])
        _write_json(output_dir / "backchannels.json", timeline["backchannel_candidates"])
    return report, report_path


def print_summary(report: dict, report_path: Path) -> None:
    info, activity, turns = report["input"], report["speech_activity"], report["turn_taking"]
    print("Pipeline: Stereo Full-Duplex Benchmark\n")
    print(f"Input: {info['path']}\nDuration: {info['duration_sec']:.2f} s | Sample rate: {info['sample_rate']} Hz | Channels: 2")
    print("\nSpeech activity")
    for name in ("left_only", "right_only", "overlap", "silence"):
        item = activity[name]
        print(f"  {name:12} {item['duration_sec']:7.2f} s  {item['percentage']:5.1f}%")
    print("\nConversation dynamics")
    print(f"  Turn exchanges/min: {_format(turns['turn_exchanges_per_min'])}")
    print(f"  Simultaneous speech: {activity['overlap']['percentage']:.1f}%")
    print(f"  Overlap transitions: {_format_pct(turns['overlapping_transition_rate'])}")
    print(f"  VAD-based backchannels/min: {_format(turns['backchannels_per_min'])}")
    print("\nOptional reference-free models")
    for name, value in report["acoustic_quality"].items():
        print(f"  {name.upper():8} {_status(value['left'])}")
    print(f"  ITC       {_status(report['speaker_identity']['itc']['left'])}")
    print(f"\nBenchmark complete\nJSON report: {report_path}")


def _status(value: dict) -> str:
    return "ok" if value.get("status") == "ok" else f"N/A ({value.get('reason', 'unavailable')})"


def _format(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _format_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
