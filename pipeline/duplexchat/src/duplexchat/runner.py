"""Purpose: Build a filtered DuplexChat conversation collection from one episode.

Inputs: One source recording and DuplexChat diarization/separation settings.
Outputs: Per-conversation 24 kHz stereo WAV artifacts and a collection manifest.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from core.orchestration.logging_style import StepTimer, get_logger
from core.outputs import write_dialogues_phase, write_json

from .audio import load_wav_tensor
from .devices import resolve_device, validate_multi_gpu
from .diarization import diarize
from .dialogue import dialogue_filter_summary, extract_valid_dialogues
from .preprocess import prepare_input
from .reconstruct import OUTPUT_SAMPLE_RATE, write_conversation_stereo
from .separation import separate_waveform
from .separation_backend import SAMPLE_RATE_IN, load_separation_models


def resolve_output_dir(output_prefix: str, output_dir: str | None = None) -> Path:
    if output_dir:
        return Path(output_dir)
    parent = Path(output_prefix).parent
    return parent if str(parent) != "." else Path("outputs") / "single_audio"


def _no_progress(_event: str, _value: int) -> None:
    pass


def _fit_channel(channel: torch.Tensor, length: int) -> torch.Tensor:
    if channel.shape[-1] < length:
        return torch.nn.functional.pad(channel, (0, length - channel.shape[-1]))
    return channel[..., :length]


def _resample(waveform: torch.Tensor, input_rate: int, output_rate: int) -> torch.Tensor:
    if input_rate == output_rate:
        return waveform
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=max(1, round(waveform.shape[-1] * output_rate / input_rate)),
        mode="linear", align_corners=False,
    ).squeeze(0)


def _release(model: object) -> None:
    if hasattr(model, "to"):
        model.to(torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _devices(runtime_device: str, device_ids: list[int] | None) -> list[str]:
    ids = validate_multi_gpu(bool(device_ids), device_ids)
    return [f"cuda:{index}" for index in ids] if ids else [resolve_device(runtime_device, allow_cpu_fallback=True)]


def _device_label(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


def _compact_path(value, width: int = 46) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return "..." + text[-(width - 3):]


def _log_run_summary(logger, audio_path, normalized, output_root, phase_dir, manifest, devices, conversations, segment_count, phase_times):
    input_width = 34
    output_width = 30
    saved_width = 48
    logger.info("Runtime devices: diarization=%s; separation=%s", _device_label(devices[0]), [_device_label(device) for device in devices])
    logger.info("Detected conversations: %d", conversations)
    logger.info("Run summary")
    logger.info("  %-22s | %-34s | %-30s | %-48s | %8s", "Phase", "Input", "Output", "Saved at", "Time")
    logger.info("  %s", "-" * 152)
    rows = [
        ("Preprocess", _compact_path(audio_path, input_width), "mono 16 kHz", _compact_path(normalized, saved_width), phase_times.get("preprocess", 0.0)),
        ("Speaker diarization", _compact_path(normalized.name, input_width), f"{segment_count} segments", _compact_path(phase_dir / "phase_02_diarization", saved_width), phase_times.get("diarization", 0.0)),
        ("Dialogue separation", f"{conversations} conversations", f"{conversations} stereo 24 kHz WAV", _compact_path(manifest, saved_width), phase_times.get("separation")),
    ]
    for phase, input_name, output, saved_at, elapsed in rows:
        duration = "SKIPPED" if elapsed is None else f"{elapsed:.2f}s"
        logger.info("  %-22s | %-34s | %-30s | %-48s | %8s", phase, input_name, output, saved_at, duration)
    logger.info("  Output root: %s", output_root)


def _separate_dialogues(waveform, sample_rate, dialogues, devices, num_steps, chunk_seconds):
    models = {device: load_separation_models(device=device) for device in devices}

    def run(pair):
        index, dialogue = pair
        start = max(0, min(waveform.shape[-1], round(dialogue.start * sample_rate)))
        end = max(start, min(waveform.shape[-1], round(dialogue.end * sample_rate)))
        crop = waveform[..., start:end].clone()
        device = devices[index % len(devices)]
        first, second, output_rate = separate_waveform(crop, sample_rate, models[device], num_steps, chunk_seconds, _no_progress)
        return index, dialogue, crop, first, second, output_rate

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            return sorted(pool.map(run, enumerate(dialogues)), key=lambda row: row[0])
    finally:
        for model in models.values():
            _release(model)


def run_single_audio(
    audio_path_str: str,
    diarize_chunk: float | None = None,
    separate_chunk: float = 120.0,
    diarization_backend: str = "auto",
    diarization_model: str = "pyannote/speaker-diarization-community-1",
    output_prefix: str = "output",
    output_dir: str | None = None,
    runtime_device: str = "auto",
    num_steps: int = 30,
    device_ids: list[int] | None = None,
) -> dict:
    audio_path = Path(audio_path_str)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    devices = _devices(runtime_device, device_ids)
    logger = get_logger("duplexchat")
    phase_dir = resolve_output_dir(output_prefix, output_dir)
    output_root = phase_dir.parent
    phase_dir.mkdir(parents=True, exist_ok=True)

    phase_times = {}
    with StepTimer(logger, "Step 0: Preprocess") as timer:
        normalized = prepare_input(audio_path, phase_dir)
    phase_times["preprocess"] = timer.elapsed
    logger.info("Runtime devices: diarization=%s; separation=%s", _device_label(devices[0]), [_device_label(device) for device in devices])
    with StepTimer(logger, "Step 1: Speaker diarization") as timer:
        diarizer, segments = diarize(normalized, phase_dir, diarization_model, diarization_backend, devices[0], diarize_chunk, _no_progress)
    phase_times["diarization"] = timer.elapsed
    _release(diarizer)

    summary = dialogue_filter_summary(segments)
    dialogues = extract_valid_dialogues(segments)
    logger.info("Conversation filtering=%s", summary)
    logger.info("Detected conversations: %d", len(dialogues))
    write_dialogues_phase(phase_dir, dialogues)
    waveform, sample_rate = load_wav_tensor(normalized)
    rows = []
    if dialogues:
        with StepTimer(logger, "Step 2: Dialogue separation") as timer:
            logger.info("DialogueSidon sample rates: input=%d Hz, output=%d Hz", SAMPLE_RATE_IN, OUTPUT_SAMPLE_RATE)
            separated = _separate_dialogues(waveform, sample_rate, dialogues, devices, num_steps, separate_chunk)
        phase_times["separation"] = timer.elapsed
        for index, dialogue, crop, first, second, output_rate in separated:
            first, second, mixture = (_resample(first, output_rate, OUTPUT_SAMPLE_RATE), _resample(second, output_rate, OUTPUT_SAMPLE_RATE), _resample(crop, sample_rate, OUTPUT_SAMPLE_RATE))
            length = min(first.shape[-1], second.shape[-1], mixture.shape[-1])
            first, second, mixture = (_fit_channel(first, length), _fit_channel(second, length), _fit_channel(mixture, length))
            conversation_dir = output_root / "conversations" / f"conversation_{index:05d}"
            metadata = {"conversation_idx": index, "start": dialogue.start, "end": dialogue.end,
                        "duration": length / OUTPUT_SAMPLE_RATE, "speakers": dialogue.speakers,
                        "segments": dialogue.segments, "sample_rate": OUTPUT_SAMPLE_RATE,
                        "input_sample_rate": SAMPLE_RATE_IN, "output_sample_rate": OUTPUT_SAMPLE_RATE,
                        "status": "complete"}
            stereo = write_conversation_stereo(conversation_dir, mixture, first, second, OUTPUT_SAMPLE_RATE, metadata)
            rows.append({**metadata, "stereo": str(stereo), "mixture": str(conversation_dir / "mixture.wav")})
    manifest = output_root / "conversations" / "manifest.json"
    write_json(manifest, {"output_kind": "conversation_collection", "sample_rate": OUTPUT_SAMPLE_RATE,
                          "input_sample_rate": SAMPLE_RATE_IN, "output_sample_rate": OUTPUT_SAMPLE_RATE,
                          "conversation_count": len(rows), "filter": summary, "conversations": rows})
    _log_run_summary(logger, audio_path, normalized, output_root, phase_dir, manifest, devices, len(rows), len(segments), phase_times)
    return {"collection": manifest, "segments": segments, "conversations": rows, "devices": devices}
