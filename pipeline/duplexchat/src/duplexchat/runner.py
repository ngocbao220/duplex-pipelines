"""Purpose: Execute DuplexChat preprocessing through stereo export.

Inputs: One mixture path, phase/model settings, output location.
Outputs: One stereo WAV file and inspectable phase artifacts.
"""
import torch
from pathlib import Path

from .audio import load_wav_tensor
from .dialogue import dialogue_filter_summary, extract_valid_dialogues, speaker_time_ratios
from .preprocess import prepare_input
from .diarization import diarize
from .reconstruct import OUTPUT_SAMPLE_RATE, write_conversation_stereo, write_stereo
from .separation import separate, separate_waveform
from .separation_backend import load_separation_models
from core.outputs import save_stereo_wav, write_dialogues_phase, write_json

from core.orchestration.logging_style import StepTimer, get_logger

import argparse


def release_diarization_gpu_memory(diarize_pipeline):
    if hasattr(diarize_pipeline, "to"):
        diarize_pipeline.to(torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_output_dir(output_prefix: str, output_dir: str | None = None) -> Path:
    if output_dir:
        return Path(output_dir)
    parent = Path(output_prefix).parent
    return parent if str(parent) != "." else Path("outputs") / "single_audio"


def _no_progress(event: str, value: int) -> None:
    """Pipeline progress is emitted as colored step logs, not tqdm bars."""


def _fit_channel(channel, length):
    if channel.shape[-1] < length:
        return torch.nn.functional.pad(channel, (0, length - channel.shape[-1]))
    return channel[..., :length]


def _resample_waveform(waveform, input_rate, output_rate):
    if input_rate == output_rate:
        return waveform
    output_length = max(1, round(waveform.shape[-1] * output_rate / input_rate))
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=output_length, mode="linear", align_corners=False,
    ).squeeze(0)


def _run_split_conversation(
    temp_wav,
    phase_output_dir,
    output_root,
    segments,
    device,
    separation_backend,
    separation_model,
    num_steps,
    separate_chunk,
    output_prefix,
    logger,
):
    waveform, input_sample_rate = load_wav_tensor(temp_wav)
    filter_summary = dialogue_filter_summary(segments)
    logger.info(
        "Conversation filtering: segments=%d speakers=%d silence_groups=%d "
        "two_speaker_runs=%d rejected_short=%d rejected_imbalanced=%d accepted=%d",
        filter_summary["segments"],
        filter_summary["speakers"],
        filter_summary["silence_groups"],
        filter_summary["two_speaker_runs"],
        filter_summary["rejected_short"],
        filter_summary["rejected_imbalanced"],
        filter_summary["accepted"],
    )
    dialogues = extract_valid_dialogues(segments)
    logger.info("Diarization produced %d valid conversations", len(dialogues))
    for index, dialogue in enumerate(dialogues):
        ratios = speaker_time_ratios(dialogue)
        ratio_text = ", ".join(
            f"{speaker}={ratio * 100:.1f}%" for speaker, ratio in sorted(ratios.items())
        )
        logger.info(
            "Conversation %05d: start=%.2fs end=%.2fs duration=%.2fs speakers=%s speech_ratio={%s}",
            index,
            dialogue.start,
            dialogue.end,
            dialogue.duration,
            ",".join(sorted(dialogue.speakers)),
            ratio_text,
        )
    write_dialogues_phase(phase_output_dir, dialogues)
    write_json(
        phase_output_dir / "phase_03_dialogues" / "manifest.json",
        {"count": len(dialogues), "dialogues": []},
    )

    manifest = []
    models = None
    output_sample_rate = OUTPUT_SAMPLE_RATE
    timeline_sample_rate = None
    timeline_first = None
    timeline_second = None
    if dialogues:
        models = load_separation_models(device=device, backend=separation_backend, model_id=separation_model)

    try:
        for index, dialogue in enumerate(dialogues):
            start_sample = max(0, min(waveform.shape[-1], round(dialogue.start * input_sample_rate)))
            end_sample = max(start_sample, min(waveform.shape[-1], round(dialogue.end * input_sample_rate)))
            crop = waveform[..., start_sample:end_sample]
            if crop.shape[-1] == 0:
                continue
            first, second, output_sample_rate = separate_waveform(
                crop, input_sample_rate, models, num_steps, separate_chunk, _no_progress,
            )
            native_output_sample_rate = output_sample_rate
            if native_output_sample_rate != OUTPUT_SAMPLE_RATE:
                first = _resample_waveform(first, native_output_sample_rate, OUTPUT_SAMPLE_RATE)
                second = _resample_waveform(second, native_output_sample_rate, OUTPUT_SAMPLE_RATE)
                output_sample_rate = OUTPUT_SAMPLE_RATE
            if timeline_first is None:
                timeline_sample_rate = output_sample_rate
                output_length = max(
                    1, round(waveform.shape[-1] * output_sample_rate / input_sample_rate),
                )
                timeline_first = torch.zeros(1, output_length, dtype=waveform.dtype)
                timeline_second = torch.zeros_like(timeline_first)
            elif output_sample_rate != timeline_sample_rate:
                raise ValueError(
                    "Split separation returned inconsistent sample rates: "
                    f"{output_sample_rate} and {timeline_sample_rate}"
                )
            output_start = max(
                0, min(timeline_first.shape[-1], round(dialogue.start * output_sample_rate)),
            )
            output_end = max(
                output_start, min(timeline_first.shape[-1], round(dialogue.end * output_sample_rate)),
            )
            output_length = output_end - output_start
            first = _fit_channel(first, output_length)
            second = _fit_channel(second, output_length)
            mixture = _resample_waveform(crop, input_sample_rate, output_sample_rate)
            mixture = _fit_channel(mixture, output_length)
            conversation_dir = output_root / "conversations" / f"conversation_{index:05d}"
            metadata = {
                "conversation_idx": index,
                "start": dialogue.start,
                "end": dialogue.end,
                "duration": dialogue.duration,
                "speakers": dialogue.speakers,
                "segments": dialogue.segments,
                "sample_rate": output_sample_rate,
                "source_start_sample": start_sample,
                "source_end_sample": end_sample,
                "status": "complete",
            }
            stereo_path = write_conversation_stereo(
                conversation_dir, mixture, first, second, output_sample_rate, metadata,
            )
            timeline_first[..., output_start:output_end] = first
            timeline_second[..., output_start:output_end] = second
            manifest.append({**metadata, "stereo": str(stereo_path), "mixture": str(conversation_dir / "mixture.wav")})
    finally:
        if models is not None and hasattr(models, "to"):
            models.to(torch.device("cpu"))

    if timeline_first is None:
        timeline_first = torch.zeros(1, max(1, waveform.shape[-1]), dtype=waveform.dtype)
        timeline_second = torch.zeros_like(timeline_first)
    root_stereo = Path(output_prefix).parent / "audio.stereo.wav"
    save_stereo_wav(root_stereo, timeline_first, timeline_second, output_sample_rate)
    write_json(
        output_root / "conversations" / "manifest.json",
        {"output_kind": "conversation_collection", "sample_rate": output_sample_rate, "conversations": manifest},
    )
    write_json(
        phase_output_dir / "phase_03_dialogues" / "manifest.json",
        {"count": len(manifest), "dialogues": manifest},
    )
    return {"stereo": root_stereo, "segments": segments, "conversations": manifest}


def run_single_audio(
    audio_path_str,
    diarize_chunk=60.0,
    separate_chunk=30.0,
    diarization_backend="auto",
    diarization_model="pyannote/speaker-diarization-community-1",
    separation_backend="dialoguesidon",
    separation_model=None,
    output_prefix="output_speaker",
    output_dir=None,
    runtime_device="auto",
    num_steps=30,
    analyze_diarization=False,
    split_conversation=False,
):
    from .devices import resolve_device
    device = resolve_device(runtime_device, allow_cpu_fallback=True)
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    logger = get_logger("duplexchat")
    logger.info("execution_mode=full_stereo")
    logger.info("Diarization backend=%s model=%s device=%s", diarization_backend, diarization_model, device)
    logger.info("Separation backend=%s model=%s", separation_backend, separation_model or "default")

    phase_output_dir = resolve_output_dir(output_prefix, output_dir)
    phase_output_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = phase_output_dir / "phase_01_preprocess" / "audio_16k_mono.wav"
    with StepTimer(logger, "Step 0: Preprocess"):
        temp_wav = prepare_input(audio_path, phase_output_dir)

    segments = []
    if analyze_diarization or split_conversation:
        title = "Step 1: Speaker diarization" if split_conversation else "Step 1: Debug speaker diarization"
        with StepTimer(logger, title):
            diarize_pipeline, segments = diarize(
                temp_wav, phase_output_dir, diarization_model, diarization_backend, device, diarize_chunk, _no_progress,
            )
        if str(device).startswith("cuda"):
            release_diarization_gpu_memory(diarize_pipeline)
        del diarize_pipeline
    if split_conversation:
        with StepTimer(logger, "Step 2: Split conversation separation"):
            result = _run_split_conversation(
                temp_wav, phase_output_dir, phase_output_dir.parent, segments, device,
                separation_backend, separation_model, num_steps, separate_chunk, output_prefix,
                logger,
            )
        logger.info("Saved split stereo output: %s", result["stereo"])
        return result
    with StepTimer(logger, "Step 2: Full-input speech separation"):
        spk0, spk1, out_sr = separate(temp_wav, device, separation_backend, separation_model, num_steps, separate_chunk, _no_progress)

    with StepTimer(logger, "Step 3: Local reconstruction"):
        stereo = write_stereo(output_prefix, phase_output_dir, spk0, spk1, out_sr, separation_backend, separation_model)
    logger.info("Saved stereo output: %s", stereo)
    return {"stereo": stereo, "segments": segments}

def main():
    parser = argparse.ArgumentParser(description="Test DuplexChat on a single audio file.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file (mp3/wav)")
    parser.add_argument("--diarize-chunk", type=float, default=60.0, help="Max chunk duration (seconds) for Diarization")
    parser.add_argument("--separate-chunk", type=float, default=60.0, help="Chunk duration (seconds) for Separation")
    parser.add_argument("--diarization-backend", default="auto", help="Diarization backend: auto, pyannote, sortformer, diarizen")
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1", help="Diarization model id or alias")
    parser.add_argument("--separation-backend", default="dialoguesidon", help="Separation backend: dialoguesidon, sepformer, mossformer2")
    parser.add_argument("--separation-model", default=None, help="Separation model id or alias")
    parser.add_argument("--output-prefix", default="output_speaker", help="Output WAV prefix, e.g. runs/sortformer__sepformer/output")
    parser.add_argument("--output-dir", default=None, help="Directory for phase outputs and labels")

    args = parser.parse_args()
    run_single_audio(
        args.audio_path,
        args.diarize_chunk,
        args.separate_chunk,
        args.diarization_backend,
        args.diarization_model,
        args.separation_backend,
        args.separation_model,
        args.output_prefix,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
