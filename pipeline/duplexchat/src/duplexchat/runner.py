"""Purpose: Execute DuplexChat preprocessing through stereo export.

Inputs: One mixture path, phase/model settings, output location.
Outputs: One stereo WAV file and inspectable phase artifacts.
"""
import torch
from pathlib import Path

from .preprocess import prepare_input
from .diarization import diarize
from .dialogues import summarize
from .separation import separate, separate_waveform
from .reconstruct import write_stereo
from .audio import load_wav_tensor
from .separation_backend import load_separation_models

from core.orchestration.logging_style import StepTimer, get_logger
from core.outputs import write_json

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
    scale=False,
):
    from .devices import resolve_device
    device = resolve_device(runtime_device, allow_cpu_fallback=True)
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    logger = get_logger("duplexchat")
    logger.info("execution_mode=%s", "per_conversation_scale" if scale else "full_input_debug")
    logger.info("Diarization backend=%s model=%s device=%s", diarization_backend, diarization_model, device)
    logger.info("Separation backend=%s model=%s", separation_backend, separation_model or "default")

    phase_output_dir = resolve_output_dir(output_prefix, output_dir)
    phase_output_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = phase_output_dir / "phase_01_preprocess" / "audio_16k_mono.wav"
    with StepTimer(logger, "Step 0: Preprocess"):
        temp_wav = prepare_input(audio_path, phase_output_dir)

    with StepTimer(logger, "Step 1: Speaker Diarization"):
        diarize_pipeline, segments = diarize(temp_wav, phase_output_dir, diarization_model, diarization_backend, device, diarize_chunk, _no_progress)
    with StepTimer(logger, "Step 2: Detect two-speaker conversations"):
        conversations, valid_dialogues = summarize(segments)
        logger.info("conversations=%d valid_two_speaker_conversations=%d", len(conversations), len(valid_dialogues))
        write_json(
            phase_output_dir / "conversations_2spk.json",
            {
                "execution_mode": "per_conversation_scale" if scale else "full_input_debug",
                "conversations": [
                    {"id": f"conversation_{index:05d}", "start": item.start, "end": item.end,
                     "duration": item.duration, "speakers": sorted(item.speakers), "segments": item.segments}
                    for index, item in enumerate(valid_dialogues)
                ],
            },
        )

    if str(device).startswith("cuda"):
        release_diarization_gpu_memory(diarize_pipeline)
    del diarize_pipeline

    if scale:
        waveform, sample_rate = load_wav_tensor(temp_wav)
        collection_dir = Path(output_prefix).parent / "conversations"
        with StepTimer(logger, "Step 3: Per-conversation speech separation"):
            models = load_separation_models(device=device, backend=separation_backend, model_id=separation_model)
            conversations = []
            for index, dialogue in enumerate(valid_dialogues):
                conversation_id = f"conversation_{index:05d}"
                start = max(0, int(round(dialogue.start * sample_rate)))
                end = min(waveform.shape[-1], int(round(dialogue.end * sample_rate)))
                if end <= start:
                    continue
                directory = collection_dir / conversation_id
                directory.mkdir(parents=True, exist_ok=True)
                mixture = waveform[:, start:end]
                mixture_path = directory / "mixture.wav"
                from core.outputs import save_stereo_wav, save_wav
                save_wav(mixture_path, mixture, sample_rate)
                spk_a, spk_b, out_sr = separate_waveform(
                    mixture, sample_rate, models, num_steps, separate_chunk, _no_progress
                )
                stereo = directory / "audio.stereo.wav"
                save_stereo_wav(stereo, spk_a, spk_b, out_sr)
                record = {
                    "id": conversation_id, "start": dialogue.start, "end": dialogue.end,
                    "duration": dialogue.duration, "speakers": sorted(dialogue.speakers),
                    "segments": dialogue.segments, "mixture": str(mixture_path),
                    "stereo": str(stereo), "sample_rate": out_sr,
                }
                write_json(directory / "metadata.json", record)
                conversations.append(record)
                logger.info("Separated %s (%.2fs)", conversation_id, dialogue.duration)
        write_json(collection_dir / "manifest.json", {"conversations": conversations})
        logger.info("Saved %d conversation collections: %s", len(conversations), collection_dir)
        return {"stereo": None, "segments": segments, "valid_dialogues": valid_dialogues, "conversations": conversations}
    with StepTimer(logger, "Step 3: Full-input speech separation"):
        spk0, spk1, out_sr = separate(temp_wav, device, separation_backend, separation_model, num_steps, separate_chunk, _no_progress)

    with StepTimer(logger, "Step 4: Local reconstruction"):
        stereo = write_stereo(output_prefix, phase_output_dir, spk0, spk1, out_sr, separation_backend, separation_model)
    logger.info("Saved stereo output: %s", stereo)
    return {"stereo": stereo, "segments": segments, "valid_dialogues": valid_dialogues}

def main():
    parser = argparse.ArgumentParser(description="Test DuplexChat on a single audio file.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file (mp3/wav)")
    parser.add_argument("--diarize-chunk", type=float, default=60.0, help="Max chunk duration (seconds) for Diarization")
    parser.add_argument("--separate-chunk", type=float, default=30.0, help="Chunk duration (seconds) for Separation")
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
