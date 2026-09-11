"""Purpose: Run DuplexChat speaker diarization as a named phase.

Inputs: Normalized audio, model/backend settings, device and chunk limit.
Outputs: Diarization model handle and speaker-labelled segments.
"""
from pathlib import Path

from .diarization_backend import load_diarization_pipeline, run_diarization
from core.orchestration.logging_style import get_logger
from core.outputs import write_diarization_phase, write_json


def diarize(audio: Path, output_dir: Path, model: str, backend: str, device: str, chunk: float, progress):
    model_instance = load_diarization_pipeline(model, device=device, backend=backend)
    diagnostics: dict = {}
    try:
        segments = run_diarization(
            model_instance, audio, max_chunk_dur=chunk, progress_callback=progress, diagnostics=diagnostics,
        )
    finally:
        progress("close", 0)
    write_diarization_phase(output_dir, segments, model=model, backend=backend)
    write_json(output_dir / "phase_02_diarization" / "linking.json", diagnostics)
    get_logger("duplexchat").info("speaker_linking=%s", diagnostics)
    return model_instance, segments
