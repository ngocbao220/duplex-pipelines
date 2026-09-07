from pathlib import Path

from duplexchat_pipe.diarize import load_diarization_pipeline, run_diarization
from duplexchat_pipe.outputs import write_diarization_phase


def diarize(audio: Path, output_dir: Path, model: str, backend: str, device: str, chunk: float, progress):
    model_instance = load_diarization_pipeline(model, device=device, backend=backend)
    try:
        segments = run_diarization(model_instance, audio, max_chunk_dur=chunk, progress_callback=progress)
    finally:
        progress("close", 0)
    write_diarization_phase(output_dir, segments, model=model, backend=backend)
    return model_instance, segments
