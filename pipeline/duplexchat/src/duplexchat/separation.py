"""Purpose: Run DuplexChat full-timeline speaker separation.

Inputs: Normalized audio, separator settings, device and chunk settings.
Outputs: Two separated waveforms and their sample rate.
"""
from .audio import load_wav_tensor
from .separation_backend import load_separation_models, run_separation


def separate(audio_path, device, backend, model, steps, chunk, progress):
    models = load_separation_models(device=device, backend=backend, model_id=model)
    waveform, sample_rate = load_wav_tensor(audio_path)
    try:
        return run_separation(waveform, sample_rate, num_steps=steps, models=models, chunk_seconds=chunk, overlap_seconds=max(1.0, chunk / 6.0), progress_callback=progress)
    finally:
        progress("close", 0)
