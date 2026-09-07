from pathlib import Path

from duplexchat_pipe.outputs import save_wav, write_json


def write_tracks(output_prefix, phase_dir, speaker_a, speaker_b, sample_rate, backend, model):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_a, out_b = prefix.parent / f"{prefix.name}A.wav", prefix.parent / f"{prefix.name}B.wav"
    save_wav(out_a, speaker_a, sample_rate)
    save_wav(out_b, speaker_b, sample_rate)
    save_wav(phase_dir / "phase_04_separation" / "speaker_A.wav", speaker_a, sample_rate)
    save_wav(phase_dir / "phase_04_separation" / "speaker_B.wav", speaker_b, sample_rate)
    write_json(phase_dir / "phase_04_separation" / "separation.json", {"backend": backend, "model": model, "sample_rate": sample_rate, "speaker_A": str(out_a), "speaker_B": str(out_b)})
    return out_a, out_b
