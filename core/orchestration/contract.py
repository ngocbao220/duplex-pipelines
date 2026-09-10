from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
import uuid
from pathlib import Path

from .logging_style import get_logger


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n')
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(pipeline: str, source: Path, config: dict, code: str) -> str:
    payload = [pipeline, sha256(source), config, code]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


STEREO_FILENAME = "audio.stereo.wav"


def validate_stereo(source: Path, stereo: Path) -> float:
    import numpy as np
    import soundfile as sf

    reference = sf.info(source)
    duration = reference.frames / reference.samplerate
    if duration <= 0:
        raise ValueError('Empty input timeline')
    info = sf.info(stereo)
    if info.channels != 2 or info.frames <= 0:
        raise ValueError(f'Expected nonempty stereo output: {stereo}')
    if abs(info.frames / info.samplerate - duration) > max(1 / info.samplerate, 1 / reference.samplerate):
        raise ValueError(f'Stereo timeline differs from input: {stereo}')
    for block in sf.blocks(stereo, blocksize=65536):
        if not np.isfinite(block).all():
            raise ValueError(f'Nonfinite samples: {stereo}')
    return duration


def validate_conversation_collection(conversations: list[dict]) -> float:
    """Validate DuplexChat's native per-conversation two-track output."""
    if not conversations:
        raise ValueError("DuplexChat scale output contains no valid two-speaker conversations")
    duration = 0.0
    for conversation in conversations:
        mixture = Path(conversation["mixture"])
        duration += validate_stereo(mixture, Path(conversation["stereo"]))
    return duration


def _output_kind(metadata: dict) -> str:
    return str(metadata.get("output_kind", "full_stereo"))


def validate_output(source: Path, stereo: Path | None, metadata: dict) -> float:
    if _output_kind(metadata) == "conversation_collection":
        return validate_conversation_collection(metadata.get("conversations", []))
    if stereo is None:
        raise ValueError("Full-input output requires a stereo WAV")
    return validate_stereo(source, stereo)


def vilier_tracks(manifest: Path) -> list[Path]:
    speakers = json.loads(manifest.read_text())['speakers']
    if len(speakers) != 2:
        raise ValueError(f'Vilier must produce exactly two speakers; got {len(speakers)}')
    return [manifest.parent / speaker['track_wav'] for speaker in speakers]


def reusable(output: Path, identity: str, source: Path) -> dict | None:
    try:
        result = json.loads((output / 'run.json').read_text())
        if result['status'] != 'complete' or result['fingerprint'] != identity:
            return None
        metadata = result.get("metadata", {})
        if _output_kind(metadata) == "conversation_collection":
            validate_conversation_collection(metadata.get("conversations", []))
        else:
            stereo = output / STEREO_FILENAME
            validate_stereo(source, stereo)
            if result['audio_sha256'] != sha256(stereo):
                return None
        return result
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def run_sample(pipeline, sample, output, config, code, adapter, force=False) -> dict:
    source = Path(sample['mixture'])
    output = Path(output)
    started = time.perf_counter()
    result = {'key': sample['key'], 'pipeline': pipeline, 'config': config, 'code': code,
              'input': str(source), 'status': 'running', 'resumed': False}
    try:
        identity = fingerprint(pipeline, source, config, code)
        previous = reusable(output, identity, source) if not force else None
        if previous:
            return {**previous, 'resumed': True}
        if output.exists():
            archive = output.parent / '.history' / uuid.uuid4().hex / output.name
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
        output.mkdir(parents=True, exist_ok=True)
        result['fingerprint'] = identity
        write_json(output / 'run.json', result)
        stereo, metadata = adapter(source, output, config)
        duration = validate_output(source, stereo, metadata)
        canonical = output / STEREO_FILENAME
        if _output_kind(metadata) != "conversation_collection":
            if stereo.resolve() != canonical.resolve():
                shutil.copy2(stereo, canonical)
        elapsed = time.perf_counter() - started
        result.update(status='complete', duration_sec=duration, inference_seconds=elapsed,
                      rtf=elapsed / duration, metadata=metadata,
                      audio_sha256=(sha256(canonical)
                                    if _output_kind(metadata) != "conversation_collection" else None))
    except Exception as exc:
        trace = traceback.format_exc()
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      traceback=trace, inference_seconds=time.perf_counter() - started)
        get_logger(pipeline).error("Sample %s failed: %s: %s", sample['key'], type(exc).__name__, exc)
    write_json(output / 'run.json', result)
    return result
