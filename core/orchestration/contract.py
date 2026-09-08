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


def validate_tracks(source: Path, tracks: list[Path]) -> float:
    import numpy as np
    import soundfile as sf

    if len(tracks) != 2 or tracks[0].resolve() == tracks[1].resolve():
        raise ValueError('Expected exactly two distinct speaker tracks')
    reference = sf.info(source)
    duration = reference.frames / reference.samplerate
    if duration <= 0:
        raise ValueError('Empty input timeline')
    for path in tracks:
        info = sf.info(path)
        if info.channels != 1 or info.frames <= 0:
            raise ValueError(f'Expected nonempty mono track: {path}')
        if abs(info.frames / info.samplerate - duration) > max(1 / info.samplerate, 1 / reference.samplerate):
            raise ValueError(f'Track timeline differs from input: {path}')
        for block in sf.blocks(path, blocksize=65536):
            if not np.isfinite(block).all():
                raise ValueError(f'Nonfinite samples: {path}')
    return duration


def validate_conversation_collection(conversations: list[dict]) -> float:
    """Validate DuplexChat's native per-conversation two-track output."""
    if not conversations:
        raise ValueError("DuplexChat scale output contains no valid two-speaker conversations")
    duration = 0.0
    for conversation in conversations:
        mixture = Path(conversation["mixture"])
        tracks = [Path(path) for path in conversation["tracks"]]
        duration += validate_tracks(mixture, tracks)
    return duration


def _output_kind(metadata: dict) -> str:
    return str(metadata.get("output_kind", "two_full_tracks"))


def validate_output(source: Path, tracks: list[Path], metadata: dict) -> float:
    if _output_kind(metadata) == "conversation_collection":
        return validate_conversation_collection(metadata.get("conversations", []))
    return validate_tracks(source, tracks)


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
            tracks = [output / name for name in ('speakerA.wav', 'speakerB.wav')]
            validate_tracks(source, tracks)
            if result['track_sha256'] != [sha256(path) for path in tracks]:
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
        tracks, metadata = adapter(source, output, config)
        duration = validate_output(source, tracks, metadata)
        canonical = [output / name for name in ('speakerA.wav', 'speakerB.wav')]
        if _output_kind(metadata) != "conversation_collection":
            for original, target in zip(tracks, canonical):
                if original.resolve() != target.resolve():
                    shutil.copy2(original, target)
        elapsed = time.perf_counter() - started
        result.update(status='complete', duration_sec=duration, inference_seconds=elapsed,
                      rtf=elapsed / duration, metadata=metadata,
                      track_sha256=([sha256(path) for path in canonical]
                                    if _output_kind(metadata) != "conversation_collection" else []))
    except Exception as exc:
        trace = traceback.format_exc()
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      traceback=trace, inference_seconds=time.perf_counter() - started)
        get_logger(pipeline).error("Sample %s failed: %s: %s", sample['key'], type(exc).__name__, exc)
    write_json(output / 'run.json', result)
    return result
