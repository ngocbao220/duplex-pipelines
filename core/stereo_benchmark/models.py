"""Lazy optional model metrics for already prepared 16 kHz speech."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000
SQUIM_WINDOW_SAMPLES = SAMPLE_RATE * 10
SQUIM_BATCH_SIZE = 8
SPEAKER_WINDOW_SAMPLES = SAMPLE_RATE * 3
SPEAKER_BATCH_SIZE = 16


def unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def prepare_speech(audio: np.ndarray, sample_rate: int, mask: np.ndarray, frame_sec: float) -> np.ndarray:
    """Extract VAD-active samples and resample once for all model metrics."""
    frame_samples = max(1, round(frame_sec * sample_rate))
    speech = np.concatenate(
        [audio[index * frame_samples:min((index + 1) * frame_samples, len(audio))]
         for index, active in enumerate(mask) if active]
    ) if mask.any() else np.array([], dtype=np.float32)
    if sample_rate == SAMPLE_RATE or not speech.size:
        return np.asarray(speech, dtype=np.float32)
    import torch
    import torchaudio.functional as ta_functional

    return ta_functional.resample(torch.from_numpy(speech).unsqueeze(0), sample_rate, SAMPLE_RATE).squeeze(0).numpy()


def acoustic_metrics(left: np.ndarray, right: np.ndarray, device: str, dnsmos_model_dir: Path | None = None) -> dict:
    """Compute reference-free acoustic metrics from prepared 16 kHz speech."""
    squim_left, squim_right = _squim_many((left, right), device)
    return {
        "dnsmos": _dnsmos_metrics(left, right, dnsmos_model_dir),
        "squim": _combine_channels(squim_left, squim_right),
    }


@lru_cache(maxsize=4)
def _dnsmos_scorer(model_dir: Path):
    from .dnsmos import DNSMOSScorer

    return DNSMOSScorer(model_dir)


def _dnsmos_metrics(left: np.ndarray, right: np.ndarray, model_dir: Path | None) -> dict:
    if model_dir is None:
        missing = unavailable("DNSMOS model directory was not configured")
        return {"left": missing, "right": missing}
    scorer = _dnsmos_scorer(Path(model_dir))
    return _combine_channels(scorer.score(left, SAMPLE_RATE), scorer.score(right, SAMPLE_RATE))


@lru_cache(maxsize=2)
def _squim_model(device: str):
    import torchaudio

    return torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(device).eval()


def _squim_many(audios: tuple[np.ndarray, np.ndarray], device: str) -> tuple[dict, dict]:
    """Score equal-length chunks together without changing per-track averaging."""
    try:
        import torch

        model = _squim_model(device)
        chunks: dict[int, list[tuple[int, object]]] = defaultdict(list)
        values = [[], []]
        for channel, audio in enumerate(audios):
            waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
            for start in range(0, waveform.numel(), SQUIM_WINDOW_SAMPLES):
                chunk = waveform[start:start + SQUIM_WINDOW_SAMPLES]
                if chunk.numel() >= SAMPLE_RATE:
                    chunks[chunk.numel()].append((channel, chunk))
        for length_chunks in chunks.values():
            for start in range(0, len(length_chunks), SQUIM_BATCH_SIZE):
                batch_items = length_chunks[start:start + SQUIM_BATCH_SIZE]
                batch = torch.stack([chunk for _, chunk in batch_items]).to(device)
                with torch.inference_mode():
                    stois, pesqs, si_sdrs = model(batch)
                for (channel, _), stoi, pesq, si_sdr in zip(batch_items, stois, pesqs, si_sdrs):
                    values[channel].append((float(stoi.item()), float(pesq.item()), float(si_sdr.item())))
        return tuple(_squim_result(channel_values) for channel_values in values)  # type: ignore[return-value]
    except Exception as error:
        failure = unavailable(f"SQUIM unavailable: {type(error).__name__}: {error}")
        return failure, failure


def _squim_result(values: list[tuple[float, float, float]]) -> dict:
    if not values:
        return unavailable("Audio too short for SQUIM")
    scores = np.asarray(values)
    return {
        "status": "ok", "sq_stoi": float(scores[:, 0].mean()), "sq_pesq": float(scores[:, 1].mean()),
        "sq_si_sdr": float(scores[:, 2].mean()),
    }


def speaker_metrics(left: np.ndarray, right: np.ndarray, device: str) -> dict:
    """ITC/ITD from cached SpeechBrain ECAPA embeddings; no ground truth is used."""
    try:
        left_embeddings, right_embeddings = _embeddings_many((left, right), device)
    except Exception as error:
        failure = unavailable(f"speaker encoder unavailable: {type(error).__name__}: {error}")
        return {"itc": {"left": failure, "right": failure, "mean": failure}, "itd": failure}
    left_itc = _itc(left_embeddings)
    right_itc = _itc(right_embeddings)
    result = {
        "itc": {"left": left_itc, "right": right_itc, "mean": _mean_available(left_itc, right_itc)},
        "itd": unavailable("insufficient_speech"),
    }
    if left_embeddings is not None and right_embeddings is not None:
        from .dynamics import cosine_distinctiveness

        left_centroid = _normalised_mean(left_embeddings)
        right_centroid = _normalised_mean(right_embeddings)
        cosine = float(np.dot(left_centroid, right_centroid))
        result["itd"] = {"status": "ok", "inter_track_cosine_similarity": cosine, "itd": cosine_distinctiveness(left_centroid, right_centroid)}
    return result


@lru_cache(maxsize=2)
def _speaker_encoder(device: str):
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})


def _embeddings_many(audios: tuple[np.ndarray, np.ndarray], device: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    import torch

    windows: list[tuple[int, object]] = []
    for channel, audio in enumerate(audios):
        signal = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        for start in range(0, signal.numel() - SPEAKER_WINDOW_SAMPLES + 1, SPEAKER_WINDOW_SAMPLES):
            windows.append((channel, signal[start:start + SPEAKER_WINDOW_SAMPLES]))
    if not windows:
        return None, None
    encoder = _speaker_encoder(device)
    values = [[], []]
    for start in range(0, len(windows), SPEAKER_BATCH_SIZE):
        batch_items = windows[start:start + SPEAKER_BATCH_SIZE]
        batch = torch.stack([window for _, window in batch_items]).to(device)
        with torch.inference_mode():
            embeddings = encoder.encode_batch(batch).detach().cpu().numpy().reshape(len(batch_items), -1)
        for (channel, _), embedding in zip(batch_items, embeddings):
            values[channel].append(embedding)
    return tuple(np.asarray(channel_values) if channel_values else None for channel_values in values)  # type: ignore[return-value]


def _combine_channels(left: dict, right: dict) -> dict:
    result = {"left": left, "right": right}
    if left.get("status") == right.get("status") == "ok":
        shared = set(left) & set(right) - {"status"}
        result["mean"] = {name: float((left[name] + right[name]) / 2) for name in shared}
    return result


def _itc(embeddings: np.ndarray | None) -> dict:
    if embeddings is None or len(embeddings) < 2:
        return unavailable("insufficient_speech")
    centroid = _normalised_mean(embeddings)
    similarities = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) @ centroid
    return {"status": "ok", "itc": float(np.mean(similarities)), "window_count": len(embeddings)}


def _normalised_mean(embeddings: np.ndarray) -> np.ndarray:
    centroid = np.mean(embeddings, axis=0)
    return centroid / np.linalg.norm(centroid)


def _mean_available(left: dict, right: dict) -> dict:
    if left.get("status") != "ok" or right.get("status") != "ok":
        return unavailable("insufficient_speech")
    return {"status": "ok", "itc": float((left["itc"] + right["itc"]) / 2)}
