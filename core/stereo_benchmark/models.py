"""Lazy optional model metrics. Each failure is converted to an unavailable result."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np


def unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def _per_channel(metric: Callable[[np.ndarray, int], dict], left: np.ndarray, right: np.ndarray, sample_rate: int) -> dict:
    left_result, right_result = metric(left, sample_rate), metric(right, sample_rate)
    result = {"left": left_result, "right": right_result}
    if left_result.get("status") == right_result.get("status") == "ok":
        shared = set(left_result) & set(right_result) - {"status"}
        result["mean"] = {name: float((left_result[name] + right_result[name]) / 2) for name in shared}
    return result


def acoustic_metrics(left: np.ndarray, right: np.ndarray, sample_rate: int, device: str, dnsmos_model_dir: Path | None = None) -> dict:
    """Compute reference-free SQUIM where its bundled model is available.

    DNSMOS runs only when both official P.835 ONNX assets are available locally.
    """
    return {
        "dnsmos": _dnsmos_metrics(left, right, sample_rate, dnsmos_model_dir),
        "squim": _per_channel(lambda audio, sr: _squim(audio, sr, device), left, right, sample_rate),
    }


def _dnsmos_metrics(left: np.ndarray, right: np.ndarray, sample_rate: int, model_dir: Path | None) -> dict:
    if model_dir is None:
        missing = unavailable("DNSMOS model directory was not configured")
        return {"left": missing, "right": missing}
    from .dnsmos import DNSMOSScorer

    scorer = DNSMOSScorer(model_dir)
    return _per_channel(scorer.score, left, right, sample_rate)


@lru_cache(maxsize=2)
def _squim_model(device: str):
    import torch
    import torchaudio

    model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(device).eval()
    return model


def _squim(audio: np.ndarray, sample_rate: int, device: str) -> dict:
    try:
        import torch
        import torchaudio.functional as ta_functional

        waveform = torch.from_numpy(audio).unsqueeze(0)
        if sample_rate != 16000:
            waveform = ta_functional.resample(waveform, sample_rate, 16000)
        
        chunk_size = 16000 * 10
        stois, pesqs, si_sdrs = [], [], []
        model = _squim_model(device)
        for start in range(0, waveform.shape[1], chunk_size):
            chunk = waveform[:, start:start + chunk_size]
            if chunk.shape[1] < 16000 * 1:
                continue
            with torch.inference_mode():
                stoi, pesq, si_sdr = model(chunk.to(device))
                stois.append(float(stoi.item()))
                pesqs.append(float(pesq.item()))
                si_sdrs.append(float(si_sdr.item()))
        
        if not stois:
            return unavailable("Audio too short for SQUIM")
        
        return {
            "status": "ok", "sq_stoi": sum(stois)/len(stois), "sq_pesq": sum(pesqs)/len(pesqs),
            "sq_si_sdr": sum(si_sdrs)/len(si_sdrs),
        }
    except Exception as error:  # optional model may require an unavailable download/runtime
        return unavailable(f"SQUIM unavailable: {type(error).__name__}: {error}")


def speaker_metrics(left: np.ndarray, right: np.ndarray, sample_rate: int, left_mask: np.ndarray, right_mask: np.ndarray, frame_sec: float, device: str) -> dict:
    """ITC/ITD from cached SpeechBrain ECAPA embeddings; no ground truth is used."""
    try:
        left_embeddings = _embeddings(left, sample_rate, left_mask, frame_sec, device)
        right_embeddings = _embeddings(right, sample_rate, right_mask, frame_sec, device)
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


def _embeddings(audio: np.ndarray, sample_rate: int, mask: np.ndarray, frame_sec: float, device: str) -> np.ndarray | None:
    import torch
    import torchaudio.functional as ta_functional

    frame_samples = max(1, round(frame_sec * sample_rate))
    speech = np.concatenate([audio[i * frame_samples:min((i + 1) * frame_samples, len(audio))] for i, active in enumerate(mask) if active]) if mask.any() else np.array([], dtype=np.float32)
    target = 16000
    signal = torch.from_numpy(speech).float().unsqueeze(0)
    if sample_rate != target and signal.numel():
        signal = ta_functional.resample(signal, sample_rate, target)
    window = target * 3
    if signal.shape[1] < window:
        return None
    encoder = _speaker_encoder(device)
    embeddings = []
    for start in range(0, signal.shape[1] - window + 1, window):
        with torch.inference_mode():
            embedding = encoder.encode_batch(signal[:, start:start + window].to(device)).squeeze().detach().cpu().numpy()
        embeddings.append(embedding)
    return np.asarray(embeddings) if embeddings else None


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
