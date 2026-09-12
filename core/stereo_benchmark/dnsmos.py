"""Local Microsoft DNSMOS P.835 scoring with two checked-in-external ONNX assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000
WINDOW_SEC = 9.01
PRIMARY_MODEL = "sig_bak_ovr.onnx"
P808_MODEL = "model_v8.onnx"


class DNSMOSScorer:
    """Reference-free DNSMOS P.835 scorer; higher SIG/BAK/OVRL/P808 values are better.

    This reproduces Microsoft's local P.835 preprocessing: 16 kHz, 9.01-second
    windows on one-second hops, and the official non-personalized calibration.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.error: str | None = self._missing_assets()
        self.primary = None
        self.p808 = None
        if self.error is None:
            try:
                import onnxruntime as ort

                self.primary = ort.InferenceSession(str(self.model_dir / PRIMARY_MODEL))
                self.p808 = ort.InferenceSession(str(self.model_dir / P808_MODEL))
            except Exception as error:  # model/runtime is optional for the complete benchmark
                self.error = f"DNSMOS unavailable: {type(error).__name__}: {error}"

    def score(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Score one mono track without any clean reference signal."""
        if self.error is not None:
            return {"status": "unavailable", "reason": self.error}
        try:
            import librosa

            signal = np.asarray(audio, dtype=np.float32)
            if sample_rate != SAMPLE_RATE:
                signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            if not signal.size:
                return {"status": "unavailable", "reason": "DNSMOS input is empty"}
            window_samples = int(WINDOW_SEC * SAMPLE_RATE)
            while signal.size < window_samples:
                signal = np.append(signal, signal)
            scores = []
            num_hops = int(np.floor(signal.size / SAMPLE_RATE) - WINDOW_SEC) + 1
            for index in range(max(0, num_hops)):
                chunk = signal[index * SAMPLE_RATE : index * SAMPLE_RATE + window_samples]
                if chunk.size != window_samples:
                    continue
                raw_sig, raw_bak, raw_ovrl = self.primary.run(
                    None, {self.primary.get_inputs()[0].name: chunk[None, :].astype(np.float32)}
                )[0][0]
                features = self._mel_features(librosa, chunk[:-160])[None, :, :].astype(np.float32)
                p808 = self.p808.run(None, {self.p808.get_inputs()[0].name: features})[0][0][0]
                sig, bak, ovrl = self._calibrate(raw_sig, raw_bak, raw_ovrl)
                scores.append((sig, bak, ovrl, p808))
            if not scores:
                return {"status": "unavailable", "reason": "DNSMOS produced no complete analysis window"}
            mean = np.mean(np.asarray(scores), axis=0)
            return {
                "status": "ok", "sig": float(mean[0]), "bak": float(mean[1]), "ovrl": float(mean[2]),
                "p808_mos": float(mean[3]), "window_count": len(scores),
            }
        except Exception as error:
            return {"status": "unavailable", "reason": f"DNSMOS unavailable: {type(error).__name__}: {error}"}

    def _missing_assets(self) -> str | None:
        missing = [name for name in (PRIMARY_MODEL, P808_MODEL) if not (self.model_dir / name).is_file()]
        if missing:
            return f"DNSMOS model asset missing: {', '.join(missing)} in {self.model_dir}"
        return None

    @staticmethod
    def _mel_features(librosa, audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=321, hop_length=160, n_mels=120)
        return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T

    @staticmethod
    def _calibrate(sig: float, bak: float, ovrl: float) -> tuple[float, float, float]:
        # Official DNSMOS P.835 non-personalized calibration polynomials.
        return (
            float(np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig)),
            float(np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak)),
            float(np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovrl)),
        )
