from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat.runner import _compact_path, _log_run_summary  # noqa: E402


class _CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)


def test_run_summary_lists_devices_conversations_and_phase_details(tmp_path):
    logger = _CaptureLogger()
    _log_run_summary(
        logger,
        tmp_path / "input.wav",
        tmp_path / "phases" / "phase_01_preprocess" / "audio_16k_mono.wav",
        tmp_path,
        tmp_path / "phases",
        tmp_path / "conversations" / "manifest.json",
        ["cuda:0", "cuda:1"],
        3,
        96,
        {"preprocess": 0.25, "diarization": 21.5, "separation": 151.5},
    )

    output = "\n".join(logger.messages)
    assert "diarization=cuda:0" in output
    assert "['cuda:0', 'cuda:1']" in output
    assert "Detected conversations: 3" in output
    assert "Preprocess" in output and "mono 16 kHz" in output and "0.25s" in output
    assert "Speaker diarization" in output and "96 segments" in output and "21.50s" in output
    assert "Dialogue separation" in output and "3 stereo 24 kHz WAV" in output and "151.50s" in output


def test_summary_compacts_long_paths_without_losing_the_filename():
    path = "/kaggle/input/datasets/ngocbaotrinhtuan/inputs/easy_1.wav"

    compacted = _compact_path(path, width=24)

    assert len(compacted) == 24
    assert compacted.startswith("...")
    assert compacted.endswith("easy_1.wav")