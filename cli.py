"""Run the shared three-pipeline OtoSpeech comparison."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    otospeech = commands.add_parser("compare-otospeech", help="Mix once, run all pipelines, then compare GT scores.")
    otospeech.add_argument("--otospeech-root", type=Path)
    otospeech.add_argument("--download-dir", type=Path, default=Path("data/otospeech"))
    otospeech.add_argument("--max-gb", type=float, default=10.0)
    otospeech.add_argument("--max-samples", type=int, default=None)
    otospeech.add_argument("--max-seconds", type=float, default=None)
    otospeech.add_argument("--output-root", type=Path, default=Path("outputs"))
    otospeech.add_argument("--debug", action="store_true")
    otospeech.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_gb <= 0 or (args.max_samples is not None and args.max_samples <= 0) or (
        args.max_seconds is not None and args.max_seconds <= 0
    ):
        raise SystemExit("OtoSpeech limits must be positive")
    from core import benchmark
    from core.config import load_config
    from core.outputs import save_wav
    from core.orchestration.runner import run_otospeech

    root = Path(__file__).resolve().parent
    options = SimpleNamespace(
        pipeline="all", config=root / "configs/config.json", otospeech_root=args.otospeech_root,
        otospeech_repo="otoearth/otoSpeech-full-duplex-turn-104h", otospeech_local_dir=args.download_dir,
        max_gb=args.max_gb, max_samples=args.max_samples, max_seconds=args.max_seconds,
        output_root=args.output_root, pred_root=None, mixture_root=None, benchmark_output=None,
        sample_rate=16000, vad_threshold_db=-40.0, crosstalk_threshold_db=-20.0,
        debug=args.debug, force=args.force, vilier_config=root / "configs/vilier.json",
        duplexchat_config=root / "configs/duplexchat.json",
    )
    return run_otospeech(load_config(options.config), options, benchmark, save_wav)


if __name__ == "__main__":
    raise SystemExit(main())
