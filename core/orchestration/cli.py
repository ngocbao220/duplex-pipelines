"""CLI primitives shared by the three independently installed pipelines."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def build_pipeline_parser(name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run the {name} speaker-separation pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser("single", help="Run one supplied mixture audio file.")
    single.add_argument("--input", type=Path, required=True)
    single.add_argument("--output-dir", type=Path, required=True)
    single.add_argument("--debug", action="store_true")
    if name == "duplexchat":
        single.add_argument("--split_conversation", action="store_true")
        single.add_argument("--diarization-backend", default="auto")
        single.add_argument(
            "--diarization-model",
            default="nvidia/diar_streaming_sortformer_4spk-v2.1",
        )
    single.add_argument("--gt-speaker-a", type=Path)
    single.add_argument("--gt-speaker-b", type=Path)

    otospeech = commands.add_parser("otospeech", help="Run a bounded OtoSpeech reference benchmark.")
    otospeech.add_argument("--otospeech-root", type=Path)
    otospeech.add_argument("--download-dir", type=Path, default=Path("data/otospeech"))
    otospeech.add_argument("--max-gb", type=float, default=10.0)
    otospeech.add_argument("--max-samples", type=int, default=None)
    otospeech.add_argument("--max-seconds", type=float, default=None)
    otospeech.add_argument("--output-root", type=Path, default=Path("outputs"))
    otospeech.add_argument("--debug", action="store_true")
    if name == "duplexchat":
        otospeech.add_argument("--split_conversation", action="store_true")
    otospeech.add_argument("--force", action="store_true")
    return parser


def run_pipeline_command(name: str, argv: list[str] | None = None) -> int:
    args = build_pipeline_parser(name).parse_args(argv)
    if args.command == "single":
        if not args.input.is_file():
            raise SystemExit(f"Input audio file does not exist: {args.input}")
        if bool(args.gt_speaker_a) != bool(args.gt_speaker_b):
            raise SystemExit("--gt-speaker-a and --gt-speaker-b must be provided together")
        for reference in (args.gt_speaker_a, args.gt_speaker_b):
            if reference is not None and not reference.is_file():
                raise SystemExit(f"Ground-truth audio file does not exist: {reference}")
        from .single import run_single
        return run_single(name, args.input.resolve(), args.output_dir.resolve(), args.debug,
                           getattr(args, "split_conversation", False),
                           getattr(args, "diarization_backend", "auto"),
                           getattr(args, "diarization_model", None),
                           args.gt_speaker_a.resolve() if args.gt_speaker_a else None,
                          args.gt_speaker_b.resolve() if args.gt_speaker_b else None)

    from types import SimpleNamespace
    from core import benchmark
    from core.config import load_config
    from core.outputs import save_wav
    from .runner import run_otospeech

    if args.max_gb <= 0 or (args.max_samples is not None and args.max_samples <= 0) or (
        args.max_seconds is not None and args.max_seconds <= 0
    ):
        raise SystemExit("OtoSpeech limits must be positive")
    options = SimpleNamespace(
        pipeline=name, config=ROOT / "configs/config.json", otospeech_root=args.otospeech_root,
        otospeech_repo="otoearth/otoSpeech-full-duplex-turn-104h", otospeech_local_dir=args.download_dir,
        max_gb=args.max_gb, max_samples=args.max_samples, max_seconds=args.max_seconds,
        output_root=args.output_root, pred_root=None, mixture_root=None, benchmark_output=None,
        sample_rate=16000, vad_threshold_db=-40.0, crosstalk_threshold_db=-20.0,
        debug=args.debug, split_conversation=getattr(args, "split_conversation", False), force=args.force,
        vilier_config=ROOT / "configs/vilier.json",
        duplexchat_config=ROOT / "configs/duplexchat.json",
    )
    return run_otospeech(load_config(options.config), options, benchmark, save_wav)
