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
        single.add_argument("--separate-chunk", type=float, default=120.0)
        single.add_argument("--device-ids", type=int, nargs="+")
    if name != "duplexchat":
        single.add_argument("--gt-speaker-a", type=Path)
        single.add_argument("--gt-speaker-b", type=Path)
    return parser


def run_pipeline_command(name: str, argv: list[str] | None = None) -> int:
    args = build_pipeline_parser(name).parse_args(argv)
    if args.command == "single":
        if not args.input.is_file():
            raise SystemExit(f"Input audio file does not exist: {args.input}")
        if name != "duplexchat" and bool(args.gt_speaker_a) != bool(args.gt_speaker_b):
            raise SystemExit("--gt-speaker-a and --gt-speaker-b must be provided together")
        for reference in (() if name == "duplexchat" else (args.gt_speaker_a, args.gt_speaker_b)):
            if reference is not None and not reference.is_file():
                raise SystemExit(f"Ground-truth audio file does not exist: {reference}")
        from .single import run_single
        return run_single(name, args.input.resolve(), args.output_dir.resolve(), args.debug,
                           getattr(args, "separate_chunk", 120.0), getattr(args, "device_ids", None),
                           args.gt_speaker_a.resolve() if getattr(args, "gt_speaker_a", None) else None,
                           args.gt_speaker_b.resolve() if getattr(args, "gt_speaker_b", None) else None)

    raise AssertionError(f"Unsupported command: {args.command}")
