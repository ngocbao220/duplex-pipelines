#!/usr/bin/env python3
"""Run a reference-free benchmark on one timeline-aligned stereo audio file."""

import argparse
from pathlib import Path

from core.stereo_benchmark.runner import print_summary, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Two-channel speaker-separated audio")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--debug", action="store_true", help="Write separated channels and detailed event files")
    args = parser.parse_args()
    report, report_path = run_benchmark(args.audio, args.output_dir, args.device, args.debug)
    print_summary(report, report_path)


if __name__ == "__main__":
    main()
