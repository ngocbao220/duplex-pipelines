"""Purpose: Run the fixed Vilier profile through two-track normalization.

Inputs: Mixture source path, output directory and Vilier configuration.
Outputs: Contract-ready speaker tracks and native Vilier manifest metadata.
"""
from __future__ import annotations

import json
from pathlib import Path


def run(source: Path, output: Path, config: dict):
    """Run the fixed Vilier profile and normalize its native manifest to two tracks."""
    from . import cli
    from core.orchestration.clearvoice import IsolatedClearVoice
    from core.orchestration.contract import vilier_tracks

    original_loader = cli.load_overlap_separator
    helpers = []

    def strict_loader(options, dry_run=False, warnings=None):
        if options.get("enabled") and options.get("backend") in {"clearvoice", "mossformer2"}:
            helper = IsolatedClearVoice(options.get("model_name", "alibabasglab/MossFormer2_SS_16K"), options.get("device", "auto"))
            helpers.append(helper)
            return helper
        separator = original_loader(options, dry_run=dry_run, warnings=warnings)
        if options.get("enabled") and separator is None:
            raise RuntimeError("Vilier separation failed to load: " + "; ".join(warnings or []))
        return separator

    cli.load_overlap_separator = strict_loader
    try:
        manifest, _ = cli.process_one(source, config, output / "native", output / "state", dry_run=False, until="pre_asr")
        native = json.loads(manifest.read_text())
        config_path = manifest.parent / "run_config.json"
        devices = json.loads(config_path.read_text()) if config_path.exists() else native.get("run_config")
        return vilier_tracks(manifest), {"native_manifest": str(manifest), "run_config": devices}
    finally:
        cli.load_overlap_separator = original_loader
        for helper in helpers:
            helper.close()
