"""Purpose: Run the fixed Vilier profile through stereo normalization.

Inputs: Mixture source path, output directory and Vilier configuration.
Outputs: Contract-ready stereo audio and native Vilier manifest metadata.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from core.orchestration.logging_style import StepTimer, get_logger


def run(source: Path, output: Path, config: dict):
    """Run the fixed Vilier profile and normalize its native manifest to stereo."""
    from .separation import preflight_overlap_separator
    from . import cli
    from core.orchestration.clearvoice import IsolatedClearVoice
    import torchaudio
    from core.orchestration.contract import vilier_tracks
    from core.outputs import save_stereo_wav

    logger = get_logger("vilier")
    with StepTimer(logger, "Step 0: SepReformer preflight"):
        preflight_overlap_separator(config.get("overlap_separation", {}))

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
        manifest, _ = cli.process_one(source, config, output / "native", output / "state", dry_run=False, until="pre_asr",
                                      progress=cli.StepLogger(logger))
        native = json.loads(manifest.read_text())
        if config.get("debug"):
            _export_overlap_debug(manifest.parent, output, native)
        config_path = manifest.parent / "run_config.json"
        devices = json.loads(config_path.read_text()) if config_path.exists() else native.get("run_config")
        first_track, second_track = vilier_tracks(manifest)
        first_audio, first_rate = torchaudio.load(str(first_track))
        second_audio, second_rate = torchaudio.load(str(second_track))
        if first_rate != second_rate:
            raise ValueError("Vilier speaker tracks have mismatched sample rates")
        stereo = save_stereo_wav(output / "audio.stereo.wav", first_audio, second_audio, first_rate)
        return stereo, {"native_manifest": str(manifest), "run_config": devices}
    finally:
        cli.load_overlap_separator = original_loader
        for helper in helpers:
            helper.close()


def _export_overlap_debug(native_dir: Path, output: Path, manifest: dict) -> None:
    """Normalize native Vilier overlap clips to the shared debug-artifact contract."""
    records = list(manifest.get("overlap_separation", {}).get("overlap_regions", []))
    debug_root = output / "debug"
    exported = []
    for record in records:
        directory = debug_root / "overlaps" / record["id"]
        directory.mkdir(parents=True, exist_ok=True)
        source_paths = {"mixture.wav": record.get("mixed_audio", "")}
        separated = record.get("separated_audio", {})
        values = list(separated.values())
        if len(values) == 2:
            source_paths.update({f"source_{index:02d}.wav": value for index, value in enumerate(values, 1)})
        for name, relative in source_paths.items():
            if relative:
                source = native_dir / relative
                if source.is_file():
                    shutil.copy2(source, directory / name)
        (directory / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        exported.append(record)
    (debug_root / "overlaps.json").parent.mkdir(parents=True, exist_ok=True)
    (debug_root / "overlaps.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
