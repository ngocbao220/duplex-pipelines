"""Single-file bridge to the same isolated workers used by OtoSpeech."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

from .runner import ROOT, code_identity, launch_pipeline, pipeline_config
from .contract import write_json


def run_single(name: str, source: Path, output: Path, debug: bool, split_conversation: bool,
               diarization_backend: str, diarization_model: str | None,
               gt_a: Path | None, gt_b: Path | None) -> int:
    """Run one adapter without ever passing reference audio to its worker."""
    from core.config import load_config
    from core import benchmark

    if bool(gt_a) != bool(gt_b):
        raise ValueError("--gt-speaker-a and --gt-speaker-b must be provided together")
    cfg = load_config(ROOT / "configs/config.json")
    args = SimpleNamespace(debug=debug, split_conversation=split_conversation,
                           diarization_backend=diarization_backend,
                           diarization_model=diarization_model,
                           vilier_config=ROOT / "configs/vilier.json",
                           duplexchat_config=ROOT / "configs/duplexchat.json", sample_rate=16000)
    run_dir = output.parent / ".runs" / uuid.uuid4().hex
    result_path = run_dir / "results.json"
    request = {"pipeline": name, "config": pipeline_config(name, args, cfg),
               "code": code_identity(name), "force": False, "pred_root": str(output.parent),
               "results": str(result_path), "samples": [{"key": output.name, "mixture": str(source)}]}
    exit_code = launch_pipeline(name, request, run_dir)
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    row = rows[0] if rows else {"status": "failed", "error": "worker did not write result"}
    if row.get("status") != "complete":
        print(f"Failed: {row.get('error', f'worker exit={exit_code}')}\n"
              f"-> run.json: {output / 'run.json'}\n"
              f"-> worker log: {run_dir / name / 'worker.log'}", flush=True)
        return exit_code or 1
    report = output / "benchmark.json"
    if gt_a and gt_b:
        score = benchmark.score_reference_sample({"key": output.name, "gt_speaker_1": str(gt_a), "gt_speaker_2": str(gt_b)}, output.parent, 16000, -40.0, -20.0)
    else:
        score = benchmark._null_reference_row(output.name, "reference_unavailable", {"ground_truth": "not provided"})
    write_json(report, score)
    manifest = json.loads((output / "run.json").read_text())
    manifest["benchmark"] = {"mode": "reference" if gt_a else "reference_free", "report": str(report), "result": score}
    write_json(output / "run.json", manifest)
    print(f"Done\n-> audio.stereo.wav: {output / 'audio.stereo.wav'}\n-> benchmark.json: {report}\n-> run.json: {output / 'run.json'}", flush=True)
    return exit_code
