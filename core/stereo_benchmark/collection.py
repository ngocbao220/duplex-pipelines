"""Reference-free benchmark orchestration for a DuplexChat conversation collection."""
from __future__ import annotations

import json
from pathlib import Path

from .report import summarize_reports
from .runner import run_benchmark


def run_collection_benchmark(manifest_path: Path, output_dir: Path, device: str = "auto") -> Path:
    """Benchmark every manifest conversation and write an inspectable collection summary."""
    manifest = json.loads(Path(manifest_path).read_text())
    rows, reports = [], []
    for conversation in manifest["conversations"]:
        stereo = Path(conversation["stereo"])
        report, report_path = run_benchmark(stereo, stereo.parent / "benchmark", device=device)
        reports.append(report)
        rows.append({"conversation_idx": conversation["conversation_idx"], "duration_sec": conversation["duration"],
                     "report": str(report_path), "metric_status": _metric_status(report)})
    summary = {
        "mode": "reference_free", "conversation_count": len(rows), "scored_conversations": len(rows),
        "conversations": rows,
        "aggregate": {"duration_sec": sum(row["duration_sec"] for row in rows), **summarize_reports(reports)},
    }
    target = Path(output_dir) / "benchmark.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _metric_status(report: dict) -> dict:
    return {
        "dnsmos": report["acoustic_quality"]["dnsmos"]["left"]["status"],
        "squim": report["acoustic_quality"]["squim"]["left"]["status"],
        "itc": report["speaker_identity"]["itc"]["mean"]["status"],
        "itd": report["speaker_identity"]["itd"]["status"],
    }
