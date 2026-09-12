"""Aggregate and render reference-free stereo benchmark reports."""
from __future__ import annotations


def summarize_reports(reports: list[dict]) -> dict:
    values = {
        "dnsmos_ovrl": [_at(report, "acoustic_quality", "dnsmos", "mean", "ovrl") for report in reports],
        "sq_stoi": [_at(report, "acoustic_quality", "squim", "mean", "sq_stoi") for report in reports],
        "sq_pesq": [_at(report, "acoustic_quality", "squim", "mean", "sq_pesq") for report in reports],
        "itc": [_at(report, "speaker_identity", "itc", "mean", "itc") for report in reports],
        "itd": [_at(report, "speaker_identity", "itd", "itd") for report in reports],
        "overlap_pct": [_at(report, "speech_activity", "overlap", "percentage") for report in reports],
        "turn_exchanges_per_min": [_at(report, "turn_taking", "turn_exchanges_per_min") for report in reports],
        "overlapping_transition_rate": [_at(report, "turn_taking", "overlapping_transition_rate") for report in reports],
        "backchannels_per_min": [_at(report, "turn_taking", "backchannels_per_min") for report in reports],
        "leakage_proxy_db": [
            _mean([
                _at(report, "leakage_proxy", "left_to_right", "median_db"),
                _at(report, "leakage_proxy", "right_to_left", "median_db"),
            ])
            for report in reports
        ],
    }
    metrics = {name: _mean(items) for name, items in values.items()}
    metrics["overlap_transition_pct"] = _scale(metrics.pop("overlapping_transition_rate"), 100)
    return {"sample_count": len(reports), "metrics": metrics}


def render_tables(summary: dict) -> str:
    """Render the requested compact Metric / Value / Meaning table."""
    metrics = summary["metrics"]
    rows = [
        ("DNSMOS", metrics["dnsmos_ovrl"], "Acoustic quality", ""),
        ("SQ-STOI", metrics["sq_stoi"], "Estimated intelligibility", ""),
        ("SQ-PESQ", metrics["sq_pesq"], "Estimated perceptual quality", ""),
        ("ITC", metrics["itc"], "Intra-track speaker consistency", ""),
        ("ITD", metrics["itd"], "Inter-track distinctiveness", ""),
        ("Overlap", metrics["overlap_pct"], "Simultaneous speech", " %"),
        ("Backchannel", metrics["backchannels_per_min"], "Estimated short responses (VAD-based)", " /min"),
        ("Turn Exchange", metrics["turn_exchanges_per_min"], "Speaker turn dynamics", " /min"),
        ("Overlap Transition", metrics["overlap_transition_pct"], "Turns initiated during overlap", " %"),
        ("Leakage Proxy", metrics["leakage_proxy_db"], "Inactive-channel energy (not SIR)", " dB"),
    ]
    formatted = [["Metric", "Value", "Meaning"]] + [[name, f"{_format(value)}{suffix if value is not None else ''}", meaning] for name, value, meaning, suffix in rows]
    return f"Samples scored: {summary['sample_count']}\n\n{_table(formatted)}"


def flatten_report(report: dict, source: str) -> dict:
    return {"source": source, "status": "ok", "duration_sec": _at(report, "input", "duration_sec"), **summarize_reports([report])["metrics"]}


def _at(data: dict, *keys: str):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return float(data) if isinstance(data, (int, float)) else None


def _mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _format(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _scale(value: float | None, multiplier: float) -> float | None:
    return None if value is None else value * multiplier


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    def line(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(rows[0]), separator, *(line(row) for row in rows[1:])])
