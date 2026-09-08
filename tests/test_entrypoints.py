from __future__ import annotations

import cli
from pathlib import Path
from core.orchestration.cli import build_pipeline_parser


def test_each_pipeline_cli_exposes_single_and_otospeech_commands():
    for name in ("vilier", "duplexchat", "cholimex"):
        parser = build_pipeline_parser(name)
        single_args = parser.parse_args(["single", "--input", "mixture.wav", "--output-dir", "out"])
        otospeech_args = parser.parse_args(["otospeech", "--max-samples", "1"])
        assert single_args.command == "single"
        assert otospeech_args.command == "otospeech"


def test_root_cli_exposes_one_explicit_comparison_command():
    args = cli.build_parser().parse_args(["compare-otospeech", "--max-samples", "1"])
    assert args.command == "compare-otospeech"


def test_each_pipeline_owns_named_source_modules_with_contract_headers():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "vilier": {"preprocess.py", "vad.py", "diarization.py", "separation.py", "reconstruct.py", "runner.py"},
        "duplexchat": {"preprocess.py", "diarization.py", "dialogues.py", "separation.py", "reconstruct.py", "runner.py"},
        "cholimex": {"preprocess.py", "vad.py", "regions.py", "speaker_assignment.py", "separation.py", "reconstruct.py", "runner.py"},
    }
    for name, modules in expected.items():
        phase_dir = root / "pipeline" / name / "src" / name
        assert modules <= {path.name for path in phase_dir.glob("*.py")}
        for source in phase_dir.glob("*.py"):
            docstring = source.read_text(encoding="utf-8").lstrip()
            assert docstring.startswith('"""')
            assert all(label in docstring.split('"""', 2)[1] for label in ("Purpose:", "Inputs:", "Outputs:"))
