from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "cholimex" / "src"))

from cholimex.speaker_assignment import assign_candidates  # noqa: E402


class IdentityEmbedding:
    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return wav.reshape(-1)


def test_overlap_assignment_records_direct_and_swapped_cosine_scores():
    extractor = IdentityEmbedding()
    first = torch.tensor([[1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0]])
    references = {0: first.reshape(-1), 1: second.reshape(-1)}

    assigned, record = assign_candidates(first, second, 16000, references, extractor, 0.5)

    assert torch.equal(assigned[0], first)
    assert record["mapping"] == {"candidate_0": 0, "candidate_1": 1}
    assert record["direct_score"] == pytest.approx(1.0)
    assert record["swapped_score"] == pytest.approx(0.0)


def test_overlap_assignment_swaps_only_when_embedding_evidence_supports_it():
    extractor = IdentityEmbedding()
    first = torch.tensor([[0.0, 1.0]])
    second = torch.tensor([[1.0, 0.0]])
    references = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}

    assigned, record = assign_candidates(first, second, 16000, references, extractor, 0.5)

    assert torch.equal(assigned[0], second)
    assert record["swapped"] is True
    assert record["mapping"] == {"candidate_0": 1, "candidate_1": 0}


def test_overlap_assignment_rejects_missing_reference_or_low_confidence():
    extractor = IdentityEmbedding()
    first = torch.tensor([[1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0]])

    with pytest.raises(RuntimeError, match="references for both speakers"):
        assign_candidates(first, second, 16000, {0: first.reshape(-1)}, extractor, 0.5)
    with pytest.raises(RuntimeError, match="below threshold"):
        assign_candidates(first, second, 16000, {0: -first.reshape(-1), 1: -second.reshape(-1)}, extractor, 0.5)
