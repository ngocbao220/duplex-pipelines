from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat.separation_backend import _canonical_export_device, _retarget_exported_module  # noqa: E402


class _Node:
    def __init__(self) -> None:
        self.args = ("cuda", {"nested": torch.device("cuda")})
        self.kwargs = {"device": "cuda:1"}


class _Module:
    def __init__(self) -> None:
        self.graph = type("Graph", (), {"nodes": [_Node()]})()
        self.recompiled = False

    def recompile(self) -> None:
        self.recompiled = True


def test_export_device_retarget_rewrites_string_and_torch_device_literals():
    module = _Module()

    result = _retarget_exported_module(module, torch.device("cuda:0"))
    node = result.graph.nodes[0]

    assert result is module
    assert node.args == ("cuda:0", {"nested": torch.device("cuda:0")})
    assert node.kwargs == {"device": "cuda:0"}
    assert module.recompiled is True


def test_gpu_zero_uses_dialoguesidon_export_canonical_cuda_device():
    assert _canonical_export_device("cuda:0") == torch.device("cuda")
    assert _canonical_export_device("cuda:1") == torch.device("cuda:1")
