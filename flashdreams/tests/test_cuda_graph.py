# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import torch

from flashdreams.infra.cuda_graph import CUDAGraphWrapper

pytestmark = pytest.mark.ci_cpu


def test_failed_capture_does_not_store_invalid_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "capture_attempts": 0,
        "in_capture": False,
        "replays": 0,
    }

    class FakeGraph:
        def replay(self) -> None:
            state["replays"] += 1

    class FakeGraphContext:
        def __init__(self, graph: FakeGraph) -> None:
            self.graph = graph

        def __enter__(self) -> None:
            del self.graph
            state["capture_attempts"] += 1
            state["in_capture"] = True

        def __exit__(self, *exc_info: Any) -> bool:
            state["in_capture"] = False
            return False

    def fake_graph(graph: FakeGraph) -> FakeGraphContext:
        return FakeGraphContext(graph)

    def fn(x: torch.Tensor) -> torch.Tensor:
        if state["in_capture"] and state["capture_attempts"] == 1:
            raise RuntimeError("capture failed")
        return x + 1

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", fake_graph)

    wrapper = CUDAGraphWrapper(fn, warmup_iters=0)

    with pytest.raises(RuntimeError, match="capture failed"):
        wrapper(torch.tensor([1]))

    assert wrapper._graph is None
    assert state["replays"] == 0

    out = wrapper(torch.tensor([2]))

    assert out.item() == 3
    assert isinstance(wrapper._graph, FakeGraph)
    assert state["replays"] == 1
