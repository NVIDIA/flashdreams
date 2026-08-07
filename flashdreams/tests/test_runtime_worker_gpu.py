# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from flashdreams.runtime import ThreadAffineRuntimeWorker

pytestmark = pytest.mark.ci_gpu


@pytest.mark.asyncio
async def test_compiled_cuda_graph_replays_stay_on_runtime_thread() -> None:
    """Exercise repeated Triton launches and CUDA-graph replay on one worker."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")

    device = torch.device("cuda", torch.cuda.current_device())
    worker = ThreadAffineRuntimeWorker(device=device, thread_name="gpu-runtime-test")
    state: dict[str, object] = {}

    def _initialize() -> None:
        static_input = torch.ones(1024, device=device)
        compiled = torch.compile(lambda value: torch.sin(value) + 1.0)
        for _ in range(3):
            compiled(static_input)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = compiled(static_input)
        state.update(
            static_input=static_input,
            static_output=static_output,
            graph=graph,
        )

    def _step(value: float) -> float:
        static_input = state["static_input"]
        static_output = state["static_output"]
        graph = state["graph"]
        assert isinstance(static_input, torch.Tensor)
        assert isinstance(static_output, torch.Tensor)
        assert isinstance(graph, torch.cuda.CUDAGraph)
        static_input.fill_(value)
        graph.replay()
        torch.cuda.synchronize(device)
        return float(static_output[0].item())

    try:
        await worker.call(_initialize)
        values = [await worker.call(_step, float(index)) for index in range(8)]
    finally:
        await worker.close()

    expected = [
        float(torch.sin(torch.tensor(float(index))) + 1.0) for index in range(8)
    ]
    assert values == pytest.approx(expected)
