# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from flashdreams.infra.acceleration import cuda_graph_dispatch

pytestmark = pytest.mark.ci_cpu


class FakeCUDAGraphWrapper:
    instances: list["FakeCUDAGraphWrapper"] = []

    def __init__(self, fn: Any, warmup_iters: int) -> None:
        self.fn = fn
        self.warmup_iters = warmup_iters
        self.calls: list[str] = []
        self.resets = 0
        self.instances.append(self)

    def drain(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("drain")
        return ("drain", self.fn(*args, **kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("graph")
        return ("graph", self.fn(*args, **kwargs))

    def reset(self) -> None:
        self.resets += 1


def test_cuda_graph_capture_ar_index_checks_chunk_alignment() -> None:
    assert (
        cuda_graph_dispatch.cuda_graph_capture_ar_index(
            sink_size_t=2,
            window_size_t=6,
            len_t=2,
        )
        == 4
    )

    with pytest.raises(AssertionError, match="whole number of AR chunks"):
        cuda_graph_dispatch.cuda_graph_capture_ar_index(
            sink_size_t=1,
            window_size_t=6,
            len_t=4,
        )


def test_dispatch_selects_drain_before_capture_and_wrapper_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeCUDAGraphWrapper.instances = []
    monkeypatch.setattr(
        cuda_graph_dispatch,
        "CUDAGraphWrapper",
        FakeCUDAGraphWrapper,
    )

    def fn(value: int, *, scale: int = 1) -> int:
        return value * scale

    dispatch = cuda_graph_dispatch.CUDAGraphDispatch(
        fn,
        enabled=True,
        capture_ar_idx=3,
        warmup_iters=5,
    )

    assert len(FakeCUDAGraphWrapper.instances) == 2
    cond_wrapper, uncond_wrapper = FakeCUDAGraphWrapper.instances
    assert cond_wrapper.warmup_iters == 5
    assert uncond_wrapper.warmup_iters == 5

    assert dispatch.select(2, uncond=False)(2, scale=10) == ("drain", 20)
    assert dispatch.select(3, uncond=False)(3, scale=10) == ("graph", 30)
    assert dispatch.select(4, uncond=True)(4, scale=10) == ("graph", 40)

    assert cond_wrapper.calls == ["drain", "graph"]
    assert uncond_wrapper.calls == ["graph"]


def test_dispatch_reset_and_rebuild_manage_wrapper_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeCUDAGraphWrapper.instances = []
    monkeypatch.setattr(
        cuda_graph_dispatch,
        "CUDAGraphWrapper",
        FakeCUDAGraphWrapper,
    )

    dispatch = cuda_graph_dispatch.CUDAGraphDispatch(
        lambda value: value,
        enabled=True,
        capture_ar_idx=1,
        warmup_iters=2,
    )
    first_cond, first_uncond = FakeCUDAGraphWrapper.instances

    dispatch.reset()

    assert first_cond.resets == 1
    assert first_uncond.resets == 1

    dispatch.rebuild(capture_ar_idx=7)

    assert dispatch.capture_ar_idx == 7
    assert len(FakeCUDAGraphWrapper.instances) == 4
    assert dispatch.cond_call is FakeCUDAGraphWrapper.instances[2]
    assert dispatch.uncond_call is FakeCUDAGraphWrapper.instances[3]


def test_disabled_dispatch_uses_eager_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeCUDAGraphWrapper.instances = []
    monkeypatch.setattr(
        cuda_graph_dispatch,
        "CUDAGraphWrapper",
        FakeCUDAGraphWrapper,
    )

    def fn(value: int) -> int:
        return value + 1

    dispatch = cuda_graph_dispatch.CUDAGraphDispatch(
        fn,
        enabled=False,
        capture_ar_idx=1,
        warmup_iters=2,
    )

    assert FakeCUDAGraphWrapper.instances == []
    assert dispatch.select(100, uncond=True)(4) == 5

    dispatch.reset()
    dispatch.rebuild(capture_ar_idx=3)

    assert dispatch.capture_ar_idx == 3
    assert dispatch.cond_call is None
    assert dispatch.uncond_call is None
    assert FakeCUDAGraphWrapper.instances == []


def test_disable_drops_wrappers_and_rebinds_eager_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeCUDAGraphWrapper.instances = []
    monkeypatch.setattr(
        cuda_graph_dispatch,
        "CUDAGraphWrapper",
        FakeCUDAGraphWrapper,
    )

    dispatch = cuda_graph_dispatch.CUDAGraphDispatch(
        lambda value: value + 1,
        enabled=True,
        capture_ar_idx=1,
        warmup_iters=2,
    )

    dispatch.disable(fn=lambda value: value + 10)

    assert dispatch.enabled is False
    assert dispatch.cond_call is None
    assert dispatch.uncond_call is None
    assert dispatch.select(100, uncond=False)(1) == 11
