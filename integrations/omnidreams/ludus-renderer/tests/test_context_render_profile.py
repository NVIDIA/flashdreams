# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch


@pytest.mark.ci_cpu
def test_context_render_records_internal_profile(monkeypatch) -> None:
    from ludus_renderer._ops import context

    class _FakePlugin:
        def __init__(self) -> None:
            self.calls = 0

        def ludus_render_fwd_cuda_timestamped(self, *args, **kwargs) -> torch.Tensor:
            del args, kwargs
            self.calls += 1
            return torch.full((1, 2, 3, 4), self.calls, dtype=torch.uint8)

    fake_plugin = _FakePlugin()
    monkeypatch.setattr(context, "_get_plugin", lambda: fake_plugin)

    ctx = context.LudusCudaTimestampedContext.__new__(
        context.LudusCudaTimestampedContext
    )
    ctx._camera_intrinsics = torch.zeros((1, 4), dtype=torch.float32)
    ctx._max_extrapolation_us = 500_000
    ctx._tessellation_threshold = 1.0
    ctx.cpp_wrapper = object()
    ctx.enable_render_profiling = True
    ctx.enable_render_profile_cuda_events = False
    empty = torch.empty(0)
    ctx._scenes = [
        {
            "timestamps": empty,
            "int32": empty,
            "vertices": empty,
            "triangles": empty,
            "floats": empty,
            "polyline_pools": empty,
            "polygon_pools": empty,
            "cube_pools": empty,
            "max_varrays_per_ts_polyline": 0,
            "max_varrays_per_ts_polygon": 0,
        }
    ]

    image = ctx.render(
        scene_ids=torch.zeros(2, dtype=torch.int32),
        camera_ids=torch.zeros(2, dtype=torch.int32),
        timestamps_us=torch.tensor([1_000_000, 1_033_333], dtype=torch.int64),
        camera_type_ids=torch.zeros(2, dtype=torch.int32),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        resolution=(2, 3),
    )

    assert image.shape == (2, 2, 3, 4)
    assert fake_plugin.calls == 2
    assert ctx.last_render_profile["ctx_render_batch_size"] == 2.0
    assert ctx.last_render_profile["ctx_render_scalar_item_host_ms_count"] == 2.0
    assert ctx.last_render_profile["ctx_render_query_prep_host_ms_count"] == 2.0
    assert ctx.last_render_profile["ctx_render_plugin_host_ms_count"] == 2.0
    assert "ctx_render_cat_host_ms" in ctx.last_render_profile
    assert "ctx_render_total_host_ms" in ctx.last_render_profile
