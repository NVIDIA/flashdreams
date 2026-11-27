# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate ownership handoffs for the native CUDA upload buffer."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from flashdreams.runtime_v2.native_window_client_window import _PresentationContext

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("has_previous_submission", [False, True])
def test_native_upload_orders_graphics_read_after_cuda_write(
    monkeypatch: pytest.MonkeyPatch, has_previous_submission: bool
) -> None:
    rgba = torch.zeros((2, 2, 4), dtype=torch.uint8)
    frame = torch.full((2, 2, 3), 42, dtype=torch.uint8)
    handoffs: list[str] = []

    def graphics_to_cuda(stream: int) -> None:
        assert stream == 7
        assert not rgba.any(), "The previous graphics read must finish before reuse."
        handoffs.append("graphics_to_cuda")

    def cuda_to_graphics(stream: int) -> None:
        assert stream == 7
        assert torch.equal(rgba[..., :3], frame)
        assert (rgba[..., 3] == 255).all()
        handoffs.append("cuda_to_graphics")

    def copy_to_texture(*_args: object) -> None:
        assert handoffs[-1] == "cuda_to_graphics"
        handoffs.append("copy_to_texture")

    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=7)
    )
    context = object.__new__(_PresentationContext)
    context._cuda_rgba = rgba
    context._cuda_buffer = object()
    context._render_device = torch.device("cuda")
    context._display_texture = object()
    context._width = context._height = 2
    context._has_cuda_submission = has_previous_submission
    context._device = SimpleNamespace(
        sync_to_device=graphics_to_cuda, sync_to_cuda=cuda_to_graphics
    )
    context._copy_cuda_frame(
        frame, SimpleNamespace(copy_buffer_to_texture=copy_to_texture)
    )
    expected = ["cuda_to_graphics", "copy_to_texture"]
    if has_previous_submission:
        expected.insert(0, "graphics_to_cuda")
    assert handoffs == expected
