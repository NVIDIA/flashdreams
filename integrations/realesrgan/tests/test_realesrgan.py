# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Real-ESRGAN upsampler integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import realesrgan.postprocess as postprocess_mod
from realesrgan.postprocess import RealESRGANPostProcessorConfig
from realesrgan.upsampler import (
    RealESRGANFrameProfile,
    RealESRGANUpsampler,
    create_model,
)

from flashdreams.infra.postprocess import VideoChunk

pytestmark = pytest.mark.ci_cpu


def test_create_x2_model_matches_expected_shape() -> None:
    model, scale = create_model("RealESRGAN_x2plus")
    model.eval()
    x = torch.zeros(1, 3, 8, 8)

    with torch.no_grad():
        y = model(x)

    assert scale == 2
    assert y.shape == (1, 3, 16, 16)


def test_random_weight_upsampler_scales_tensor_without_checkpoint() -> None:
    upsampler = RealESRGANUpsampler(
        scale=2,
        model_name="RealESRGAN_x2plus",
        pre_pad=0,
        half=False,
        device="cpu",
        load_checkpoint=False,
    )
    frame = torch.zeros(3, 8, 8)

    output = upsampler.upsample_frame_tensor(frame)

    assert output.shape == (3, 16, 16)
    assert output.dtype == torch.float32


def test_random_weight_upsampler_profiled_tensor_reports_cpu_profile() -> None:
    upsampler = RealESRGANUpsampler(
        scale=2,
        model_name="RealESRGAN_x2plus",
        pre_pad=0,
        half=False,
        device="cpu",
        load_checkpoint=False,
    )
    frame = torch.zeros(3, 8, 8)

    output, profile = upsampler.upsample_frame_tensor_profiled(frame)

    assert output.shape == (3, 16, 16)
    assert profile == RealESRGANFrameProfile(model_ms=None)


def test_upsampler_can_compile_model(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls = []

    def fake_compile(model: torch.nn.Module, *, mode: str) -> torch.nn.Module:
        compile_calls.append(mode)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)

    RealESRGANUpsampler(
        scale=2,
        model_name="RealESRGAN_x2plus",
        pre_pad=0,
        half=False,
        compile_model=True,
        compile_mode="reduce-overhead",
        device="cpu",
        load_checkpoint=False,
    )

    assert compile_calls == ["reduce-overhead"]


def test_bgr_image_output_uses_opencv_channel_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsampler = RealESRGANUpsampler(
        scale=2,
        model_name="RealESRGAN_x2plus",
        pre_pad=0,
        half=False,
        device="cpu",
        load_checkpoint=False,
    )

    def fake_upsample_frame_tensor_profiled(
        frame: torch.Tensor,
    ) -> tuple[torch.Tensor, RealESRGANFrameProfile]:
        rgb = torch.empty(3, 2, 2)
        rgb[0].fill_(0.2)
        rgb[1].fill_(0.4)
        rgb[2].fill_(0.6)
        return rgb * 2.0 - 1.0, RealESRGANFrameProfile(model_ms=None)

    monkeypatch.setattr(
        upsampler,
        "upsample_frame_tensor_profiled",
        fake_upsample_frame_tensor_profiled,
    )

    image = np.zeros((1, 1, 3), dtype=np.uint8)
    output, mode = upsampler.upsample_bgr_image(image)

    assert mode == "RGB"
    assert output.dtype == np.uint8
    assert output.shape == (2, 2, 3)
    assert output[0, 0].tolist() == [153, 102, 51]


def test_bgra_image_output_preserves_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    upsampler = RealESRGANUpsampler(
        scale=2,
        model_name="RealESRGAN_x2plus",
        pre_pad=0,
        half=False,
        device="cpu",
        load_checkpoint=False,
    )

    def fake_upsample_frame_tensor_profiled(
        frame: torch.Tensor,
    ) -> tuple[torch.Tensor, RealESRGANFrameProfile]:
        return torch.zeros(3, 2, 2), RealESRGANFrameProfile(model_ms=None)

    monkeypatch.setattr(
        upsampler,
        "upsample_frame_tensor_profiled",
        fake_upsample_frame_tensor_profiled,
    )

    image = np.array([[[10, 20, 30, 77]]], dtype=np.uint8)
    output, mode = upsampler.upsample_bgr_image(image)

    assert mode == "RGBA"
    assert output.dtype == np.uint8
    assert output.shape == (2, 2, 4)
    assert np.all(output[:, :, 3] == 77)


@dataclass
class _FakeUpsampler:
    scale: int
    model_name: str | None = None
    model_path: str | None = None
    tile: int = 0
    tile_pad: int = 10
    pre_pad: int = 10
    half: bool = True
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    device: str = "cuda"

    def upsample_video_tensor(self, video: torch.Tensor) -> torch.Tensor:
        return video.repeat_interleave(self.scale, dim=-2).repeat_interleave(
            self.scale,
            dim=-1,
        )


def test_postprocess_scales_batched_views(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(postprocess_mod, "RealESRGANUpsampler", _FakeUpsampler)
    session = (
        RealESRGANPostProcessorConfig(scale=2)
        .setup()
        .start(postprocess_mod.VideoSpec(height=2, width=3, channels=3))
    )
    chunk = VideoChunk(
        tensor=torch.zeros(1, 2, 4, 3, 2, 3),
        layout="bvtchw",
    )

    outputs = session.process(chunk)

    assert len(outputs) == 1
    assert outputs[0].layout == "bvtchw"
    assert outputs[0].metadata["source"] == "realesrgan"
    assert outputs[0].tensor.shape == (1, 2, 4, 3, 4, 6)


def test_postprocess_reports_scaled_output_spec() -> None:
    config = RealESRGANPostProcessorConfig(scale=4)

    spec = config.output_spec(postprocess_mod.VideoSpec(height=8, width=12, fps=24))

    assert spec == postprocess_mod.VideoSpec(height=32, width=48, fps=24)


def test_realesrgan_postprocess_preset_is_registered() -> None:
    from flashdreams.plugins.registry import resolve_postprocess_preset

    processor = resolve_postprocess_preset("realesrgan")

    assert isinstance(processor, RealESRGANPostProcessorConfig)
    assert processor.scale == 2
    assert processor.device == "cuda"
    assert processor.half is True
