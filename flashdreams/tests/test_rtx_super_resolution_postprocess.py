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

"""CPU tests for the RTX Video Super Resolution post-processor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import flashdreams.infra.postprocess.rtx as rtx_module
from flashdreams.infra.postprocess import (
    RTXVideoSuperResolutionPostProcessorConfig,
    VideoChunk,
    VideoSpec,
)

pytestmark = pytest.mark.ci_cpu


def test_rtx_super_resolution_output_spec_uses_scale() -> None:
    config = RTXVideoSuperResolutionPostProcessorConfig(scale=2.0)

    spec = config.output_spec(VideoSpec(height=540, width=960, fps=24))

    assert spec == VideoSpec(height=1080, width=1920, fps=24, channels=3)


def test_rtx_super_resolution_output_spec_uses_explicit_dimensions() -> None:
    config = RTXVideoSuperResolutionPostProcessorConfig(
        output_height=720,
        output_width=1280,
        scale=4.0,
    )

    spec = config.output_spec(VideoSpec(height=360, width=640))

    assert spec == VideoSpec(height=720, width=1280, channels=3)


def test_rtx_super_resolution_rejects_partial_explicit_dimensions() -> None:
    config = RTXVideoSuperResolutionPostProcessorConfig(output_width=1280)

    with pytest.raises(ValueError, match="both output_width and output_height"):
        config.output_spec(VideoSpec(height=360, width=640))


def test_rtx_super_resolution_rejects_non_rgb_input() -> None:
    config = RTXVideoSuperResolutionPostProcessorConfig()

    with pytest.raises(ValueError, match="expects RGB"):
        config.output_spec(VideoSpec(height=8, width=8, channels=1))


@pytest.mark.parametrize("quality", ["DENOISE_HIGH", "DEBLUR_ULTRA"])
def test_rtx_super_resolution_rejects_scaled_same_resolution_quality(
    quality: str,
) -> None:
    config = RTXVideoSuperResolutionPostProcessorConfig(
        quality=quality,  # ty: ignore[invalid-argument-type]
        scale=2.0,
    )

    with pytest.raises(ValueError, match="same-resolution mode"):
        config.output_spec(VideoSpec(height=8, width=8))


def test_rtx_super_resolution_processes_frames_with_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeVideoSuperRes:
        class QualityLevel:
            HIGH = "HIGH"

        def __init__(self, *, quality: str, device: int) -> None:
            self.quality = quality
            self.device = device
            self.output_width: int | None = None
            self.output_height: int | None = None
            self.loaded = False
            self.closed = False
            self.calls: list[tuple[tuple[int, ...], bool, int]] = []
            created.append(self)

        def load(self) -> None:
            self.loaded = True

        def run(
            self,
            input_array: torch.Tensor,
            *,
            non_blocking: bool,
            stream_ptr: int,
        ) -> SimpleNamespace:
            assert self.loaded
            assert self.output_height is not None
            assert self.output_width is not None
            assert input_array.dtype == torch.float32
            assert input_array.min() >= 0
            assert input_array.max() <= 1
            self.calls.append((tuple(input_array.shape), non_blocking, stream_ptr))
            image = F.interpolate(
                input_array.unsqueeze(0),
                size=(self.output_height, self.output_width),
                mode="nearest",
            )[0]
            return SimpleNamespace(image=image)

        def close(self) -> None:
            self.closed = True

    created: list[_FakeVideoSuperRes] = []

    monkeypatch.setattr(
        rtx_module, "_load_video_super_res_class", lambda: _FakeVideoSuperRes
    )
    monkeypatch.setattr(
        rtx_module,
        "_torch_device_for_nvvfx_device",
        lambda device: torch.device("cpu"),
    )

    config = RTXVideoSuperResolutionPostProcessorConfig(scale=2.0)
    session = config.setup().start(VideoSpec(height=2, width=3, fps=12))
    video = torch.linspace(-1.0, 1.0, 2 * 3 * 2 * 3).reshape(2, 3, 2, 3)

    outputs = session.process(
        VideoChunk(
            tensor=video,
            layout="tchw",
            metadata={"autoregressive_index": 0},
        )
    )

    assert len(outputs) == 1
    assert outputs[0].layout == "bvtchw"
    assert outputs[0].tensor.shape == (1, 1, 2, 3, 4, 6)
    assert outputs[0].metadata["source"] == "rtx_video_super_resolution"
    assert len(created) == 1
    assert created[0].quality == "HIGH"
    assert created[0].device == 0
    assert created[0].output_height == 4
    assert created[0].output_width == 6
    assert created[0].calls == [((3, 2, 3), False, 0), ((3, 2, 3), False, 0)]
    expected_first = (
        F.interpolate(
            video[0].add(1.0).mul(0.5).unsqueeze(0),
            size=(4, 6),
            mode="nearest",
        )[0]
        .mul(2.0)
        .sub(1.0)
    )
    assert torch.equal(outputs[0].tensor[0, 0, 0], expected_first)

    assert session.flush() == []
    assert created[0].closed


def test_rtx_super_resolution_normalizes_uint8_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vfx_inputs: list[torch.Tensor] = []

    class _FakeVideoSuperRes:
        class QualityLevel:
            HIGH = "HIGH"

        def __init__(self, **_kwargs: object) -> None:
            self.output_width = 0
            self.output_height = 0

        def load(self) -> None:
            pass

        def run(self, input_array: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
            vfx_inputs.append(input_array.clone())
            image = F.interpolate(
                input_array.unsqueeze(0),
                size=(self.output_height, self.output_width),
                mode="nearest",
            )[0]
            return SimpleNamespace(image=image)

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        rtx_module, "_load_video_super_res_class", lambda: _FakeVideoSuperRes
    )
    monkeypatch.setattr(
        rtx_module,
        "_torch_device_for_nvvfx_device",
        lambda device: torch.device("cpu"),
    )
    video = torch.tensor(
        [[[[[[0, 64], [128, 255]], [[16, 80], [144, 240]], [[32, 96], [160, 224]]]]]],
        dtype=torch.uint8,
    )
    config = RTXVideoSuperResolutionPostProcessorConfig(scale=2.0)
    session = config.setup().start(VideoSpec(height=2, width=2))

    outputs = session.process(VideoChunk(tensor=video, layout="bvtchw"))

    expected_input = video[0, 0, 0].float().div(255.0)
    assert len(vfx_inputs) == 1
    assert torch.allclose(vfx_inputs[0], expected_input)
    expected_output = (
        F.interpolate(expected_input.unsqueeze(0), size=(4, 4), mode="nearest")[0]
        .mul(2.0)
        .sub(1.0)
    )
    assert torch.allclose(outputs[0].tensor[0, 0, 0], expected_output)


def test_rtx_super_resolution_synchronizes_nonblocking_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeVideoSuperRes:
        class QualityLevel:
            HIGH = "HIGH"

        def __init__(self, **_kwargs: object) -> None:
            self.output_width = 0
            self.output_height = 0

        def load(self) -> None:
            pass

        def run(self, input_array: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(image=input_array)

        def close(self) -> None:
            pass

    synchronizations: list[tuple[torch.device, bool]] = []
    monkeypatch.setattr(
        rtx_module, "_load_video_super_res_class", lambda: _FakeVideoSuperRes
    )
    monkeypatch.setattr(
        rtx_module,
        "_torch_device_for_nvvfx_device",
        lambda device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        rtx_module,
        "_synchronize_nonblocking_output",
        lambda *, device, enabled: synchronizations.append((device, enabled)),
    )
    config = RTXVideoSuperResolutionPostProcessorConfig(
        scale=1.0,
        non_blocking=True,
    )
    session = config.setup().start(VideoSpec(height=2, width=3))

    session.process(VideoChunk(tensor=torch.zeros(1, 3, 2, 3)))

    assert synchronizations == [(torch.device("cpu"), True)]


def test_rtx_super_resolution_empty_chunk_does_not_load_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_load() -> object:
        raise AssertionError("backend should not load for empty chunks")

    monkeypatch.setattr(rtx_module, "_load_video_super_res_class", _fail_load)
    config = RTXVideoSuperResolutionPostProcessorConfig(scale=2.0)
    session = config.setup().start(VideoSpec(height=2, width=3))

    outputs = session.process(VideoChunk(tensor=torch.empty(0, 3, 2, 3)))

    assert len(outputs) == 1
    assert outputs[0].tensor.shape == (1, 1, 0, 3, 4, 6)


def test_rtx_super_resolution_loader_uses_optional_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeVfx:
        VideoSuperRes = object()

    imported: list[str] = []

    def _fake_import_module(name: str) -> object:
        imported.append(name)
        return _FakeVfx

    monkeypatch.setattr(rtx_module, "import_module", _fake_import_module)

    assert rtx_module._load_video_super_res_class() is _FakeVfx.VideoSuperRes
    assert imported == ["nvvfx"]


def test_rtx_super_resolution_loader_missing_vfx_symbol_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rtx_module, "import_module", lambda name: object())

    with pytest.raises(RuntimeError, match="nvvfx.VideoSuperRes"):
        rtx_module._load_video_super_res_class()


def test_rtx_super_resolution_missing_backend_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing_backend() -> object:
        raise RuntimeError("install nvidia-vfx")

    monkeypatch.setattr(rtx_module, "_load_video_super_res_class", _missing_backend)
    config = RTXVideoSuperResolutionPostProcessorConfig(scale=2.0)
    session = config.setup().start(VideoSpec(height=2, width=3))

    with pytest.raises(RuntimeError, match="nvidia-vfx"):
        session.process(VideoChunk(tensor=torch.zeros(1, 3, 2, 3)))
