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

"""NVIDIA RTX Video Super Resolution post-processor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import torch
from torch import Tensor

from flashdreams.infra.postprocess.base import (
    VideoChunk,
    VideoPostProcessor,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoSpec,
    to_bvtchw,
)

RTXVideoSuperResolutionQuality = Literal[
    "BICUBIC",
    "LOW",
    "MEDIUM",
    "HIGH",
    "ULTRA",
    "DENOISE_LOW",
    "DENOISE_MEDIUM",
    "DENOISE_HIGH",
    "DENOISE_ULTRA",
    "DEBLUR_LOW",
    "DEBLUR_MEDIUM",
    "DEBLUR_HIGH",
    "DEBLUR_ULTRA",
    "HIGHBITRATE_LOW",
    "HIGHBITRATE_MEDIUM",
    "HIGHBITRATE_HIGH",
    "HIGHBITRATE_ULTRA",
]
"""Quality modes exposed by ``nvvfx.VideoSuperRes.QualityLevel``."""

_QUALITY_NAMES = get_args(RTXVideoSuperResolutionQuality)
_SAME_RESOLUTION_QUALITIES = {
    "DENOISE_LOW",
    "DENOISE_MEDIUM",
    "DENOISE_HIGH",
    "DENOISE_ULTRA",
    "DEBLUR_LOW",
    "DEBLUR_MEDIUM",
    "DEBLUR_HIGH",
    "DEBLUR_ULTRA",
}


@dataclass(kw_only=True)
class RTXVideoSuperResolutionPostProcessorConfig(VideoPostProcessorConfig):
    """Post-process RGB video frames with NVIDIA RTX Video Super Resolution.

    The runtime is provided by the optional ``nvidia-vfx`` Python package, which
    exposes ``nvvfx.VideoSuperRes``. The processor preserves FlashDreams'
    ``[-1, 1]`` tensor range at its boundary and converts each frame to the
    ``[0, 1]`` float32 CUDA input expected by the VFX binding.
    """

    _target: type["RTXVideoSuperResolutionPostProcessor"] = field(
        default_factory=lambda: RTXVideoSuperResolutionPostProcessor
    )

    scale: float = 2.0
    """Spatial upsample factor used when explicit output dimensions are unset."""

    output_width: int | None = None
    """Optional explicit output width in pixels."""

    output_height: int | None = None
    """Optional explicit output height in pixels."""

    quality: RTXVideoSuperResolutionQuality = "HIGH"
    """RTX Video Super Resolution quality mode."""

    device: int = 0
    """CUDA device index passed to ``nvvfx.VideoSuperRes``."""

    clamp_input: bool = True
    """Clamp incoming FlashDreams frames to ``[-1, 1]`` before VFX conversion."""

    non_blocking: bool = False
    """Request asynchronous VFX execution before synchronizing at the boundary."""

    use_current_stream: bool = True
    """Pass the current PyTorch CUDA stream pointer to ``VideoSuperRes.run``."""

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        """Return the RGB stream specification produced by RTX VSR."""
        if input_spec.channels != 3:
            raise ValueError(
                f"RTX Video Super Resolution expects RGB chunks with 3 channels; "
                f"got {input_spec.channels}."
            )
        output_height, output_width = self._output_dimensions(input_spec)
        return VideoSpec(
            height=output_height,
            width=output_width,
            fps=input_spec.fps,
            channels=3,
        )

    def _output_dimensions(self, input_spec: VideoSpec) -> tuple[int, int]:
        if (self.output_width is None) != (self.output_height is None):
            raise ValueError(
                "RTX Video Super Resolution requires both output_width and "
                "output_height when explicit output dimensions are configured."
            )
        if self.output_width is not None and self.output_height is not None:
            output_height = self.output_height
            output_width = self.output_width
        else:
            if self.scale <= 0:
                raise ValueError(
                    "RTX Video Super Resolution scale must be positive; "
                    f"got {self.scale}."
                )
            output_height = int(round(input_spec.height * self.scale))
            output_width = int(round(input_spec.width * self.scale))
        if output_height <= 0 or output_width <= 0:
            raise ValueError(
                "RTX Video Super Resolution output dimensions must be positive; "
                f"got {output_height}x{output_width}."
            )
        if self.quality in _SAME_RESOLUTION_QUALITIES and (
            output_height,
            output_width,
        ) != (input_spec.height, input_spec.width):
            raise ValueError(
                f"RTX Video Super Resolution quality {self.quality!r} is a "
                "same-resolution mode; set scale=1.0 or explicit dimensions "
                f"{input_spec.height}x{input_spec.width}."
            )
        return output_height, output_width


class RTXVideoSuperResolutionPostProcessor(
    VideoPostProcessor[RTXVideoSuperResolutionPostProcessorConfig]
):
    """Factory for RTX Video Super Resolution sessions."""

    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        """Start a lazy RTX VSR session for one generated stream."""
        return _RTXVideoSuperResolutionPostProcessorSession(self.config, spec)


class _RTXVideoSuperResolutionPostProcessorSession(VideoPostProcessorSession):
    """Stateful RTX VSR processor that applies the VFX effect per frame."""

    def __init__(
        self, config: RTXVideoSuperResolutionPostProcessorConfig, spec: VideoSpec
    ) -> None:
        self._config = config
        self._input_spec = spec
        self._output_spec = config.output_spec(spec)
        self._effect: Any | None = None
        self._closed = False

    @torch.no_grad()
    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Upsample every frame in ``chunk`` and emit one output chunk."""
        if self._closed:
            raise RuntimeError(
                "cannot process RTX Video Super Resolution after flush()"
            )
        canonical = self._chunk_to_bvtchw(chunk)
        if canonical.shape[2] == 0:
            return [
                VideoChunk(
                    tensor=_empty_output_like(canonical, spec=self._output_spec),
                    layout="bvtchw",
                    metadata={
                        "source": "rtx_video_super_resolution",
                        "input_chunk": dict(chunk.metadata),
                    },
                )
            ]
        effect = self._ensure_effect()
        output = self._run_vsr(canonical, effect)
        metadata = {
            "source": "rtx_video_super_resolution",
            "input_chunk": dict(chunk.metadata),
        }
        return [VideoChunk(tensor=output, layout="bvtchw", metadata=metadata)]

    def flush(self) -> list[VideoChunk]:
        """Close the VFX effect and emit no tail frames."""
        if not self._closed:
            self._closed = True
            if self._effect is not None:
                self._effect.close()
                self._effect = None
        return []

    def _chunk_to_bvtchw(self, chunk: VideoChunk) -> Tensor:
        canonical = to_bvtchw(chunk.tensor, layout=chunk.layout)
        _, _, _, channels, height, width = canonical.shape
        if channels != 3:
            raise ValueError(
                f"RTX Video Super Resolution expects RGB chunks with 3 channels; "
                f"got {channels}."
            )
        if height != self._input_spec.height or width != self._input_spec.width:
            raise ValueError(
                "RTX Video Super Resolution stream dimensions changed from "
                f"{self._input_spec.height}x{self._input_spec.width} to "
                f"{height}x{width}."
            )
        return canonical

    def _ensure_effect(self) -> Any:
        if self._effect is not None:
            return self._effect

        video_super_res = _load_video_super_res_class()
        effect = video_super_res(
            quality=_resolve_quality(video_super_res, self._config.quality),
            device=self._config.device,
        )
        effect.output_width = self._output_spec.width
        effect.output_height = self._output_spec.height
        effect.load()
        self._effect = effect
        return effect

    def _run_vsr(self, canonical: Tensor, effect: Any) -> Tensor:
        batch, views, frames, _, _, _ = canonical.shape
        device = _torch_device_for_nvvfx_device(self._config.device)
        stream_ptr = _current_cuda_stream_ptr(
            device=device,
            enabled=self._config.use_current_stream,
        )
        outputs: list[Tensor] = []
        for batch_idx in range(batch):
            for view_idx in range(views):
                for frame_idx in range(frames):
                    outputs.append(
                        self._run_frame(
                            canonical[batch_idx, view_idx, frame_idx],
                            effect=effect,
                            device=device,
                            stream_ptr=stream_ptr,
                        )
                    )
        stacked = torch.stack(outputs, dim=0)
        return stacked.reshape(
            batch,
            views,
            frames,
            3,
            self._output_spec.height,
            self._output_spec.width,
        )

    def _run_frame(
        self,
        frame: Tensor,
        *,
        effect: Any,
        device: torch.device,
        stream_ptr: int,
    ) -> Tensor:
        # The VFX binding expects contiguous channels-first float32 CUDA frames
        # in [0, 1]. Most FlashDreams runners emit float RGB in [-1, 1], while
        # serving integrations such as Omnidreams emit display-ready uint8 RGB.
        if frame.dtype == torch.uint8:
            frame = frame.to(device=device, dtype=torch.float32).mul(1.0 / 255.0)
        else:
            frame = frame.to(device=device, dtype=torch.float32)
            if self._config.clamp_input:
                frame = frame.clamp(-1.0, 1.0)
            frame = frame.add(1.0).mul(0.5)
        frame = frame.contiguous()
        result = effect.run(
            frame,
            non_blocking=self._config.non_blocking,
            stream_ptr=stream_ptr,
        )
        _synchronize_nonblocking_output(
            device=device,
            enabled=self._config.non_blocking,
        )
        output = torch.from_dlpack(result.image).clone()
        return output.mul(2.0).sub(1.0)


def _empty_output_like(canonical: Tensor, *, spec: VideoSpec) -> Tensor:
    batch, views, _, _, _, _ = canonical.shape
    return canonical.new_empty(
        (batch, views, 0, 3, spec.height, spec.width),
        dtype=torch.float32,
    )


def _load_video_super_res_class() -> Any:
    try:
        from nvvfx import (  # ty: ignore[unresolved-import]  # noqa: PLC0415
            VideoSuperRes,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RTX Video Super Resolution post-processing requires the optional "
            "`nvidia-vfx` package, which provides `nvvfx.VideoSuperRes`. "
            "Install it in the active environment before selecting the "
            "`rtx-super-resolution` postprocess preset."
        ) from exc
    return VideoSuperRes


def _resolve_quality(video_super_res: Any, quality: str) -> Any:
    try:
        return getattr(video_super_res.QualityLevel, quality)
    except AttributeError as exc:
        available = ", ".join(_QUALITY_NAMES)
        raise ValueError(
            f"Unsupported RTX Video Super Resolution quality {quality!r}. "
            f"Supported values: {available}."
        ) from exc


def _torch_device_for_nvvfx_device(device: int) -> torch.device:
    return torch.device(f"cuda:{device}")


def _current_cuda_stream_ptr(*, device: torch.device, enabled: bool) -> int:
    if not enabled or device.type != "cuda":
        return 0
    return torch.cuda.current_stream(device=device).cuda_stream


def _synchronize_nonblocking_output(*, device: torch.device, enabled: bool) -> None:
    """Wait until an asynchronous VFX write is safe to import through DLPack."""
    if enabled:
        torch.cuda.synchronize(device=device)


POSTPROCESS_PRESET_RTX_SUPER_RESOLUTION = RTXVideoSuperResolutionPostProcessorConfig()
"""Two-times RTX Video Super Resolution with ``HIGH`` quality."""

POSTPROCESS_PRESET_RTX_SUPER_RESOLUTION_ULTRA = (
    RTXVideoSuperResolutionPostProcessorConfig(quality="ULTRA")
)
"""Two-times RTX Video Super Resolution with ``ULTRA`` quality."""

POSTPROCESS_PRESET_RTX_DEBLUR_ULTRA = RTXVideoSuperResolutionPostProcessorConfig(
    scale=1.0,
    quality="DEBLUR_ULTRA",
)
"""Same-resolution RTX Video Deblur with ``ULTRA`` quality."""
