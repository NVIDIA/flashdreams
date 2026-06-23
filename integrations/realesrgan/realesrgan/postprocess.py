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

"""FlashDreams post-processing wrapper for Real-ESRGAN."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from flashdreams.infra.postprocess import (
    VideoChunk,
    VideoPostProcessor,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoSpec,
    to_bvtchw,
    to_minus_one_one,
)
from realesrgan.upsampler import RealESRGANUpsampler, default_model_name


@dataclass(kw_only=True)
class RealESRGANPostProcessorConfig(VideoPostProcessorConfig):
    """Post-process generated RGB video with Real-ESRGAN."""

    _target: type["RealESRGANPostProcessor"] = field(
        default_factory=lambda: RealESRGANPostProcessor
    )

    scale: Literal[2, 4] = 2
    """Spatial upsample factor."""

    model_name: str | None = None
    """Real-ESRGAN checkpoint name. Defaults to the general model for scale."""

    model_path: str | Path | None = None
    """Optional local checkpoint path."""

    tile: int = 0
    """Input tile size. ``0`` processes each frame as a single tensor."""

    tile_pad: int = 10
    """Input tile overlap in pixels."""

    pre_pad: int = 10
    """Reflection padding applied before model inference."""

    half: bool = True
    """Use fp16 on CUDA."""

    compile_model: bool = False
    """Compile the Real-ESRGAN model with ``torch.compile``."""

    compile_mode: str = "reduce-overhead"
    """Mode passed to ``torch.compile`` when :attr:`compile_model` is enabled."""

    device: str = "cuda"
    """Torch device used by the Real-ESRGAN model."""


class RealESRGANPostProcessor(VideoPostProcessor[RealESRGANPostProcessorConfig]):
    """Factory for Real-ESRGAN post-processing sessions."""

    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        """Start one Real-ESRGAN post-processing stream."""
        return _RealESRGANPostProcessorSession(self.config)


class _RealESRGANPostProcessorSession(VideoPostProcessorSession):
    def __init__(self, config: RealESRGANPostProcessorConfig) -> None:
        self._config = config
        self._upsampler: RealESRGANUpsampler | None = None

    @torch.no_grad()
    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        canonical = to_bvtchw(
            to_minus_one_one(chunk.tensor, value_range=chunk.value_range),
            layout=chunk.layout,
        )
        batch, views, _, channels, _, _ = canonical.shape
        if channels != 3:
            raise ValueError(
                f"Real-ESRGAN expects RGB chunks; got {channels} channels."
            )
        upsampler = self._ensure_upsampler()
        outputs = []
        for batch_idx in range(batch):
            view_outputs = []
            for view_idx in range(views):
                view_outputs.append(
                    upsampler.upsample_video_tensor(canonical[batch_idx, view_idx])
                )
            outputs.append(torch.stack(view_outputs, dim=0))
        output = torch.stack(outputs, dim=0)
        return [
            VideoChunk(
                tensor=output,
                layout="bvtchw",
                value_range="minus_one_one",
                is_final=chunk.is_final,
                metadata={**chunk.metadata, "source": "realesrgan"},
            )
        ]

    def flush(self) -> list[VideoChunk]:
        """Real-ESRGAN is frame-local and keeps no buffered tail."""
        return []

    def _ensure_upsampler(self) -> RealESRGANUpsampler:
        if self._upsampler is None:
            model_name = self._config.model_name or default_model_name(
                self._config.scale
            )
            self._upsampler = RealESRGANUpsampler(
                scale=self._config.scale,
                model_name=model_name,
                model_path=self._config.model_path,
                tile=self._config.tile,
                tile_pad=self._config.tile_pad,
                pre_pad=self._config.pre_pad,
                half=self._config.half,
                compile_model=self._config.compile_model,
                compile_mode=self._config.compile_mode,
                device=self._config.device,
            )
        return self._upsampler
