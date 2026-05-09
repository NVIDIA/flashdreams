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

"""I2V control encoder with Plücker camera control for Lingbot World."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from einops import rearrange
from torch import Tensor

from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoderCache,
    StreamingVideoEncoder,
)
from flashdreams.recipes.alpadreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderCache,
    PixelShuffleVAEEncoderConfig,
)
from flashdreams.recipes.wan.autoencoder.i2v import (
    I2VCtrl,
    I2VCtrlEncoderCache,
    I2VCtrlEncoderConfig,
)

from .utils import (
    compute_relative_poses_causal,
    get_plucker_embeddings,
)


@dataclass(kw_only=True)
class CamCtrlInput:
    """Per-AR-step camera payload."""

    intrinsics: Tensor
    """Per-frame camera intrinsics of shape ``[..., T, 4]`` (fx, fy, cx, cy)."""

    poses: Tensor
    """Per-frame camera-to-world poses of shape ``[..., T, 4, 4]``."""

    world_scale: float
    """Scalar applied to translations when normalizing world coordinates."""


@dataclass(kw_only=True)
class I2VCamCtrlInput:
    """Composite per-AR-step input: image chunk + camera payload."""

    i2v: Tensor | None = None
    """Per-AR-step image chunk to encode through the I2V branch; ``None`` when omitted."""

    camctrl: CamCtrlInput
    """Per-AR-step camera intrinsics, poses, and world scale."""


@dataclass(kw_only=True)
class I2VCamCtrlEmbeddings:
    """Encoded I2V latent + Plücker volume the transformer cross-attends to."""

    i2v: I2VCtrl
    """Output of the Wan-VAE I2V encoder branch."""

    plucker: Tensor
    """Plücker pixel volume after the PixelShuffle encoder."""

    _is_patchified: bool = False
    """``True`` once the consuming transformer has patchified this payload in place."""


@dataclass(kw_only=True)
class I2VCamCtrlEncoderConfig(EncoderConfig):
    """Config for the composite I2V + Plücker encoder."""

    _target: type = field(default_factory=lambda: I2VCamCtrlEncoder)

    i2v: I2VCtrlEncoderConfig = field(default_factory=I2VCtrlEncoderConfig)
    """Config for the Wan-VAE I2V encoder branch."""

    plucker: PixelShuffleVAEEncoderConfig = field(
        default_factory=PixelShuffleVAEEncoderConfig
    )
    """Config for the PixelShuffle pseudo-VAE encoder applied to the Plücker volume."""


@dataclass(kw_only=True)
class I2VCamCtrlEncoderCache(StreamingEncoderCache):
    """Per-AR-step cache for the composite I2V + camera-control encoder."""

    i2v: I2VCtrlEncoderCache
    """Per-rollout cache for the I2V encoder branch."""

    plucker: PixelShuffleVAEEncoderCache
    """Per-rollout cache for the Plücker PixelShuffle encoder branch."""

    camera_last_pose: Tensor | None = None
    """Last-pose anchor used to make ``compute_relative_poses_causal``
    deterministic across AR steps; ``None`` at AR step 0."""


class I2VCamCtrlEncoder(StreamingVideoEncoder[I2VCamCtrlEncoderCache]):
    """Pairs a Wan-VAE I2V encoder with a PixelShuffle Plücker encoder."""

    def __init__(self, config: I2VCamCtrlEncoderConfig) -> None:
        super().__init__(config)
        self.i2v_encoder = config.i2v.setup()
        self.plucker_encoder = config.plucker.setup()

    def initialize_autoregressive_cache(self) -> I2VCamCtrlEncoderCache:
        return I2VCamCtrlEncoderCache(
            i2v=self.i2v_encoder.initialize_autoregressive_cache(),
            plucker=self.plucker_encoder.initialize_autoregressive_cache(),
        )

    @torch.no_grad()
    def forward(
        self,
        input: I2VCamCtrlInput,
        autoregressive_index: int = 0,
        cache: I2VCamCtrlEncoderCache | None = None,
    ) -> I2VCamCtrlEmbeddings:
        """Encode the per-AR-step image chunk and Plücker camera volume.

        Args:
            input: Image chunk plus camera intrinsics/poses for this AR step.
            autoregressive_index: AR step index forwarded to both branches.
            cache: Per-rollout encoder cache. Typed ``Optional`` only to
                match the :class:`Encoder` base signature (some encoders
                are stateless); this encoder advances per-AR-step state
                in ``cache`` and asserts when it is ``None``.

        Returns:
            Composite I2V latent + Plücker embedding for the transformer to cross-attend to.
        """
        assert cache is not None, "I2VCamCtrlEncoder requires a per-rollout cache."
        assert input.i2v is not None, (
            "I2VCamCtrlEncoder.forward requires the per-AR-step image chunk."
        )
        height, width = input.i2v.shape[-2:]
        i2v = self.i2v_encoder(
            input=input.i2v,
            autoregressive_index=autoregressive_index,
            cache=cache.i2v,
        )
        plucker = self._render_plucker(
            height=height,
            width=width,
            intrinsics=input.camctrl.intrinsics,
            poses=input.camctrl.poses,
            world_scale=input.camctrl.world_scale,
            cache=cache,
        )
        plucker = self.plucker_encoder(
            input=plucker,
            autoregressive_index=autoregressive_index,
            cache=cache.plucker,
        )
        return I2VCamCtrlEmbeddings(i2v=i2v, plucker=plucker)

    @property
    def temporal_compression_ratio(self) -> int:
        return self.i2v_encoder.temporal_compression_ratio

    @property
    def spatial_compression_ratio(self) -> int:
        return self.i2v_encoder.spatial_compression_ratio

    def get_output_temporal_size(
        self, autoregressive_index: int, input_temporal_size: int
    ) -> int:
        return self.i2v_encoder.get_output_temporal_size(
            autoregressive_index, input_temporal_size
        )

    def get_input_temporal_size(
        self, autoregressive_index: int, output_temporal_size: int
    ) -> int:
        return self.i2v_encoder.get_input_temporal_size(
            autoregressive_index, output_temporal_size
        )

    @torch.no_grad()
    def _render_plucker(
        self,
        height: int,
        width: int,
        intrinsics: Tensor,
        poses: Tensor,
        world_scale: float,
        cache: I2VCamCtrlEncoderCache,
    ) -> Tensor:
        """Render the per-AR-step Plücker pixel volume.

        Args:
            height: Pixel-space height.
            width: Pixel-space width.
            intrinsics: ``[..., T, 4]``.
            poses: ``[..., T, 4, 4]``.
            world_scale: Translation normalization scale.
            cache: The per-rollout pipeline cache (its ``last_pose`` is
                read and updated for causal cross-AR-step continuity).

        Returns:
            ``[..., T, 6, H, W]`` Plücker tensor in ``bfloat16``.
        """
        assert intrinsics.dtype == poses.dtype == torch.float32
        *batch_shape, _4, _4_ = poses.shape
        batch_size = math.prod(batch_shape)
        intrinsics_flat = intrinsics.view(batch_size, 4)
        poses_flat = poses.view(batch_size, 4, 4)

        relative_poses = compute_relative_poses_causal(
            poses_flat, world_scale, ref_pose=cache.camera_last_pose
        )
        plucker = get_plucker_embeddings(relative_poses, intrinsics_flat, height, width)
        plucker = rearrange(plucker, "b h w c -> b c h w").to(torch.bfloat16)
        plucker = plucker.reshape(*batch_shape, *plucker.shape[-3:])

        cache.camera_last_pose = poses_flat[..., -1:, :, :]
        return plucker
