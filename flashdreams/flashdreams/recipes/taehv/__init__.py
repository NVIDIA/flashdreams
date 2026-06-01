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

"""TAEHV video decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.core.io.internal import use_internal_storage
from flashdreams.infra.decoder import DecoderConfig, StreamingVideoDecoder
from flashdreams.recipes.taehv.checkpoint import (
    StateDictTransform,
    compose,
    legacy_to_blocks_keys,
    truncate_oversize_tgrow_weights,
)
from flashdreams.recipes.taehv.impl import TAEHV, TAEHVCache

_INTERNAL_TAEHV_CHECKPOINT_PATHS = {
    "lighttae": "s3://flashdreams/assets/checkpoints/autoencoders/lighttaew2_1.pth",
}

_PUBLIC_TAEHV_CHECKPOINT_PATHS = {
    "lighttae": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/lighttaew2_1.pth",
}

AVAILABLE_TAEHV_CHECKPOINT_PATHS = (
    _INTERNAL_TAEHV_CHECKPOINT_PATHS
    if use_internal_storage()
    else _PUBLIC_TAEHV_CHECKPOINT_PATHS
)
"""Resolved at module import; set ``FLASHDREAMS_INTERNAL_STORAGE`` first."""


_LIGHTTAE_CHANNELS: tuple[int, int, int, int] = (256, 128, 64, 64)
"""TAEHV ``Decoder`` block widths the ``lighttae`` weights were trained
for. Mirrors :attr:`TAEHV.channels` for the default config; kept here so
:data:`lighttae_state_dict_transform` can pre-compute the per-``TGrow``
expected output channel widths without an instantiated model."""


lighttae_state_dict_transform: StateDictTransform = compose(
    legacy_to_blocks_keys,
    truncate_oversize_tgrow_weights(channels=_LIGHTTAE_CHANNELS),
)
"""Per-checkpoint remap for the ``lighttae`` weights. Rewrites the flat
``decoder.<i>.*`` keys to the current ``decoder.blocks.<i>.*`` layout
and clips the stride=2 ``TGrow`` weights at idx 7 down to the stride=1
slice the live model expects."""


@dataclass(kw_only=True)
class TeahvVAEDecoderConfig(DecoderConfig):
    """Config for the TAEHV decoder."""

    _target: type["TeahvVAEDecoder"] = field(default_factory=lambda: TeahvVAEDecoder)

    checkpoint_path: str = AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"]
    """Path to a pretrained TAEHV checkpoint. Defaults to the ``lighttae`` weights."""

    state_dict_transform: StateDictTransform | None = lighttae_state_dict_transform
    """Pre-load state-dict remap. Defaults to
    :data:`lighttae_state_dict_transform`; ``None`` falls through to the
    bare :class:`TAEHV` default (see
    :meth:`~flashdreams.recipes.taehv.impl.TAEHV.load_from_checkpoint`)."""

    dtype: torch.dtype = torch.bfloat16
    """Network parameter / activation dtype."""

    use_cuda_graph: bool = True
    """Wrap the decoder forward in a CUDA graph for replay."""

    use_compile: bool = True
    """``torch.compile(mode="max-autotune-no-cudagraphs")``."""


class TeahvVAEDecoder(StreamingVideoDecoder[TAEHVCache]):
    """TAEHV (Tiny AutoEncoder for Hunyuan Video) decoder.

    Forward input is a latent ``[..., Tl, Cl, Hl, Wl]``; output is a video
    tensor ``[..., T, C, H, W]`` in ``[-1, 1]``.

    Set ``torch.backends.cudnn.benchmark = True`` at process start for ~5%
    extra on the eager seed/tail chunks.
    """

    TEMPORAL_COMPRESSION_RATIO = TAEHV.TEMPORAL_COMPRESSION_RATIO
    SPATIAL_COMPRESSION_RATIO = TAEHV.SPATIAL_COMPRESSION_RATIO

    mean: Tensor
    """Per-channel latent mean buffer; registered only when ``need_scaled``."""

    std: Tensor
    """Per-channel latent standard deviation buffer; registered only when ``need_scaled``."""

    _LIGHTTAE_MEAN: tuple[float, ...] = (
        -0.7571, -0.7089, -0.9113, 0.1075,
        -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632,
        -0.1922, -0.9497, 0.2503, -0.2921,
    )  # fmt: skip
    """Per-channel mean for the ``lighttae`` checkpoint's latent scaling."""

    _LIGHTTAE_STD: tuple[float, ...] = (
        2.8184, 1.4541, 2.3275, 2.6558,
        1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579,
        1.6382, 1.1253, 2.8251, 1.9160,
    )  # fmt: skip
    """Per-channel standard deviation for the ``lighttae`` checkpoint's latent scaling."""

    def __init__(self, config: TeahvVAEDecoderConfig) -> None:
        super().__init__(config)
        self.config: TeahvVAEDecoderConfig = config

        self.need_scaled = "lighttae" in config.checkpoint_path
        self.taehv = TAEHV(
            checkpoint_path=config.checkpoint_path,
            use_cuda_graph=config.use_cuda_graph,
            use_compile=config.use_compile,
            state_dict_transform=config.state_dict_transform,
        ).to(dtype=config.dtype)

        if self.need_scaled:
            self.register_buffer(
                "mean",
                torch.tensor(self._LIGHTTAE_MEAN, dtype=config.dtype).view(
                    1, 1, -1, 1, 1
                ),
                persistent=False,
            )
            self.register_buffer(
                "std",
                torch.tensor(self._LIGHTTAE_STD, dtype=config.dtype).view(
                    1, 1, -1, 1, 1
                ),
                persistent=False,
            )

    def initialize_autoregressive_cache(self) -> TAEHVCache:
        """Return an empty streaming decoder cache."""
        return self.taehv.prepare_cache()

    @torch.inference_mode()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: TAEHVCache | None = None,
    ) -> Tensor:
        """Decode a latent chunk to a video tensor in ``[-1, 1]``.

        Args:
            input: Latent of shape ``[..., Tl, Cl, Hl, Wl]``.
            autoregressive_index: Unused by TAEHV; kept for the
                :class:`~flashdreams.infra.decoder.StreamingDecoder` interface.
            cache: Streaming decoder cache; created on the fly when ``None``.

        Returns:
            Video tensor of shape ``[..., T, C, H, W]`` in ``[-1, 1]``.
        """
        if cache is None:
            cache = self.initialize_autoregressive_cache()

        assert input.ndim >= 4, (
            f"Expected input to have shape [..., T, C, H, W] (ndim>=4), "
            f"got ndim={input.ndim}"
        )

        *batch_shape, T, C, H, W = input.shape
        batch_size = math.prod(batch_shape)
        z = input.reshape(batch_size, T, C, H, W)

        if self.need_scaled:
            z = z * self.std
            z = z + self.mean

        x = self.taehv.decode(z, cache=cache).mul_(2).sub_(1)
        return x.reshape(*batch_shape, *x.shape[1:])

    @property
    def temporal_compression_ratio(self) -> int:
        """Pixel frames / latent frames in steady state (AR >= 1)."""
        return self.TEMPORAL_COMPRESSION_RATIO

    @property
    def spatial_compression_ratio(self) -> int:
        """Pixel side / latent side."""
        return self.SPATIAL_COMPRESSION_RATIO

    def get_output_temporal_size(
        self, autoregressive_index: int, input_temporal_size: int
    ) -> int:
        """Return pixel frame count from ``input_temporal_size`` latent frames.

        AR 0 applies causal padding: the first latent frame yields one pixel
        frame, remaining frames yield ``temporal_compression_ratio`` each.
        AR >= 1 is a plain multiply."""
        r = self.temporal_compression_ratio
        if autoregressive_index == 0:
            return 1 + (input_temporal_size - 1) * r
        return input_temporal_size * r

    def get_input_temporal_size(
        self, autoregressive_index: int, output_temporal_size: int
    ) -> int:
        """Return latent frame count needed to produce ``output_temporal_size`` pixel frames.

        Inverse of :meth:`get_output_temporal_size`."""
        r = self.temporal_compression_ratio
        if autoregressive_index == 0:
            assert (output_temporal_size - 1) % r == 0, (
                f"AR 0 output_temporal_size={output_temporal_size} must satisfy "
                f"(N - 1) % temporal_compression_ratio={r} == 0."
            )
            return 1 + (output_temporal_size - 1) // r
        assert output_temporal_size % r == 0, (
            f"AR>=1 output_temporal_size={output_temporal_size} must be divisible "
            f"by temporal_compression_ratio={r}."
        )
        return output_temporal_size // r


if __name__ == "__main__":
    import tyro

    config = tyro.cli(TeahvVAEDecoderConfig)
    model = config.setup()
    print(model)
