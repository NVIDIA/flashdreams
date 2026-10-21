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

"""Streaming TAEHV2 Super decoder for normalized Wan 2.2 latents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_TAEHV2SUPER_REVISION = "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c"
TAEHV2SUPER_CHECKPOINT = (
    "https://raw.githubusercontent.com/madebyollin/taehv/"
    f"{_TAEHV2SUPER_REVISION}/safetensors/taew2_2_super.safetensors"
)
"""Author-published TAEHV2 Super weights, pinned to an immutable revision."""

DecoderState = tuple[Tensor, ...]


@dataclass
class TAEHV2SuperCache:
    """Per-rollout temporal state for :class:`TAEHV2SuperDecoder`."""

    decoder_state: DecoderState | None = None
    has_decoded_prefix: bool = False


def _conv(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


class _SuperMemoryBlock(nn.Module):
    """Causal temporal residual block used by TAEHV2 Super."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        joined_channels = channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                joined_channels,
                joined_channels,
                7,
                padding=3,
                groups=joined_channels,
                bias=False,
            ),
            nn.Conv2d(joined_channels, channels * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, channels, 1, bias=False),
        )

    def forward(self, current: Tensor, past: Tensor) -> Tensor:
        return self.conv(torch.cat([current, past], dim=1)) + current


class _TemporalGrow(nn.Module):
    """Increase temporal resolution by reshaping channels into frames."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x.reshape(
            -1,
            x.shape[1] // self.stride,
            x.shape[2],
            x.shape[3],
        )


class TAEHV2SuperDecoder(nn.Module):
    """TAEHV2 Super decoder with cache-explicit autoregressive streaming."""

    _dtype_reference: Tensor
    """Zero-size buffer carrying the decoder parameter dtype and device."""

    latent_channels = 48
    image_channels = 3
    patch_size = 2
    spatial_upsample_factor = 2
    temporal_upsample_factors = (1, 2, 2)
    memory_blocks_per_stage = 3
    stage_channels = (512, 256, 128, 64)

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        credential_path: str | None = None,
    ) -> None:
        super().__init__()
        self.decoder_output_channels = self.image_channels * self.patch_size**2
        self.temporal_upsample_factor = 1
        for factor in self.temporal_upsample_factors:
            self.temporal_upsample_factor *= factor
        self.frames_to_trim = self.temporal_upsample_factor - 1
        self.register_buffer("_dtype_reference", torch.empty(0))
        self.decoder = self._build_decoder()

        if checkpoint_path is not None:
            checkpoint = self._load_checkpoint(checkpoint_path, credential_path)
            decoder_state = {
                key.removeprefix("decoder."): value
                for key, value in checkpoint.items()
                if key.startswith("decoder.")
            }
            if not decoder_state:
                raise ValueError(
                    f"TAEHV2 Super checkpoint has no decoder weights: {checkpoint_path}"
                )
            self.decoder.load_state_dict(decoder_state, strict=True)

    def _load_checkpoint(
        self,
        checkpoint_path: str | Path,
        credential_path: str | None,
    ) -> dict[str, Tensor]:
        """Load and validate a TAEHV2 Super checkpoint state dictionary."""
        from flashdreams.core.checkpoint.load import load_single_checkpoint

        source = str(checkpoint_path)
        if source == TAEHV2SUPER_CHECKPOINT:
            from safetensors import safe_open

            from flashdreams.core.io.disk import default_flashdreams_cache_dir
            from flashdreams.core.io.download import download_to_cache

            def validate(path: Path) -> None:
                expected = {
                    f"decoder.{key}": tuple(value.shape)
                    for key, value in self.decoder.state_dict().items()
                }
                with safe_open(path, framework="pt", device="cpu") as weights:
                    actual = {
                        key: tuple(weights.get_slice(key).get_shape())
                        for key in weights.keys()
                        if key.startswith("decoder.")
                    }
                if actual != expected:
                    raise ValueError("TAEHV2 Super decoder keys or shapes do not match")

            source = str(
                download_to_cache(
                    source,
                    cache_dir=default_flashdreams_cache_dir()
                    / "taehv2super"
                    / _TAEHV2SUPER_REVISION,
                    validator=validate,
                )
            )
        if credential_path is None:
            checkpoint = load_single_checkpoint(source, map_location="cpu")
        else:
            checkpoint = load_single_checkpoint(
                source,
                credential_path=credential_path,
                map_location="cpu",
            )
        if type(checkpoint) is not dict or not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in checkpoint.items()
        ):
            raise ValueError(
                f"TAEHV2 Super checkpoint has an invalid state-dict format: {source}"
            )
        return checkpoint

    @staticmethod
    def prepare_cache() -> TAEHV2SuperCache:
        """Return empty temporal state for an independent video."""
        return TAEHV2SuperCache()

    def _append_decoder_stage(
        self,
        layers: list[nn.Module],
        in_channels: int,
        out_channels: int,
        temporal_upsample_factor: int,
    ) -> None:
        """Append one checkpoint-compatible Super decoder stage."""
        layers.extend(
            _SuperMemoryBlock(in_channels) for _ in range(self.memory_blocks_per_stage)
        )
        shuffle_channels = out_channels * self.spatial_upsample_factor**2
        layers.extend(
            [
                _conv(in_channels, shuffle_channels),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(self.spatial_upsample_factor),
                _TemporalGrow(out_channels, temporal_upsample_factor),
            ]
        )

    def _build_decoder(self) -> nn.Sequential:
        """Build the checkpoint-compatible TAEHV2 Super decoder topology."""
        layers: list[nn.Module] = [
            nn.Conv2d(
                self.latent_channels,
                self.stage_channels[0],
                1,
                bias=False,
            )
        ]
        transitions = zip(
            self.stage_channels[:-1],
            self.stage_channels[1:],
            self.temporal_upsample_factors,
            strict=True,
        )
        for in_channels, out_channels, temporal_factor in transitions:
            self._append_decoder_stage(
                layers,
                in_channels,
                out_channels,
                temporal_factor,
            )
        layers.append(_conv(self.stage_channels[-1], self.decoder_output_channels))
        return nn.Sequential(*layers)

    def _decode_chunk(
        self,
        latent: Tensor,
        decoder_state: DecoderState | None,
    ) -> tuple[Tensor, DecoderState]:
        """Decode a chunk as a functional cache-in/cache-out operation."""
        batch, _, frames, height, width = latent.shape
        x = latent.permute(0, 2, 1, 3, 4).reshape(
            batch * frames,
            self.latent_channels,
            height,
            width,
        )
        cache_index = 0
        next_cache: list[Tensor] = []
        for layer in self.decoder:
            if isinstance(layer, _SuperMemoryBlock):
                sequence = x.reshape(
                    batch,
                    -1,
                    x.shape[1],
                    x.shape[2],
                    x.shape[3],
                )
                if decoder_state is None:
                    cached_frame = torch.zeros_like(sequence[:, :1])
                else:
                    cached_frame = decoder_state[cache_index]
                past = torch.cat([cached_frame, sequence], dim=1)[:, :-1]
                past = past.reshape_as(x)
                next_cache.append(sequence[:, -1:].detach())
                cache_index += 1
                x = layer(x, past)
            else:
                x = layer(x)

        x = x.reshape(
            batch,
            -1,
            self.decoder_output_channels,
            x.shape[2],
            x.shape[3],
        )
        x = F.pixel_shuffle(x.flatten(0, 1), self.patch_size)
        x = x.reshape(batch, -1, self.image_channels, x.shape[2], x.shape[3])
        return x.permute(0, 2, 1, 3, 4), tuple(next_cache)

    def forward(
        self,
        latent: Tensor,
        *,
        cache: TAEHV2SuperCache | None = None,
    ) -> Tensor:
        """Decode a full video or advance the supplied rollout cache."""
        if latent.ndim != 5 or latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected latent [B,{self.latent_channels},T,H,W], "
                f"got {tuple(latent.shape)}"
            )
        cache = self.prepare_cache() if cache is None else cache
        output_dtype = latent.dtype
        latent = latent.to(self._dtype_reference)
        pixels, cache.decoder_state = self._decode_chunk(
            latent,
            cache.decoder_state,
        )
        trim = self.frames_to_trim if not cache.has_decoded_prefix else 0
        cache.has_decoded_prefix = True
        return pixels[:, :, trim:].clamp_(0, 1).to(output_dtype)


def make_taehv2super_decoder(
    *,
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: str | Path | None = None,
    credential_path: str | None = None,
    device: str | torch.device = "cuda",
) -> TAEHV2SuperDecoder:
    """Load the TAEHV2 Super module; rollout state is created separately."""
    from ..checkpoint import resolve_credential_path

    source = str(checkpoint_path or TAEHV2SUPER_CHECKPOINT)
    credential = (
        resolve_credential_path(credential_path) if source.startswith("s3://") else None
    )
    print(f"[vae] decoder=taehv2super checkpoint={source}", flush=True)
    decoder = TAEHV2SuperDecoder(
        source,
        credential_path=credential,
    )
    return decoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
