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

"""Wan 2.x VAE: streaming causal encode / decode.

Covers both the Wan 2.1 / 14B 8x-spatial 16-channel VAE (default knobs)
and the Wan 2.2 TI2V 5B 16x-spatial 48-channel residual VAE
(``base_dim=160`` / ``decoder_base_dim=256`` / ``z_dim=48`` /
``patch_size=2`` / ``is_residual=True``). The streaming + causal
caching infrastructure is shared; the 5B-specific knobs flip in
diffusers' ``WanResidualDownBlock`` / ``WanResidualUpBlock``
shortcuts and the outer spatial patchify/unpatchify wrap.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, TypedDict, get_args

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.compile import compile_module
from flashdreams.infra.cuda_graph import CUDAGraphWrapper, set_or_copy
from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoderCache,
    StreamingVideoDecoder,
)
from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoderCache,
    StreamingVideoEncoder,
)

# Wan 2.2 TI2V 5B's VAE ships in the diffusers Wan-AI repo. The
# loader pulls the diffusers safetensors shard and remaps keys via
# :func:`wan22_ti2v_5b_vae_state_dict_transform` (``encoder.conv_in``
# / ``quant_conv`` / ``post_quant_conv`` etc. -> our internal layout).
WAN22_TI2V_5B_VAE_DIFFUSERS_PATH = "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/resolve/main/vae/diffusion_pytorch_model.safetensors"

# Alternative upstream single-file checkpoint. Top-level key prefixes
# (``encoder.*`` / ``decoder.*`` / ``conv1`` / ``conv2``) line up with
# our internal :mod:`_residual_vae` model shape, BUT the ``.pth`` does
# not cover every parameter our ``WanVAE`` wrapper builds (some encoder
# / decoder slots stay on meta and ``model.to(device)`` raises
# ``NotImplementedError: Cannot copy out of meta tensor``). Kept as a
# constant for callers who want to invest in a tighter audit + a
# tailored loader; the diffusers path above is still the default.
WAN22_TI2V_5B_VAE_PATH = (
    "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth"
)

AVAILABLE_WAN_VAE_CHECKPOINT_PATHS = {
    "lightvae": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/lightvaew2_1.pth",
    "vae": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/Wan2.1_VAE.pth",
}
"""Checkpoint paths for the Wan VAE encoder."""

CACHE_T = 2
TEMPORAL_WINDOW = 4


_WAN21_LATENT_MEAN: tuple[float, ...] = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
"""16-channel latent mean for Wan 2.1 / 14B VAE."""

_WAN21_LATENT_STD: tuple[float, ...] = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)
"""16-channel latent std for Wan 2.1 / 14B VAE."""


_WAN22_TI2V_5B_LATENT_MEAN: tuple[float, ...] = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)  # fmt: skip
"""48-channel latent mean for the Wan 2.2 TI2V 5B VAE
(``Wan-AI/Wan2.2-TI2V-5B-Diffusers/vae/config.json``)."""

_WAN22_TI2V_5B_LATENT_STD: tuple[float, ...] = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)  # fmt: skip
"""48-channel latent std for the Wan 2.2 TI2V 5B VAE
(``Wan-AI/Wan2.2-TI2V-5B-Diffusers/vae/config.json``)."""


@dataclass
class WanVAECache(StreamingEncoderCache, StreamingDecoderCache):
    """Streaming state for one encode + decode rollout.

    Both dicts are populated lazily on the first chunk and advanced in place
    thereafter. Get a fresh instance via ``WanVAE.prepare_cache``.

    Per-block buffers are keyed by ``id(module)``.
    """

    enc_state: Dict[int, torch.Tensor] = field(default_factory=dict)
    """Per-block encoder cache."""

    dec_state: Dict[int, torch.Tensor] = field(default_factory=dict)
    """Per-block decoder cache."""


# Forwarded verbatim to ``F.pad(mode=...)``; using its names avoids a
# translation step and keeps a single source of truth for the runtime
# check below (``get_args(PadMode)``) and the static type.
PadMode = Literal["constant", "replicate"]


class CausalConv3d(nn.Conv3d):
    """3D conv with causal time padding and a streaming left-context slot.

    ``pad_mode`` matches ``F.pad`` mode names:

    - ``"constant"`` (default; Wan VAE): zero-pad spatial + temporal halos.
    - ``"replicate"`` (FlashVSR projector): replicate-pad both halos. The
      FlashVSR projector relies on this so the cold-start chunk's first
      frames remain bounded at the activation level.
    """

    # Concrete attribute types so callers don't see ``Tensor | Module``.
    _spatial_pad: tuple[int, int, int, int]
    _has_spatial_pad: bool
    _time_pad: int
    _pad_mode: PadMode

    def __init__(
        self,
        *args,
        pad_mode: PadMode = "constant",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # ``nn.Conv3d.padding`` is typed as the ``Union[int, _size, str]``
        # that the constructor accepts; narrow once here so the rest of
        # the class can subscript it without ignoring type errors.
        assert isinstance(self.padding, tuple), (
            f"CausalConv3d expects a tuple padding; got {self.padding!r}"
        )
        ph, pw = self.padding[1], self.padding[2]
        self._spatial_pad = (pw, pw, ph, ph)
        self._has_spatial_pad = ph > 0 or pw > 0
        self._time_pad = 2 * self.padding[0]
        assert pad_mode in get_args(PadMode), (
            f"CausalConv3d pad_mode must be one of {get_args(PadMode)}; got {pad_mode!r}"
        )
        self._pad_mode = pad_mode
        self.padding = (0, 0, 0)

    def forward(
        self, x: torch.Tensor, prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        time_pad = self._time_pad
        if prev is not None and time_pad > 0:
            x = torch.cat([prev, x], dim=2)
            time_pad = max(0, time_pad - prev.shape[2])
        if time_pad or self._has_spatial_pad:
            x = F.pad(x, (*self._spatial_pad, time_pad, 0), mode=self._pad_mode)
        return super().forward(x)

    def cache_step(
        self, x: torch.Tensor, state: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Run the conv and advance the streaming left-context cache.

        The new tail is the last ``CACHE_T`` frames of ``x``. The
        ``< CACHE_T`` branch only fires on the eager first chunk.
        """
        key = id(self)
        prev = state.get(key)
        out = self.forward(x, prev)
        new_tail = x[:, :, -CACHE_T:]
        if new_tail.shape[2] < CACHE_T and prev is not None:
            new_tail = torch.cat([prev[:, :, -1:], new_tail], dim=2)
        set_or_copy(state, key, new_tail)
        return out


class RMS_norm(nn.Module):
    """RMS-normalisation with a learnable channel scale and optional bias.

    ``bias=False`` (default; Wan VAE) keeps the parameter count at one
    learnable scale per channel. ``bias=True`` adds a matching learnable
    offset (FlashVSR projector convention).
    """

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        broadcast = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcast) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        # Sentinel scalar zero when no bias is requested -- avoids a
        # branch in ``forward`` and lets ``self.bias`` be a tensor or a
        # Python float without changing the call site.
        self.bias: nn.Parameter | float = (
            nn.Parameter(torch.zeros(shape)) if bias else 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma + self.bias


def _bt_flatten(x: torch.Tensor) -> torch.Tensor:
    """[b, c, t, h, w] -> [b*t, c, h, w] (b outer, t inner)."""
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def _bt_unflatten(x: torch.Tensor, b: int) -> torch.Tensor:
    """[b*t, c, h, w] -> [b, c, t, h, w] (inverse of ``_bt_flatten``)."""
    bt, c, h, w = x.shape
    t = bt // b
    return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class Resample(nn.Module):
    """Spatial 2x resample, optionally with temporal up/down sample.

    The 3D modes (``upsample3d`` / ``downsample3d``) keep a streaming
    left-context slot in ``state``; the 2D modes are stateless.
    """

    def __init__(self, dim: int, mode: str):
        assert mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d"), (
            f"Unknown resample mode: {mode}"
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
                )
        else:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return _bt_unflatten(self.resample(_bt_flatten(x)), b)

    def forward(self, x: torch.Tensor, state: Dict[int, torch.Tensor]) -> torch.Tensor:
        if self.mode == "upsample3d":
            return self._spatial(self._upsample3d_step(x, state))
        elif self.mode == "downsample3d":
            return self._downsample3d_step(self._spatial(x), state)
        else:
            return self._spatial(x)

    def _upsample3d_step(
        self, x: torch.Tensor, state: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        b, c, t, h, w = x.shape
        key = id(self)
        prev = state.get(key)

        if prev is not None:
            # Steady-state body: in-place write to the existing slot.
            x_up = self._interleave_time(self.time_conv(x, prev), b, c, t, h, w)
            set_or_copy(state, key, x[:, :, -CACHE_T:])
            return x_up

        # Eager first-chunk path (shape varies, never captured).
        first = x[:, :, :1]
        rest = x[:, :, 1:]
        if rest.shape[2] > 0:
            up = self._interleave_time(self.time_conv(rest), b, c, rest.shape[2], h, w)
            state[key] = self._first_chunk_tail(rest, b, c, h, w)
            return torch.cat([first, up], dim=2)

        # 1-frame first chunk: zero cache (legacy "Rep" sentinel).
        state[key] = x.new_zeros(b, c, CACHE_T, h, w)
        return first

    def _downsample3d_step(
        self, x: torch.Tensor, state: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        key = id(self)
        prev = state.get(key)
        # Snapshot the input tail before time_conv to match the legacy cache.
        new_tail = x[:, :, -1:]
        if prev is not None:
            x = self.time_conv(torch.cat([prev, x], dim=2))
        set_or_copy(state, key, new_tail)
        return x

    @staticmethod
    def _interleave_time(
        x: torch.Tensor, b: int, c: int, t: int, h: int, w: int
    ) -> torch.Tensor:
        """[b, 2c, t, h, w] -> [b, c, 2t, h, w], interleaved along time."""
        x = x.reshape(b, 2, c, t, h, w)
        return torch.stack((x[:, 0], x[:, 1]), dim=3).reshape(b, c, t * 2, h, w)

    @staticmethod
    def _first_chunk_tail(
        rest: torch.Tensor, b: int, c: int, h: int, w: int
    ) -> torch.Tensor:
        """Last CACHE_T frames of ``rest``, zero-padded if too short."""
        tail = rest[:, :, -CACHE_T:].clone()
        if tail.shape[2] < CACHE_T:
            tail = torch.cat([rest.new_zeros(b, c, 1, h, w), tail], dim=2)
        return tail


class AvgDown3D(nn.Module):
    """Average-pool 3D shortcut path for ``ResidualDownBlock`` (Wan 2.2).

    Pixel-shuffle-style spatial / temporal downsample that maps
    ``[B, in_ch, T, H, W]`` to ``[B, out_ch, T/factor_t, H/factor_s,
    W/factor_s]`` by mean-pooling over a ``(factor_t, factor_s,
    factor_s)`` neighborhood after channel-grouping. Used as the
    by-pass arm next to the residual block + ``Resample`` main path,
    so the shortcut has the same shape but doesn't go through the
    learned residual stack. Stateless (no streaming cache).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s

        assert in_channels * self.factor % out_channels == 0, (
            f"AvgDown3D: in_channels*factor ({in_channels} * {self.factor}) "
            f"must be divisible by out_channels ({out_channels})."
        )
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal left-pad on T to make T divisible by factor_t.
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        x = x.view(
            b,
            c,
            t // self.factor_t,
            self.factor_t,
            h // self.factor_s,
            self.factor_s,
            w // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            b,
            c * self.factor,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )
        x = x.view(
            b,
            self.out_channels,
            self.group_size,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    """Repeat-interleave 3D shortcut path for ``ResidualUpBlock`` (Wan 2.2).

    Inverse of :class:`AvgDown3D`: maps ``[B, in_ch, T, H, W]`` to
    ``[B, out_ch, T*factor_t, H*factor_s, W*factor_s]`` by repeating
    each channel ``repeats = out_ch * factor / in_ch`` times and
    rearranging into the upsampled grid. ``first_chunk=True`` drops
    the leading ``factor_t - 1`` frames so the AR-step-0 output stays
    one decoded frame, matching the encoder's seed semantics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s

        assert out_channels * self.factor % in_channels == 0, (
            f"DupUp3D: out_channels*factor ({out_channels} * {self.factor}) "
            f"must be divisible by in_channels ({in_channels})."
        )
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class ResidualBlock(nn.Module):
    """Two-conv residual block with RMS-norm + SiLU."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut: nn.Module = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor, state: Dict[int, torch.Tensor]) -> torch.Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            x = (
                layer.cache_step(x, state)
                if isinstance(layer, CausalConv3d)
                else layer(x)
            )
        return x + h


class AttentionBlock(nn.Module):
    """Single-head self-attention; stateless across streaming chunks."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    # ``state`` is accepted (and ignored) for a uniform iteration call site.
    def forward(
        self, x: torch.Tensor, state: Optional[Dict[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        b, c, t, h, w = x.shape
        identity = x
        x = _bt_flatten(x)
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, h * w)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        return _bt_unflatten(x, b) + identity


class ResidualDownBlock(nn.Module):
    """Wan 2.2 down-stage: residual blocks + Resample + ``AvgDown3D`` shortcut.

    Wraps ``num_res_blocks`` :class:`ResidualBlock` instances followed
    by an optional :class:`Resample` (``downsample3d`` /
    ``downsample2d`` depending on ``temperal_downsample``) and adds
    an :class:`AvgDown3D` by-pass arm. The output is
    ``main_path(x) + avg_shortcut(x)``. Used by the residual-style
    encoder (``WanVAE(is_residual=True)``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        num_res_blocks: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets: list[nn.Module] = []
        for _ in range(num_res_blocks):
            resnets.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler: Resample | None = Resample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x: torch.Tensor, state: Dict[int, torch.Tensor]) -> torch.Tensor:
        # Snapshot input for the avg shortcut before mutating ``x``.
        x_shortcut = x
        for resnet in self.resnets:
            x = resnet(x, state)
        if self.downsampler is not None:
            x = self.downsampler(x, state)
        return x + self.avg_shortcut(x_shortcut)


class ResidualUpBlock(nn.Module):
    """Wan 2.2 up-stage: residual blocks + Resample + ``DupUp3D`` shortcut.

    Inverse counterpart to :class:`ResidualDownBlock`. Uses
    ``num_res_blocks + 1`` :class:`ResidualBlock` instances followed
    by an optional :class:`Resample` upsampler that *keeps* the
    channel count (``out_dim -> out_dim`` rather than the channel-
    halving the legacy ``Decoder3d`` uses) and adds a :class:`DupUp3D`
    shortcut.

    ``first_chunk=True`` is forwarded into the shortcut so AR step 0
    matches the encoder's single-frame seed.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ) -> None:
        super().__init__()
        if up_flag:
            self.avg_shortcut: DupUp3D | None = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        resnets: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            # Wan 2.2 keeps the channel count through the upsampler
            # (unlike the legacy Wan 2.1 ``Resample`` which halves
            # to ``dim // 2``); patch the inner conv after-the-fact
            # so we don't have to fork ``Resample``.
            up = Resample(out_dim, mode=mode)
            assert isinstance(up.resample[1], nn.Conv2d), (
                "Resample's spatial conv slot 1 layout changed; refusing to "
                "blindly rewire ResidualUpBlock's keep-channel upsampler."
            )
            up.resample[1] = nn.Conv2d(out_dim, out_dim, 3, padding=1)
            self.upsampler: Resample | None = up
        else:
            self.upsampler = None

    def forward(
        self,
        x: torch.Tensor,
        state: Dict[int, torch.Tensor],
        first_chunk: bool = False,
    ) -> torch.Tensor:
        x_shortcut = x
        for resnet in self.resnets:
            x = resnet(x, state)
        if self.upsampler is not None:
            x = self.upsampler(x, state)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_shortcut, first_chunk=first_chunk)
        return x


def _patchify(x: Tensor, patch_size: int) -> Tensor:
    """Pre-encode spatial pixel-shuffle, ``[B, C, T, H, W]`` -> ``[B, C*p*p, T, H/p, W/p]``.

    ``patch_size == 1`` short-circuits to identity. Used by Wan 2.2
    TI2V 5B's VAE to fold an extra 2x spatial compression into the
    encoder's input channel count, lifting the effective spatial
    compression from 8x (encoder stages) to 16x without changing the
    encoder body.
    """
    if patch_size == 1:
        return x
    assert x.ndim == 5, f"_patchify expects [B, C, T, H, W]; got {tuple(x.shape)}"
    return rearrange(
        x,
        "b c t (h ph) (w pw) -> b (c ph pw) t h w",
        ph=patch_size,
        pw=patch_size,
    )


def _unpatchify(x: Tensor, patch_size: int) -> Tensor:
    """Post-decode spatial inverse of :func:`_patchify`."""
    if patch_size == 1:
        return x
    assert x.ndim == 5, f"_unpatchify expects [B, C, T, H, W]; got {tuple(x.shape)}"
    return rearrange(
        x,
        "b (c ph pw) t h w -> b c t (h ph) (w pw)",
        ph=patch_size,
        pw=patch_size,
    )


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        temperal_downsample=(True, True, False),
        dropout: float = 0.0,
        pruning_rate: float = 0.0,
        in_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        dims = [int(dim * u * (1 - pruning_rate)) for u in (1,) + tuple(dim_mult)]
        scale = 1.0

        self.conv1 = CausalConv3d(in_channels, dims[0], 3, padding=1)

        downsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            is_last_stage = i == len(dim_mult) - 1
            if is_residual:
                # Wan 2.2: residual down-stage bundles
                # ``num_res_blocks + Resample + AvgDown3D``-shortcut so
                # the per-stage shortcut composition stays correct.
                # The legacy path keeps the original Wan 2.1
                # ResidualBlock + AttentionBlock layout.
                downsamples.append(
                    ResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=(
                            temperal_downsample[i] if not is_last_stage else False
                        ),
                        down_flag=not is_last_stage,
                    )
                )
                if not is_last_stage:
                    scale /= 2.0
                continue
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if not is_last_stage:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, state: Dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.conv1.cache_step(x, state)
        for layer in self.downsamples:
            x = layer(x, state)
        for layer in self.middle:
            x = layer(x, state)
        norm, act, conv = self.head
        assert isinstance(conv, CausalConv3d)
        return conv.cache_step(act(norm(x)), state)

    @torch.no_grad()
    def normalize_state_for_body(self, state: Dict[int, torch.Tensor]) -> None:
        """Pad every CausalConv3d's seed-shaped state entry up to ``CACHE_T`` frames.

        After ``WanVAE.encode``'s 1-frame seed call, each ``CausalConv3d`` has
        stored ``state[id(cc3d)]`` as the last-CACHE_T-frames slice of a 1-frame
        input -- so the stored tensor has T=1 rather than T=CACHE_T. This
        triggers a whole-graph recompile on the first body chunk (T=4) of the
        NEXT AR step because Dynamo specialized on the AR0 body's T=1 prev
        state and AR1 body's prev state is T=CACHE_T=2.

        Prepending zeros to make every entry T=CACHE_T is bit-equivalent to the
        ``F.pad`` zero-prepad ``CausalConv3d.forward`` would have done if
        ``prev`` had been shorter than ``time_pad`` -- verified by tracing the
        conv input in both code paths. ``Resample._upsample3d_step`` /
        ``_downsample3d_step`` already store T=CACHE_T (the upsample's 1-frame
        branch explicitly allocates ``x.new_zeros(..., CACHE_T, ...)``, the
        downsample stores a fixed T=1 tail), so this only touches CausalConv3d
        entries.
        """
        for module in self.modules():
            if not isinstance(module, CausalConv3d):
                continue
            key = id(module)
            if key not in state:
                continue
            prev = state[key]
            if prev.shape[2] >= CACHE_T:
                continue
            pad = CACHE_T - prev.shape[2]
            b, c, _, h, w = prev.shape
            zeros = prev.new_zeros(b, c, pad, h, w)
            state[key] = torch.cat([zeros, prev], dim=2)


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        temperal_upsample=(False, True, True),
        dropout: float = 0.0,
        pruning_rate: float = 0.0,
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        dims = [
            int(dim * u * (1 - pruning_rate))
            for u in (dim_mult[-1],) + tuple(dim_mult[::-1])
        ]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        self._is_residual_decoder = is_residual

        upsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            is_last_stage = i == len(dim_mult) - 1
            if is_residual:
                # Wan 2.2: bundle ``num_res_blocks + 1`` residual blocks
                # with the matched ``DupUp3D`` shortcut. The residual
                # variant keeps the channel count through the upsampler
                # so we don't need the legacy ``in_dim //= 2`` adjust.
                upsamples.append(
                    ResidualUpBlock(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        num_res_blocks=num_res_blocks,
                        dropout=dropout,
                        temperal_upsample=(
                            temperal_upsample[i] if not is_last_stage else False
                        ),
                        up_flag=not is_last_stage,
                    )
                )
                if not is_last_stage:
                    scale *= 2.0
                continue
            # Wan 2.1 legacy path: stages 1-3 halve their input dim
            # because the preceding ``Resample`` already halved channels.
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if not is_last_stage:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Dict[int, torch.Tensor],
        first_chunk: bool = False,
    ) -> torch.Tensor:
        x = self.conv1.cache_step(x, state)
        for layer in self.middle:
            x = layer(x, state)
        for layer in self.upsamples:
            # ``ResidualUpBlock`` needs the ``first_chunk`` flag so its
            # ``DupUp3D`` shortcut drops the leading ``factor_t-1`` frames
            # on AR step 0; legacy ``ResidualBlock`` / ``Resample`` /
            # ``AttentionBlock`` ignore it.
            if isinstance(layer, ResidualUpBlock):
                x = layer(x, state, first_chunk=first_chunk)
            else:
                x = layer(x, state)
        norm, act, conv = self.head
        # ``nn.Sequential`` typing hands back ``Module``; the final entry
        # is a ``CausalConv3d`` and we need its non-``Module`` ``cache_step``.
        assert isinstance(conv, CausalConv3d)
        return conv.cache_step(act(norm(x)), state)


class WanVAE(nn.Module):
    """Wan 2.x video VAE: streaming causal encode and decode.

    Configurable for two model families that share the streaming + causal
    cache infrastructure:

    * **Wan 2.1 / 14B (default knobs).** Input ``[B, 3, T, H, W]`` in
      ``[-1, 1]``; latent ``[B, 16, Tl, H/8, W/8]``. ``base_dim=96``,
      ``z_dim=16``, ``patch_size=1``, ``is_residual=False``, ``8x``
      spatial compression.
    * **Wan 2.2 TI2V 5B.** Input ``[B, 3, T, H, W]`` in ``[-1, 1]``;
      latent ``[B, 48, Tl, H/16, W/16]``. ``base_dim=160`` (encoder),
      ``decoder_base_dim=256``, ``z_dim=48``, ``patch_size=2``,
      ``is_residual=True``, ``16x`` spatial compression. The 2x outer
      patchify wrap turns the encoder's 8x stages into 16x effective
      compression and matches diffusers ``AutoencoderKLWan`` (see
      ``Wan-AI/Wan2.2-TI2V-5B-Diffusers/vae/config.json``).

    With ``use_cuda_graph=True``, rollout 1 drains Inductor autotune on the
    eager path so rollout 2 can warmup + capture; subsequent same-shape body
    chunks replay the captured graph.

    Set ``enable_encoder=False`` (or ``enable_decoder=False``) when only one
    direction is needed; the unused half's parameters and graph state are
    skipped, saving VRAM.

    Examples:

        vae = WanVAE(vae_path="...", use_lightvae=True).to("cuda", torch.bfloat16)
        cache = vae.prepare_cache()
        z = vae.encode(video, cache=cache)
        x = vae.decode(z, cache=vae.prepare_cache())

    Set ``torch.backends.cudnn.benchmark = True`` at process start for ~5%
    extra on the eager seed/tail chunks.
    """

    # Class-level defaults match the Wan 2.1 layout for back-compat;
    # the constructor's ``base_dim`` / ``z_dim`` / ``patch_size`` /
    # ``is_residual`` knobs override them for Wan 2.2 5B.
    Z_DIM = 16
    BASE_DIM = 96
    PATCH_SIZE = 1
    IS_RESIDUAL = False

    TEMPORAL_COMPRESSION_RATIO = 4
    SPATIAL_COMPRESSION_RATIO = 8

    mean: Tensor
    inv_std: Tensor

    def __init__(
        self,
        vae_path: str,
        use_lightvae: bool = False,
        use_cuda_graph: bool = True,
        use_compile: bool = True,
        warmup_iters: int = 2,
        enable_encoder: bool = True,
        enable_decoder: bool = True,
        base_dim: int = BASE_DIM,
        decoder_base_dim: int | None = None,
        z_dim: int = Z_DIM,
        patch_size: int = PATCH_SIZE,
        is_residual: bool = IS_RESIDUAL,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        temperal_downsample: Sequence[bool] = (False, True, True),
        temperal_upsample: Sequence[bool] | None = None,
        latent_mean: Sequence[float] = _WAN21_LATENT_MEAN,
        latent_std: Sequence[float] = _WAN21_LATENT_STD,
        encoder_in_channels: int = 3,
        decoder_out_channels: int = 3,
        state_dict_transform: Callable[
            [Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
        ]
        | None = None,
    ):
        super().__init__()
        assert enable_encoder or enable_decoder, (
            "WanVAE: at least one of enable_encoder / enable_decoder must be True"
        )
        # Asymmetric base dim is a Wan 2.2 5B idiosyncrasy (160 / 256);
        # default to symmetric so the Wan 2.1 path is a no-op.
        if decoder_base_dim is None:
            decoder_base_dim = base_dim
        # Temporal upsample mirrors temperal_downsample by default
        # (reversed, since the decoder runs in reverse stage order).
        if temperal_upsample is None:
            temperal_upsample = tuple(reversed(temperal_downsample))

        assert len(latent_mean) == z_dim, (
            f"latent_mean has {len(latent_mean)} entries, but z_dim={z_dim}"
        )
        assert len(latent_std) == z_dim, (
            f"latent_std has {len(latent_std)} entries, but z_dim={z_dim}"
        )

        pruning_rate = 0.75 if use_lightvae else 0.0
        self._z_dim = z_dim
        self._patch_size = patch_size
        # Effective spatial compression: 3 stage downsamples in the
        # encoder body (8x) multiplied by the outer patchify factor
        # (1 for Wan 2.1, 2 for Wan 2.2 5B = 16x total).
        self._spatial_compression_ratio = self.SPATIAL_COMPRESSION_RATIO * patch_size

        # The encoder's pre-VAE in_channels absorb the patchify factor:
        # video has 3 channels, patchify packs each ``patch_size`` x
        # ``patch_size`` spatial neighborhood into the channel dim, so
        # the encoder receives ``encoder_in_channels * patch_size ** 2``
        # input channels. Likewise for the decoder output.
        encoder_in = encoder_in_channels * patch_size * patch_size
        decoder_out = decoder_out_channels * patch_size * patch_size

        # TypedDict so the encoder/decoder factories see concrete
        # kwarg types rather than ``object``-typed dict values.
        class _CommonKwargs(TypedDict):
            dim_mult: tuple[int, ...]
            num_res_blocks: int
            attn_scales: tuple[float, ...]
            dropout: float
            pruning_rate: float
            is_residual: bool

        common: _CommonKwargs = {
            "dim_mult": tuple(dim_mult),
            "num_res_blocks": num_res_blocks,
            "attn_scales": tuple(attn_scales),
            "dropout": 0.0,
            "pruning_rate": pruning_rate,
            "is_residual": is_residual,
        }
        # Build on ``meta`` so only the checkpoint allocates real memory; skip
        # the disabled half so its params never materialise.
        with torch.device("meta"):
            if enable_encoder:
                self.encoder = Encoder3d(
                    dim=base_dim,
                    z_dim=z_dim * 2,
                    temperal_downsample=tuple(temperal_downsample),
                    in_channels=encoder_in,
                    **common,
                )
                self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
            if enable_decoder:
                self.decoder = Decoder3d(
                    dim=decoder_base_dim,
                    z_dim=z_dim,
                    temperal_upsample=tuple(temperal_upsample),
                    out_channels=decoder_out,
                    **common,
                )
                self.conv2 = CausalConv3d(z_dim, z_dim, 1)

        # assign=True: meta params become the checkpoint tensors directly;
        # caller does .to(device, dtype) afterward. strict=False tolerates
        # the disabled half (encoder-only or decoder-only). The optional
        # ``state_dict_transform`` rewires upstream key conventions (e.g.
        # diffusers ``encoder.down_blocks.*`` to flashdreams
        # ``encoder.downsamples.*``) before the load.
        state_dict = load_checkpoint(vae_path)
        if state_dict_transform is not None:
            state_dict = state_dict_transform(state_dict)
        self.load_state_dict(state_dict, strict=False, assign=True)

        self.register_buffer(
            "mean",
            torch.tensor(latent_mean).view(1, z_dim, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "inv_std",
            (1.0 / torch.tensor(latent_std)).view(1, z_dim, 1, 1, 1),
            persistent=False,
        )

        self.eval().requires_grad_(False)

        self._enable_encoder = enable_encoder
        self._enable_decoder = enable_decoder
        self._use_cuda_graph = use_cuda_graph

        self._encoder_wrapper: CUDAGraphWrapper | None = None
        self._decoder_wrapper: CUDAGraphWrapper | None = None

        if enable_encoder:
            if use_compile:
                self.encoder = compile_module(self.encoder)
            if use_cuda_graph:
                self._encoder_wrapper = CUDAGraphWrapper(
                    self.encoder, warmup_iters=warmup_iters
                )
            self._encoder_call: Callable[..., torch.Tensor] = (
                self._encoder_wrapper
                if self._encoder_wrapper is not None
                else self.encoder
            )
        if enable_decoder:
            if use_compile:
                self.decoder = compile_module(self.decoder)
            if use_cuda_graph:
                self._decoder_wrapper = CUDAGraphWrapper(
                    self.decoder, warmup_iters=warmup_iters
                )
            self._decoder_call: Callable[..., torch.Tensor] = (
                self._decoder_wrapper
                if self._decoder_wrapper is not None
                else self.decoder
            )

    def prepare_cache(self) -> WanVAECache:
        """Return a fresh empty cache and drop any captured CUDA graphs.

        Captured kernels reference the previous cache's slot pointers, so a
        new cache invalidates them and forces a re-warmup on the next call.
        """
        if self._use_cuda_graph:
            if self._enable_encoder:
                assert self._encoder_wrapper is not None
                self._encoder_wrapper.reset()
            if self._enable_decoder:
                assert self._decoder_wrapper is not None
                self._decoder_wrapper.reset()
        return WanVAECache()

    @torch.inference_mode()
    def encode(self, x: torch.Tensor, cache: WanVAECache) -> torch.Tensor:
        """Streaming causal encode (output is mean/std-normalised).

        First chunk (cache empty): 1-frame seed + 4-frame body chunks +
        optional ``< 4``-frame tail. Subsequent chunks must have ``T % 4 == 0``.
        Requires ``enable_encoder=True`` at construction.

        The outer 2x patchify wrap (Wan 2.2 5B; ``patch_size=2``) folds
        a spatial pixel-shuffle into the encoder's input channels; it
        is a no-op when ``patch_size == 1``.
        """
        assert self._enable_encoder, (
            "WanVAE.encode called but the model was constructed with "
            "enable_encoder=False"
        )
        x = _patchify(x, self._patch_size)
        state = cache.enc_state
        # Body-chunk dispatch: rollout 1 drains autotune through the same
        # static buffer the wrapper will capture against (single Inductor
        # specialisation); rollout 2+ runs the captured graph. Bind before
        # the seed call populates ``state``.
        if self._use_cuda_graph:
            assert self._encoder_wrapper is not None
            encoder_body = (
                self._encoder_wrapper.drain if not state else self._encoder_call
            )
        else:
            encoder_body = self.encoder

        outs: list[torch.Tensor] = []
        if not state:
            outs.append(self.encoder(x[:, :, :1], state))
            x = x[:, :, 1:]
            # Pad CausalConv3d states from T=1 (seed) to T=CACHE_T so the
            # AR0 body chunk and every AR1+ body chunk see identical state
            # shapes -- this kills the ~68 s whole-graph encoder recompile
            # at AR1. ``normalize_state_for_body`` is an eager ``Encoder3d``
            # helper; the ``torch.compile`` proxy forwards the call to it.
            self.encoder.normalize_state_for_body(state)
        else:
            assert x.shape[2] % TEMPORAL_WINDOW == 0, (
                f"Streaming encode after the first chunk requires T % "
                f"{TEMPORAL_WINDOW} == 0; got T={x.shape[2]}"
            )
        t = x.shape[2]
        body = (t // TEMPORAL_WINDOW) * TEMPORAL_WINDOW
        for i in range(0, body, TEMPORAL_WINDOW):
            outs.append(encoder_body(x[:, :, i : i + TEMPORAL_WINDOW], state))
        if body < t:
            outs.append(self.encoder(x[:, :, body:], state))
        mu, _log_var = self.conv1(torch.cat(outs, dim=2)).chunk(2, dim=1)
        return (mu - self.mean) * self.inv_std

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, cache: WanVAECache) -> torch.Tensor:
        """Streaming causal decode. Output is clamped to ``[-1, 1]``.

        Requires ``enable_decoder=True`` at construction.

        For Wan 2.2 5B (``is_residual=True``) the eager first-chunk call
        passes ``first_chunk=True`` so :class:`ResidualUpBlock`'s
        ``DupUp3D`` shortcut drops the leading ``factor_t-1`` frames and
        emits a single seed frame, matching diffusers'
        ``MyVAE._decode(is_first_chunk=True)``. All subsequent (captured)
        calls run with ``first_chunk=False``.
        """
        assert self._enable_decoder, (
            "WanVAE.decode called but the model was constructed with "
            "enable_decoder=False"
        )
        is_first_chunk = not cache.dec_state
        z = z / self.inv_std + self.mean
        # See encode() for the rollout 1 vs 2+ dispatch rationale.
        if self._use_cuda_graph:
            assert self._decoder_wrapper is not None
            decoder = (
                self._decoder_wrapper.drain if is_first_chunk else self._decoder_call
            )
        else:
            decoder = self.decoder
        out = decoder(
            self.conv2(z), cache.dec_state, first_chunk=is_first_chunk
        ).clamp_(-1, 1)
        return _unpatchify(out, self._patch_size)

    @property
    def temporal_compression_ratio(self) -> int:
        return self.TEMPORAL_COMPRESSION_RATIO

    @property
    def spatial_compression_ratio(self) -> int:
        return self._spatial_compression_ratio

    @property
    def z_dim(self) -> int:
        return self._z_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size


@dataclass(kw_only=True)
class WanVAEEncoderConfig(EncoderConfig):
    """Config for the Wan VAE encoder.

    Defaults reproduce the Wan 2.1 / 14B 8x-spatial 16-channel
    streaming VAE. Override ``base_dim`` / ``z_dim`` / ``patch_size`` /
    ``is_residual`` (and the latent mean/std) to load the Wan 2.2
    TI2V 5B 16x-spatial 48-channel residual VAE; see
    :class:`Wan22TI2V5BVAEEncoderConfig` for the pre-rolled set.
    """

    _target: type["WanVAEEncoder"] = field(default_factory=lambda: WanVAEEncoder)

    checkpoint_path: str = AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"]
    dtype: torch.dtype = torch.bfloat16
    use_cuda_graph: bool = True
    """Wrap the encoder forward in a CUDA graph for replay."""

    use_compile: bool = False
    """``torch.compile(mode="max-autotune-no-cudagraphs")``. Off by default:
    Inductor autotune workspaces can add several GiB of transient VRAM per
    unique input shape, surfacing as 'illegal memory access' on smaller GPUs
    with the full-channel ``vae`` checkpoint."""

    # Wan 2.x VAE architecture knobs (default = Wan 2.1).
    base_dim: int = WanVAE.BASE_DIM
    """Encoder base channel count (``WanVAE`` ``dim``). 96 for Wan 2.1,
    160 for Wan 2.2 TI2V 5B."""
    z_dim: int = WanVAE.Z_DIM
    """Latent channels. 16 for Wan 2.1, 48 for Wan 2.2 TI2V 5B."""
    patch_size: int = WanVAE.PATCH_SIZE
    """Outer spatial pixel-shuffle factor (1 = no patchify; 2 for Wan
    2.2 TI2V 5B)."""
    is_residual: bool = WanVAE.IS_RESIDUAL
    """Use ``ResidualDownBlock`` (Wan 2.2) instead of the legacy
    ``ResidualBlock + AttentionBlock`` down-stage (Wan 2.1)."""
    latent_mean: tuple[float, ...] = _WAN21_LATENT_MEAN
    """Per-channel latent mean used for normalisation; must match
    ``z_dim`` entries."""
    latent_std: tuple[float, ...] = _WAN21_LATENT_STD
    """Per-channel latent std used for normalisation."""
    state_dict_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None
    """Optional pre-``load_state_dict`` key remap (e.g. diffusers ->
    flashdreams layout). See :func:`wan22_ti2v_5b_vae_state_dict_transform`
    for the Wan 2.2 TI2V 5B remap."""


class WanVAEEncoder(StreamingVideoEncoder[WanVAECache]):
    """Wan VAE encoder.

    Forward input is a video tensor ``[..., T, C, H, W]`` in ``[-1, 1]``;
    output is the latent ``[..., Tl, Cl, Hl, Wl]``. The cache is advanced
    in-place across AR encode steps; passing ``cache=None`` allocates a
    fresh single-shot cache.
    """

    def __init__(self, config: WanVAEEncoderConfig) -> None:
        super().__init__(config)
        self.config: WanVAEEncoderConfig = config

        use_lightvae = "lightvae" in config.checkpoint_path
        self.vae = WanVAE(
            vae_path=config.checkpoint_path,
            use_lightvae=use_lightvae,
            use_cuda_graph=config.use_cuda_graph,
            use_compile=config.use_compile,
            enable_encoder=True,
            enable_decoder=False,
            base_dim=config.base_dim,
            z_dim=config.z_dim,
            patch_size=config.patch_size,
            is_residual=config.is_residual,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            state_dict_transform=config.state_dict_transform,
        ).to(dtype=config.dtype)

    def initialize_autoregressive_cache(self) -> WanVAECache:
        return self.vae.prepare_cache()

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: WanVAECache | None = None,
    ) -> Tensor:
        if cache is None:
            cache = self.initialize_autoregressive_cache()

        assert input.ndim >= 4, "Expected input to have shape [..., T, C, H, W]"

        *batch_shape, T, C, H, W = input.shape
        batch_size = math.prod(batch_shape)
        x = input.reshape(batch_size, T, C, H, W)

        z = self.vae.encode(x.transpose(1, 2), cache=cache).transpose(1, 2)
        return z.reshape(*batch_shape, *z.shape[1:])

    @property
    def temporal_compression_ratio(self) -> int:
        return self.vae.temporal_compression_ratio

    @property
    def spatial_compression_ratio(self) -> int:
        return self.vae.spatial_compression_ratio

    def get_output_temporal_size(
        self, autoregressive_index: int, input_temporal_size: int
    ) -> int:
        """Causal: AR 0 needs an extra (un-grouped) pixel frame for the first latent."""
        r = self.temporal_compression_ratio
        if autoregressive_index == 0:
            assert (input_temporal_size - 1) % r == 0, (
                f"AR 0 input_temporal_size={input_temporal_size} must satisfy "
                f"(N - 1) % temporal_compression_ratio={r} == 0."
            )
            return 1 + (input_temporal_size - 1) // r
        assert input_temporal_size % r == 0, (
            f"AR>=1 input_temporal_size={input_temporal_size} must be divisible "
            f"by temporal_compression_ratio={r}."
        )
        return input_temporal_size // r

    def get_input_temporal_size(
        self, autoregressive_index: int, output_temporal_size: int
    ) -> int:
        r = self.temporal_compression_ratio
        if autoregressive_index == 0:
            return 1 + (output_temporal_size - 1) * r
        return output_temporal_size * r


@dataclass(kw_only=True)
class WanVAEDecoderConfig(DecoderConfig):
    """Config for the Wan VAE decoder.

    Defaults reproduce the Wan 2.1 / 14B 8x-spatial 16-channel
    streaming VAE. Override the architecture knobs (or use
    :class:`Wan22TI2V5BVAEDecoderConfig`) to load the Wan 2.2 TI2V 5B
    decoder, which has asymmetric ``base_dim=160`` / ``decoder_base_dim
    =256`` and the residual up-stage with ``DupUp3D`` shortcut.
    """

    _target: type["WanVAEDecoder"] = field(default_factory=lambda: WanVAEDecoder)

    checkpoint_path: str = AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"]
    dtype: torch.dtype = torch.bfloat16
    use_cuda_graph: bool = True
    """Wrap the decoder forward in a CUDA graph for replay."""

    use_compile: bool = False
    """``torch.compile(mode="max-autotune-no-cudagraphs")``. See
    ``WanVAEEncoderConfig.use_compile`` for the VRAM caveat."""

    # Wan 2.x VAE architecture knobs (default = Wan 2.1). The decoder
    # needs the encoder's ``base_dim`` too because the checkpoint's
    # encoder weights are loaded by the same ``WanVAE`` instance (with
    # ``enable_encoder=False`` the encoder isn't built, but the dim
    # itself controls absolutely nothing on the decoder branch -- we
    # accept it here so :class:`Wan22TI2V5BVAEDecoderConfig` only has
    # to differ in checkpoint + knobs, not in shape).
    base_dim: int = WanVAE.BASE_DIM
    decoder_base_dim: int | None = None
    """Decoder base channel count. ``None`` mirrors ``base_dim``
    (Wan 2.1). Wan 2.2 TI2V 5B uses an asymmetric 256."""
    z_dim: int = WanVAE.Z_DIM
    patch_size: int = WanVAE.PATCH_SIZE
    is_residual: bool = WanVAE.IS_RESIDUAL
    latent_mean: tuple[float, ...] = _WAN21_LATENT_MEAN
    latent_std: tuple[float, ...] = _WAN21_LATENT_STD
    state_dict_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None
    """Optional pre-``load_state_dict`` key remap. See
    :func:`wan22_ti2v_5b_vae_state_dict_transform` for the Wan 2.2 TI2V
    5B diffusers -> flashdreams remap."""


class WanVAEDecoder(StreamingVideoDecoder[WanVAECache]):
    """Wan VAE decoder.

    Forward input is a latent ``[..., Tl, Cl, Hl, Wl]``; output is a video
    tensor ``[..., T, C, H, W]`` in ``[-1, 1]``. The cache is advanced
    in-place across AR decode steps; passing ``cache=None`` allocates a
    fresh single-shot cache.
    """

    TEMPORAL_COMPRESSION_RATIO = WanVAE.TEMPORAL_COMPRESSION_RATIO
    SPATIAL_COMPRESSION_RATIO = WanVAE.SPATIAL_COMPRESSION_RATIO

    def __init__(self, config: WanVAEDecoderConfig) -> None:
        super().__init__(config)
        self.config: WanVAEDecoderConfig = config

        use_lightvae = "lightvae" in config.checkpoint_path
        self.vae = WanVAE(
            vae_path=config.checkpoint_path,
            use_lightvae=use_lightvae,
            use_cuda_graph=config.use_cuda_graph,
            use_compile=config.use_compile,
            enable_encoder=False,
            enable_decoder=True,
            base_dim=config.base_dim,
            decoder_base_dim=config.decoder_base_dim,
            z_dim=config.z_dim,
            patch_size=config.patch_size,
            is_residual=config.is_residual,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            state_dict_transform=config.state_dict_transform,
        ).to(dtype=config.dtype)

    def initialize_autoregressive_cache(self) -> WanVAECache:
        return self.vae.prepare_cache()

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: WanVAECache | None = None,
    ) -> Tensor:
        if cache is None:
            cache = self.initialize_autoregressive_cache()

        assert input.ndim >= 4, "Expected input to have shape [..., T, C, H, W]"

        *batch_shape, T, C, H, W = input.shape
        batch_size = math.prod(batch_shape)
        z = input.reshape(batch_size, T, C, H, W)

        x = self.vae.decode(z.transpose(1, 2), cache=cache).transpose(1, 2)
        return x.reshape(*batch_shape, *x.shape[1:])

    @property
    def temporal_compression_ratio(self) -> int:
        return self.vae.temporal_compression_ratio

    @property
    def spatial_compression_ratio(self) -> int:
        return self.vae.spatial_compression_ratio

    def get_output_temporal_size(
        self, autoregressive_index: int, input_temporal_size: int
    ) -> int:
        """Causal: AR 0 first latent frame decodes to a single pixel frame."""
        r = self.temporal_compression_ratio
        if autoregressive_index == 0:
            return 1 + (input_temporal_size - 1) * r
        return input_temporal_size * r

    def get_input_temporal_size(
        self, autoregressive_index: int, output_temporal_size: int
    ) -> int:
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


## Wan 2.2 TI2V 5B configs


# Diffusers ``AutoencoderKLWan`` (Wan 2.2 5B) -> flashdreams ``WanVAE``
# key remap. The production configs below use upstream's
# ``Wan2.2_VAE.pth`` whose layout matches our model directly (no remap
# needed); this dict + :func:`wan22_ti2v_5b_vae_state_dict_transform`
# are kept in tree as an opt-in fallback for callers who'd rather
# point at the diffusers safetensors shard.
_WAN22_TI2V_5B_VAE_KEY_REMAP: dict[str, str] = {
    # Top-level quant convs.
    r"^quant_conv\.(.*)$": r"conv1.\1",
    r"^post_quant_conv\.(.*)$": r"conv2.\1",
    # Encoder / decoder entry convs.
    r"^encoder\.conv_in\.(.*)$": r"encoder.conv1.\1",
    r"^decoder\.conv_in\.(.*)$": r"decoder.conv1.\1",
    # Encoder / decoder norm_out + conv_out (our head Sequential
    # places RMS_norm at index 0 and the final CausalConv3d at
    # index 2; SiLU sits at index 1 with no params).
    r"^encoder\.norm_out\.(.*)$": r"encoder.head.0.\1",
    r"^encoder\.conv_out\.(.*)$": r"encoder.head.2.\1",
    r"^decoder\.norm_out\.(.*)$": r"decoder.head.0.\1",
    r"^decoder\.conv_out\.(.*)$": r"decoder.head.2.\1",
    # Mid block: diffusers stores ``resnets.{0,1}`` + ``attentions.0``
    # while our Sequential is (Residual, Attention, Residual) ->
    # indices (0, 1, 2). ``WanMidBlock.forward`` runs resnets[0], then
    # (attentions[0], resnets[1]), so the ordering matches our
    # (middle.0, middle.2) layout. Per-field remap mirrors the
    # down/up-block resnets below.
    r"^encoder\.mid_block\.resnets\.0\.norm1\.(.*)$": r"encoder.middle.0.residual.0.\1",
    r"^encoder\.mid_block\.resnets\.0\.conv1\.(.*)$": r"encoder.middle.0.residual.2.\1",
    r"^encoder\.mid_block\.resnets\.0\.norm2\.(.*)$": r"encoder.middle.0.residual.3.\1",
    r"^encoder\.mid_block\.resnets\.0\.conv2\.(.*)$": r"encoder.middle.0.residual.6.\1",
    r"^encoder\.mid_block\.resnets\.0\.conv_shortcut\.(.*)$": r"encoder.middle.0.shortcut.\1",
    r"^encoder\.mid_block\.resnets\.1\.norm1\.(.*)$": r"encoder.middle.2.residual.0.\1",
    r"^encoder\.mid_block\.resnets\.1\.conv1\.(.*)$": r"encoder.middle.2.residual.2.\1",
    r"^encoder\.mid_block\.resnets\.1\.norm2\.(.*)$": r"encoder.middle.2.residual.3.\1",
    r"^encoder\.mid_block\.resnets\.1\.conv2\.(.*)$": r"encoder.middle.2.residual.6.\1",
    r"^encoder\.mid_block\.resnets\.1\.conv_shortcut\.(.*)$": r"encoder.middle.2.shortcut.\1",
    r"^encoder\.mid_block\.attentions\.0\.(.*)$": r"encoder.middle.1.\1",
    r"^decoder\.mid_block\.resnets\.0\.norm1\.(.*)$": r"decoder.middle.0.residual.0.\1",
    r"^decoder\.mid_block\.resnets\.0\.conv1\.(.*)$": r"decoder.middle.0.residual.2.\1",
    r"^decoder\.mid_block\.resnets\.0\.norm2\.(.*)$": r"decoder.middle.0.residual.3.\1",
    r"^decoder\.mid_block\.resnets\.0\.conv2\.(.*)$": r"decoder.middle.0.residual.6.\1",
    r"^decoder\.mid_block\.resnets\.0\.conv_shortcut\.(.*)$": r"decoder.middle.0.shortcut.\1",
    r"^decoder\.mid_block\.resnets\.1\.norm1\.(.*)$": r"decoder.middle.2.residual.0.\1",
    r"^decoder\.mid_block\.resnets\.1\.conv1\.(.*)$": r"decoder.middle.2.residual.2.\1",
    r"^decoder\.mid_block\.resnets\.1\.norm2\.(.*)$": r"decoder.middle.2.residual.3.\1",
    r"^decoder\.mid_block\.resnets\.1\.conv2\.(.*)$": r"decoder.middle.2.residual.6.\1",
    r"^decoder\.mid_block\.resnets\.1\.conv_shortcut\.(.*)$": r"decoder.middle.2.shortcut.\1",
    r"^decoder\.mid_block\.attentions\.0\.(.*)$": r"decoder.middle.1.\1",
    # Per-residual-block field remap: diffusers ``conv1 / conv2 /
    # norm1 / norm2 / conv_shortcut`` -> our Sequential entries.
    # The Sequential layout in ResidualBlock.__init__ is:
    #   0: RMS_norm (norm1)
    #   1: SiLU
    #   2: CausalConv3d (conv1)
    #   3: RMS_norm (norm2)
    #   4: SiLU
    #   5: Dropout
    #   6: CausalConv3d (conv2)
    # Plus ``shortcut`` (renamed from diffusers ``conv_shortcut``).
    r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.norm1\.(.*)$": (
        r"encoder.downsamples.\1.resnets.\2.residual.0.\3"
    ),
    r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.conv1\.(.*)$": (
        r"encoder.downsamples.\1.resnets.\2.residual.2.\3"
    ),
    r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.norm2\.(.*)$": (
        r"encoder.downsamples.\1.resnets.\2.residual.3.\3"
    ),
    r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.conv2\.(.*)$": (
        r"encoder.downsamples.\1.resnets.\2.residual.6.\3"
    ),
    r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.conv_shortcut\.(.*)$": (
        r"encoder.downsamples.\1.resnets.\2.shortcut.\3"
    ),
    # Encoder down-block extras: residual stage shortcut + downsampler.
    r"^encoder\.down_blocks\.(\d+)\.avg_shortcut\.(.*)$": (
        r"encoder.downsamples.\1.avg_shortcut.\2"
    ),
    r"^encoder\.down_blocks\.(\d+)\.downsampler\.(.*)$": (
        r"encoder.downsamples.\1.downsampler.\2"
    ),
    # Decoder up-block residual-conv remap (same Sequential layout).
    r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.norm1\.(.*)$": (
        r"decoder.upsamples.\1.resnets.\2.residual.0.\3"
    ),
    r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.conv1\.(.*)$": (
        r"decoder.upsamples.\1.resnets.\2.residual.2.\3"
    ),
    r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.norm2\.(.*)$": (
        r"decoder.upsamples.\1.resnets.\2.residual.3.\3"
    ),
    r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.conv2\.(.*)$": (
        r"decoder.upsamples.\1.resnets.\2.residual.6.\3"
    ),
    r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.conv_shortcut\.(.*)$": (
        r"decoder.upsamples.\1.resnets.\2.shortcut.\3"
    ),
    r"^decoder\.up_blocks\.(\d+)\.avg_shortcut\.(.*)$": (
        r"decoder.upsamples.\1.avg_shortcut.\2"
    ),
    r"^decoder\.up_blocks\.(\d+)\.upsampler\.(.*)$": (
        r"decoder.upsamples.\1.upsampler.\2"
    ),
}


def wan22_ti2v_5b_vae_state_dict_transform(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Remap a diffusers ``AutoencoderKLWan`` state-dict to ``WanVAE`` keys.

    Applied automatically when :class:`Wan22TI2V5BVAEEncoderConfig` /
    :class:`Wan22TI2V5BVAEDecoderConfig` load the upstream
    ``Wan-AI/Wan2.2-TI2V-5B-Diffusers/vae/diffusion_pytorch_model.safetensors``
    checkpoint. The mapping is purely structural -- no tensors are
    copied or reshaped.

    Note:
        Patterns are applied in iteration order via
        :func:`flashdreams.core.checkpoint.remap.remap_checkpoint_keys`.
        Any key without a matching pattern passes through unchanged,
        which surfaces as a ``load_state_dict`` ``unexpected_keys``
        warning so missing remap entries are easy to spot.
    """
    from flashdreams.core.checkpoint.remap import remap_checkpoint_keys

    return remap_checkpoint_keys(state_dict, _WAN22_TI2V_5B_VAE_KEY_REMAP)


@dataclass(kw_only=True)
class Wan22TI2V5BVAEEncoderConfig(WanVAEEncoderConfig):
    """Pre-rolled config for the Wan 2.2 TI2V 5B encoder.

    Pins the diffusers upstream checkpoint, the 16x-spatial / 48ch /
    residual / patchify architecture knobs, and the matching diffusers
    -> flashdreams key remap. Equivalent to the Wan 2.1 encoder config
    plus the 5B-specific knobs flipped on.
    """

    checkpoint_path: str = WAN22_TI2V_5B_VAE_DIFFUSERS_PATH
    base_dim: int = 160
    z_dim: int = 48
    patch_size: int = 2
    is_residual: bool = True
    latent_mean: tuple[float, ...] = _WAN22_TI2V_5B_LATENT_MEAN
    latent_std: tuple[float, ...] = _WAN22_TI2V_5B_LATENT_STD
    state_dict_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = (
        wan22_ti2v_5b_vae_state_dict_transform
    )


@dataclass(kw_only=True)
class Wan22TI2V5BVAEDecoderConfig(WanVAEDecoderConfig):
    """Pre-rolled config for the Wan 2.2 TI2V 5B decoder.

    Mirrors :class:`Wan22TI2V5BVAEEncoderConfig` but with the asymmetric
    ``decoder_base_dim=256``.
    """

    checkpoint_path: str = WAN22_TI2V_5B_VAE_DIFFUSERS_PATH
    base_dim: int = 160
    decoder_base_dim: int | None = 256
    z_dim: int = 48
    patch_size: int = 2
    is_residual: bool = True
    latent_mean: tuple[float, ...] = _WAN22_TI2V_5B_LATENT_MEAN
    latent_std: tuple[float, ...] = _WAN22_TI2V_5B_LATENT_STD
    state_dict_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = (
        wan22_ti2v_5b_vae_state_dict_transform
    )
