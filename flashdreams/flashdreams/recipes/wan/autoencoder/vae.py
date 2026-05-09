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

"""Wan 2.1 VAE: streaming causal encode / decode."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.compile import compile_module
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
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

AVAILABLE_WAN_VAE_CHECKPOINT_PATHS = {
    "lightvae": "s3://flashdreams/assets/checkpoints/autoencoders/lightvaew2_1.pth",
    "vae": "s3://flashdreams/assets/checkpoints/autoencoders/Wan2.1_VAE.pth",
}

CACHE_T = 2
TEMPORAL_WINDOW = 4


def _set_or_copy(
    state: Dict[int, torch.Tensor], key: int, new_value: torch.Tensor
) -> None:
    """Write ``new_value`` into ``state[key]``, preserving the storage pointer.

    In-place ``copy_`` once the slot exists at the matching shape, else
    allocate a fresh clone. Pointer stability is required for CUDA-graph
    capture, since captured kernels reference the slot's storage address.
    """
    cur = state.get(key)
    if cur is not None and cur.shape == new_value.shape:
        cur.copy_(new_value)
    else:
        state[key] = new_value.clone()


_LATENT_MEAN = (
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
_LATENT_STD = (
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


class CausalConv3d(nn.Conv3d):
    """3D conv with causal time padding and a streaming left-context slot."""

    # Concrete attribute types so callers don't see ``Tensor | Module``.
    _spatial_pad: tuple[int, int, int, int]
    _has_spatial_pad: bool
    _time_pad: int

    def __init__(self, *args, **kwargs):
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
        self.padding = (0, 0, 0)

    def forward(
        self, x: torch.Tensor, prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        time_pad = self._time_pad
        if prev is not None and time_pad > 0:
            x = torch.cat([prev, x], dim=2)
            time_pad = max(0, time_pad - prev.shape[2])
        if time_pad or self._has_spatial_pad:
            x = F.pad(x, (*self._spatial_pad, time_pad, 0))
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
        _set_or_copy(state, key, new_tail)
        return out


class RMS_norm(nn.Module):
    """RMS-normalisation with a learnable channel scale (no bias)."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True):
        super().__init__()
        broadcast = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcast) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma


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
            _set_or_copy(state, key, x[:, :, -CACHE_T:])
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
        _set_or_copy(state, key, new_tail)
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
    ):
        super().__init__()
        dims = [int(dim * u * (1 - pruning_rate)) for u in (1,) + tuple(dim_mult)]
        scale = 1.0

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        downsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
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

        upsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Match legacy weight shapes: stages 1-3 halve their input dim
            # because the preceding ``Resample`` already halved channels.
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, state: Dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.conv1.cache_step(x, state)
        for layer in self.middle:
            x = layer(x, state)
        for layer in self.upsamples:
            x = layer(x, state)
        norm, act, conv = self.head
        # ``nn.Sequential`` typing hands back ``Module``; the final entry
        # is a ``CausalConv3d`` and we need its non-``Module`` ``cache_step``.
        assert isinstance(conv, CausalConv3d)
        return conv.cache_step(act(norm(x)), state)


class WanVAE(nn.Module):
    """Wan 2.x video VAE: streaming causal encode and decode.

    Input video shape is ``[B, 3, T, H, W]`` in ``[-1, 1]``; the latent shape
    is ``[B, 16, Tl, H/8, W/8]``. Each rollout uses a fresh ``WanVAECache``.

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

    Z_DIM = 16

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
    ):
        super().__init__()
        assert enable_encoder or enable_decoder, (
            "WanVAE: at least one of enable_encoder / enable_decoder must be True"
        )

        pruning_rate = 0.75 if use_lightvae else 0.0

        # TypedDict so ``**common`` unpacks with concrete kwarg types
        # rather than the ``object``-typed values an untyped ``dict``
        # literal would yield.
        class _CommonKwargs(TypedDict):
            dim: int
            dim_mult: tuple[int, ...]
            num_res_blocks: int
            attn_scales: tuple[float, ...]
            dropout: float
            pruning_rate: float

        common: _CommonKwargs = {
            "dim": 96,
            "dim_mult": (1, 2, 4, 4),
            "num_res_blocks": 2,
            "attn_scales": (),
            "dropout": 0.0,
            "pruning_rate": pruning_rate,
        }
        # Build on ``meta`` so only the checkpoint allocates real memory; skip
        # the disabled half so its params never materialise.
        with torch.device("meta"):
            if enable_encoder:
                self.encoder = Encoder3d(
                    z_dim=self.Z_DIM * 2,
                    temperal_downsample=(False, True, True),
                    **common,
                )
                self.conv1 = CausalConv3d(self.Z_DIM * 2, self.Z_DIM * 2, 1)
            if enable_decoder:
                self.decoder = Decoder3d(
                    z_dim=self.Z_DIM,
                    temperal_upsample=(True, True, False),
                    **common,
                )
                self.conv2 = CausalConv3d(self.Z_DIM, self.Z_DIM, 1)

        # assign=True: meta params become the checkpoint tensors directly;
        # caller does .to(device, dtype) afterward. strict=False tolerates
        # the disabled half (encoder-only or decoder-only).
        self.load_state_dict(load_checkpoint(vae_path), strict=False, assign=True)

        self.register_buffer(
            "mean",
            torch.tensor(_LATENT_MEAN).view(1, self.Z_DIM, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "inv_std",
            (1.0 / torch.tensor(_LATENT_STD)).view(1, self.Z_DIM, 1, 1, 1),
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
        """
        assert self._enable_encoder, (
            "WanVAE.encode called but the model was constructed with "
            "enable_encoder=False"
        )
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
        """
        assert self._enable_decoder, (
            "WanVAE.decode called but the model was constructed with "
            "enable_decoder=False"
        )
        z = z / self.inv_std + self.mean
        # See encode() for the rollout 1 vs 2+ dispatch rationale.
        if self._use_cuda_graph:
            assert self._decoder_wrapper is not None
            decoder = (
                self._decoder_wrapper.drain
                if not cache.dec_state
                else self._decoder_call
            )
        else:
            decoder = self.decoder
        return decoder(self.conv2(z), cache.dec_state).clamp_(-1, 1)

    @property
    def temporal_compression_ratio(self) -> int:
        return self.TEMPORAL_COMPRESSION_RATIO

    @property
    def spatial_compression_ratio(self) -> int:
        return self.SPATIAL_COMPRESSION_RATIO


@dataclass(kw_only=True)
class WanVAEEncoderConfig(EncoderConfig):
    """Config for the Wan VAE encoder."""

    _target: type = field(default_factory=lambda: WanVAEEncoder)

    checkpoint_path: str = AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"]
    dtype: torch.dtype = torch.bfloat16
    use_cuda_graph: bool = True
    """Wrap the encoder forward in a CUDA graph for replay."""

    use_compile: bool = False
    """``torch.compile(mode="max-autotune-no-cudagraphs")``. Off by default:
    Inductor autotune workspaces can add several GiB of transient VRAM per
    unique input shape, surfacing as 'illegal memory access' on smaller GPUs
    with the full-channel ``vae`` checkpoint."""


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
    """Config for the Wan VAE decoder."""

    _target: type = field(default_factory=lambda: WanVAEDecoder)

    checkpoint_path: str = AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"]
    dtype: torch.dtype = torch.bfloat16
    use_cuda_graph: bool = True
    """Wrap the decoder forward in a CUDA graph for replay."""

    use_compile: bool = False
    """``torch.compile(mode="max-autotune-no-cudagraphs")``. See
    ``WanVAEEncoderConfig.use_compile`` for the VRAM caveat."""


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
