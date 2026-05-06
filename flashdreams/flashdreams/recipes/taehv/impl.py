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

"""Streaming causal decoder for TAEHV (Tiny AutoEncoder for Hunyuan Video)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.compile import compile_module
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.decoder import StreamingDecoderCache


@dataclass
class TAEHVCache(StreamingDecoderCache):
    """Streaming decoder cache; one slot per ``MemBlock`` keyed by ``id(module)``.

    Each slot holds the last input frame of the previous chunk, used as
    rolled-in left context. Slot storage addresses are stable after the
    first chunk so CUDA-graph replay can write through them in place.
    """

    dec_state: Dict[int, torch.Tensor] = field(default_factory=dict)


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


def _conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    """Soft saturating clamp ``tanh(x/3) * 3``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Residual block with a 1-frame temporal-left memory slot.

    Concatenates ``x`` with a 1-step time-shifted copy along channels, runs
    a 3-conv stack, and adds the skip projection. ``cache_step`` snapshots
    the last input frame so the next call sees it as ``past``.
    """

    def __init__(self, n_in: int, n_out: int, act_func: nn.Module):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out),
            act_func,
            _conv(n_out, n_out),
            act_func,
            _conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.act = act_func

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

    def cache_step(
        self, x: torch.Tensor, state: Dict[int, torch.Tensor], batch: int
    ) -> torch.Tensor:
        """Apply with streaming left-context drawn from ``state``.

        Rolls ``x`` right one step and pads with the previous chunk's last
        frame (or zeros on the first chunk). Bit-for-bit compatible with the
        legacy ``cache_mem[i] = _x; ... prev_mem[:, -1:]`` pattern.
        """
        key = id(self)
        bt, c, h, w = x.shape
        t = bt // batch
        x5 = x.view(batch, t, c, h, w)
        prev = state.get(key)
        if prev is None:
            past = F.pad(x5, (0, 0, 0, 0, 0, 0, 1, 0))[:, :t]
        else:
            past = torch.cat([prev, x5[:, :-1]], dim=1)
        out = self.forward(x, past.reshape(bt, c, h, w))
        _set_or_copy(state, key, x5[:, -1:])
        return out


class TGrow(nn.Module):
    """Temporal upsample by ``stride`` (channel-expand + reshape; stateless)."""

    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _NT, C, H, W = x.shape
        return self.conv(x).reshape(-1, C, H, W)


class Decoder(nn.Module):
    """TAEHV decoder body.

    Input: ``[B, T, C_z, H, W]`` latent. Output: raw frames
    ``[B, T_out, C_img * patch**2, H_out, W_out]`` (clamp / pixel-shuffle /
    trim happen in ``TAEHV.decode``).
    """

    def __init__(
        self,
        n_f: tuple[int, int, int, int],
        latent_channels: int,
        image_channels: int,
        patch_size: int,
        decoder_time_upscale: tuple[bool, bool],
        decoder_space_upscale: tuple[bool, bool, bool],
        act_func: nn.Module,
    ):
        super().__init__()
        # Layer indices must match the legacy nn.Sequential so checkpoint
        # keys (``decoder.<idx>.<param>``) load unchanged.
        self.blocks = nn.Sequential(
            Clamp(),
            _conv(latent_channels, n_f[0]),
            act_func,
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            _conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            _conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            _conv(n_f[2], n_f[3], bias=False),
            act_func,
            _conv(n_f[3], image_channels * patch_size**2),
        )

    def forward(
        self, z: torch.Tensor, state: Dict[int, torch.Tensor], batch: int
    ) -> torch.Tensor:
        b, t, c, h, w = z.shape
        x = z.reshape(b * t, c, h, w)
        for blk in self.blocks:
            if isinstance(blk, MemBlock):
                x = blk.cache_step(x, state, batch)
            else:
                x = blk(x)
        bt, c_out, h_out, w_out = x.shape
        return x.reshape(b, bt // b, c_out, h_out, w_out)


def _patch_tgrow_state_dict(
    sd: Dict[str, torch.Tensor], decoder_blocks: nn.Sequential
) -> Dict[str, torch.Tensor]:
    """Truncate over-sized TGrow weights in ``sd`` to the model's expected channels.

    Some shipped checkpoints store TGrow weights for stride=2 even when the
    model is configured stride=1; keep only the last-timestep slice.
    """
    sd = dict(sd)
    for i, layer in enumerate(decoder_blocks):
        if isinstance(layer, TGrow):
            key = f"decoder.blocks.{i}.conv.weight"
            if key in sd:
                expected = layer.conv.weight.shape[0]
                if sd[key].shape[0] > expected:
                    sd[key] = sd[key][-expected:]
    return sd


class TAEHV(nn.Module):
    """TAEHV streaming decode-only network.

    Loads a TAEHV checkpoint and exposes ``decode``. Encoder weights in the
    checkpoint are silently dropped. With ``use_cuda_graph=True``, rollout 1
    drains Inductor autotune on the eager path; rollout 2 warms up and
    captures, after which same-shape body chunks replay.

    Supported ``model_type``: ``"wan21"`` (default; ReLU, patch_size=1,
    latent_channels=16) and ``"wan22"`` (ReLU, patch_size=2,
    latent_channels=48). The legacy ``"hy15"`` and ``"taecvx"`` variants are
    not ported.

    Examples:

        taehv = TAEHV(checkpoint_path="...").to("cuda", torch.bfloat16)
        cache = taehv.prepare_cache()
        x = taehv.decode(z, cache=cache)

    Set ``torch.backends.cudnn.benchmark = True`` at process start for ~5%
    extra on the eager seed/tail chunks.
    """

    TEMPORAL_COMPRESSION_RATIO = 4
    SPATIAL_COMPRESSION_RATIO = 8

    SUPPORTED_MODEL_TYPES = ("wan21", "wan22")

    def __init__(
        self,
        checkpoint_path: str = "taew2_1.pth",
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
        patch_size: int = 1,
        latent_channels: int = 16,
        model_type: str = "wan21",
        use_cuda_graph: bool = True,
        use_compile: bool = False,
        warmup_iters: int = 2,
    ):
        super().__init__()
        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"TAEHV: model_type={model_type!r} is not supported by this slim "
                f"impl (supported: {self.SUPPORTED_MODEL_TYPES}). The legacy "
                f"'hy15' / 'taecvx' branches (different activation, clamp range, "
                f"or trim semantics) were dropped in the decode-only refactor."
            )
        if checkpoint_path is not None and "taecvx" in checkpoint_path:
            raise ValueError(
                f"TAEHV: cogvideox checkpoint {checkpoint_path!r} is not "
                f"supported by this slim impl."
            )
        if model_type == "wan22":
            patch_size, latent_channels = 2, 48
        act_func = nn.ReLU(inplace=True)

        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        self.model_type = model_type
        # Frames the decoder drops from the front of its first chunk output
        # (matches the legacy 2 ** sum(time_upscale) - 1 formula).
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1

        n_f = (256, 128, 64, 64)
        # Build on meta so only the checkpoint allocates real memory.
        with torch.device("meta"):
            self.decoder = Decoder(
                n_f=n_f,
                latent_channels=latent_channels,
                image_channels=self.image_channels,
                patch_size=patch_size,
                decoder_time_upscale=decoder_time_upscale,
                decoder_space_upscale=decoder_space_upscale,
                act_func=act_func,
            )

        sd = load_checkpoint(checkpoint_path)
        # Re-key legacy ``decoder.<i>.*`` to ``decoder.blocks.<i>.*`` because
        # the new Decoder wraps the Sequential in an attribute.
        sd = {
            (
                k.replace("decoder.", "decoder.blocks.", 1)
                if k.startswith("decoder.") and not k.startswith("decoder.blocks.")
                else k
            ): v
            for k, v in sd.items()
        }
        sd = _patch_tgrow_state_dict(sd, self.decoder.blocks)
        # assign=True: meta params become the checkpoint tensors directly;
        # strict=False: silently drop encoder-only weights.
        self.load_state_dict(sd, strict=False, assign=True)

        self.eval().requires_grad_(False)

        self._use_cuda_graph = use_cuda_graph

        if use_compile:
            self.decoder = compile_module(self.decoder)
        self._decoder_wrapper: CUDAGraphWrapper | None = (
            CUDAGraphWrapper(self.decoder, warmup_iters=warmup_iters)
            if use_cuda_graph
            else None
        )
        self._decoder_call: Callable[..., torch.Tensor] = (
            self._decoder_wrapper if self._decoder_wrapper is not None else self.decoder
        )

    def prepare_cache(self) -> TAEHVCache:
        """Return a fresh empty cache and drop any captured CUDA graph.

        Captured kernels reference the previous cache's slot pointers, so a
        new cache forces a re-warmup on the next decode.
        """
        if self._use_cuda_graph:
            assert self._decoder_wrapper is not None
            self._decoder_wrapper.reset()
        return TAEHVCache()

    @torch.inference_mode()
    def decode(
        self, z: torch.Tensor, cache: Optional[TAEHVCache] = None
    ) -> torch.Tensor:
        """Streaming decode of an ``[N, T, C_z, H, W]`` latent.

        First call (cache empty) runs the decoder eagerly and trims the
        leading ``frames_to_trim`` frames; same-shape body chunks replay
        the captured graph thereafter.
        """
        if cache is None:
            cache = self.prepare_cache()
        state = cache.dec_state
        first_decode = not state
        # Bind decoder before the first call so steady-state goes through the
        # wrapper while the autotune-during-capture shape stays on the eager
        # path.
        if self._use_cuda_graph:
            assert self._decoder_wrapper is not None
            decoder = (
                self._decoder_wrapper.drain if first_decode else self._decoder_call
            )
        else:
            decoder = self.decoder

        b = z.shape[0]
        x = decoder(z, state, b)
        # Clamp / pixel-shuffle / trim happen outside the captured region;
        # the wrapper returns static_output.clone() so clamp_ is safe in-place.
        x = x.clamp_(0, 1)
        if self.patch_size > 1:
            n, t, c, h, w = x.shape
            x = F.pixel_shuffle(x.reshape(n * t, c, h, w), self.patch_size)
            x = x.reshape(n, t, x.shape[1], x.shape[2], x.shape[3])
        if first_decode:
            x = x[:, self.frames_to_trim :]
        return x
