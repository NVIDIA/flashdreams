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

"""Tiny AutoEncoder for Hunyuan Video (TAEHV) -- streaming causal decode.

Decode-only slim port: the TAEHV encoder side is unused in our pipelines and
not included here. Encoder weights present in the checkpoint are dropped via
``strict=False`` at load time.

Per-rollout dispatch (when ``use_cuda_graph=True``):
    - Rollout 1 (cache empty): bare module / wrapper.drain -- runs the
      decoder eagerly through the wrapper's static buffer so any
      Inductor / triton autotunes happen on the eager path (illegal
      during graph capture).
    - Rollout 2+ (cache populated): wrapper.__call__ -- ``warmup_iters``
      eager warmups, then capture, then pure replay for every
      same-shape body chunk.

Example::

    decoder = TAEHV(checkpoint_path="...").to(device, torch.bfloat16)
    cache = decoder.prepare_cache()
    x_first = decoder.decode(z_first, cache=cache)   # 5 frames (trimmed)
    x_body  = decoder.decode(z_body,  cache=cache)   # 8 frames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.decoder import DecoderAutoregressiveCache


@dataclass
class TAEHVCache(DecoderAutoregressiveCache):
    """Streaming decoder cache; one slot per ``MemBlock`` keyed by ``id(module)``.

    Slots hold a single ``[B, 1, C, H, W]`` frame -- the last input frame
    of the previous chunk -- which becomes the rolled-in left context
    for the next chunk's MemBlock. Slots have stable storage addresses
    after the first chunk so CUDA-graph replay can write through them
    in place.
    """

    dec_state: Dict[int, torch.Tensor] = field(default_factory=dict)


def _set_or_copy(
    state: Dict[int, torch.Tensor], key: int, new_value: torch.Tensor
) -> None:
    """Write ``new_value`` into ``state[key]``: in-place ``copy_`` once the
    slot exists at the matching shape, else allocate a fresh clone.

    Pointer stability is required for CUDA-graph capture (kernels
    reference the slot's storage address).
    """
    cur = state.get(key)
    if cur is not None and cur.shape == new_value.shape:
        cur.copy_(new_value)
    else:
        state[key] = new_value.clone()


def _conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    """Soft saturating clamp -- ``tanh(x/3) * 3`` (used at decoder input)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Residual block with a 1-frame temporal-left memory slot.

    The forward concatenates ``x`` with a 1-step time-shifted copy
    (``past``) along channels, runs a 3-conv stack, and adds the ``skip``
    projection. :meth:`cache_step` advances the per-instance slot:
    snapshot the last input frame -> next call uses it as ``past``.
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

        Builds ``past`` by rolling ``x`` right one step, padded with the
        previous chunk's last frame (or zeros on the first chunk). On
        steady-state calls only the cached single-frame slot is read,
        matching the legacy ``cache_mem[i] = _x; ... prev_mem[:, -1:]``
        access pattern bit-for-bit.
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
    """Temporal upsample by ``stride``: 1x1 conv expands channels, then
    reshape splits the new channel chunks into consecutive timesteps.

    Stateless across streaming chunks (each chunk's frames are upsampled
    independently)."""

    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _NT, C, H, W = x.shape
        return self.conv(x).reshape(-1, C, H, W)


class Decoder(nn.Module):
    """TAEHV decoder body, owned by :class:`TAEHV`.

    Input: ``[B, T, C_z, H, W]`` latent (T == ``z`` time dim).
    Output: ``[B, T_out, C_img * patch**2, H_out, W_out]`` raw frames
    (no clamp / pixel-shuffle / trim -- those happen in
    :meth:`TAEHV.decode`).
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
        # Layer indices match the legacy ``self.decoder = nn.Sequential(...)``
        # so checkpoint keys (``decoder.<idx>.<param>``) load unchanged.
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
    """Truncate over-sized TGrow ``conv.weight`` rows in ``sd`` to the
    model's expected output channels.

    Some shipped checkpoints store TGrow weights for ``stride=2`` even
    when the model is configured with ``stride=1``; keep only the
    last-timestep slice (matches legacy ``patch_tgrow_layers``).
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
    """Tiny AutoEncoder for Hunyuan Video / Wan -- streaming decode-only.

    Loads a TAEHV checkpoint (encoder weights in the file are silently
    dropped). ``decode`` accepts an ``[N, T, C_z, H, W]`` latent and
    returns an ``[N, T_out, C_img, H*patch*scale, W*patch*scale]``
    image tensor in ``[0, 1]``. Each rollout uses a fresh
    :class:`TAEHVCache`.

    Supported ``model_type`` values: ``"wan21"`` (default; ReLU,
    ``patch_size=1``, ``latent_channels=16``) and ``"wan22"`` (ReLU,
    ``patch_size=2``, ``latent_channels=48``). The legacy ``"hy15"``
    (LeakyReLU + ``[-1, 1]`` clamp) and ``"taecvx"`` (cogvideox even-T
    skip-trim) variants are NOT supported here -- pass them and the
    constructor raises.

    Per-rollout dispatch when ``use_cuda_graph=True``:
        - Rollout 1: bare decoder / wrapper.drain -- drains Inductor
          autotune on the eager path against the wrapper's static
          buffer.
        - Rollout 2+: wrapper.__call__ -- 2 warmups + 1 capture, then
          pure replays for every same-shape body chunk.

    Example::

        taehv = TAEHV(checkpoint_path="...").to("cuda", torch.bfloat16)
        cache = taehv.prepare_cache()
        x = taehv.decode(z, cache=cache)

    Note:
        Set ``torch.backends.cudnn.benchmark = True`` once at process
        start for ~5% extra on the eager seed/tail chunks.
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
            # CogVideoX checkpoints relied on a legacy ``skip_trim`` branch
            # (``is_cogvideox and x.shape[1] % 2 == 0``) that is not ported.
            raise ValueError(
                f"TAEHV: cogvideox checkpoint {checkpoint_path!r} is not "
                f"supported by this slim impl."
            )
        # ``wan22`` uses a different patch-size / latent-channel config; the
        # other supported model types share the defaults above.
        if model_type == "wan22":
            patch_size, latent_channels = 2, 48
        act_func = nn.ReLU(inplace=True)

        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        self.model_type = model_type
        # Frames the decoder must drop from the front of its FIRST chunk
        # output. ``2 ** sum(time_upscale) - 1`` matches legacy.
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1

        n_f = (256, 128, 64, 64)
        # Build on `meta` -- only the checkpoint allocates real memory.
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
        # Re-key from legacy ``decoder.<i>.*`` to ``decoder.blocks.<i>.*``
        # (the new ``Decoder`` wraps the Sequential in an attribute).
        sd = {
            (
                k.replace("decoder.", "decoder.blocks.", 1)
                if k.startswith("decoder.") and not k.startswith("decoder.blocks.")
                else k
            ): v
            for k, v in sd.items()  # ty:ignore[call-non-callable]
        }
        sd = _patch_tgrow_state_dict(sd, self.decoder.blocks)
        # ``assign=True``: meta params become the checkpoint tensors as-is;
        # ``strict=False``: silently drop encoder-only weights.
        self.load_state_dict(sd, strict=False, assign=True)

        self.eval().requires_grad_(False)

        self._use_cuda_graph = use_cuda_graph

        if use_compile:
            self.decoder = torch.compile(  # type: ignore[assignment]
                self.decoder, mode="max-autotune-no-cudagraphs"
            )
        self._decoder_call: Callable[..., torch.Tensor] = (
            CUDAGraphWrapper(self.decoder, warmup_iters=warmup_iters)
            if use_cuda_graph
            else self.decoder
        )

    def prepare_cache(self) -> TAEHVCache:
        """Return a fresh empty cache and drop any captured CUDA graph.

        Captured kernels reference the previous cache's slot pointers,
        which a new cache invalidates -- warmup + capture re-run on the
        next decode of each shape.
        """
        if self._use_cuda_graph:
            self._decoder_call.reset()  # type: ignore[union-attr]  # ty:ignore[unresolved-attribute]
        return TAEHVCache()

    @torch.inference_mode()
    def decode(
        self, z: torch.Tensor, cache: Optional[TAEHVCache] = None
    ) -> torch.Tensor:
        """Streaming decode of ``z`` (``[N, T, C_z, H, W]`` latent).

        First call (``cache.dec_state`` empty): runs the decoder eagerly
        and trims the leading ``frames_to_trim`` frames; subsequent
        same-shape calls go through the captured graph.
        """
        if cache is None:
            cache = self.prepare_cache()
        state = cache.dec_state
        first_decode = not state
        # Bind decoder before the first call so steady-state calls go through
        # the wrapper while the autotune-during-capture shape is the eager
        # path.
        if self._use_cuda_graph:
            decoder = (
                self._decoder_call.drain  # type: ignore[union-attr]  # ty:ignore[unresolved-attribute]
                if first_decode
                else self._decoder_call
            )
        else:
            decoder = self.decoder

        b = z.shape[0]
        x = decoder(z, state, b)
        # Clamp / pixel-shuffle / trim happen outside the captured region;
        # the wrapper returns ``static_output.clone()`` so ``clamp_`` is
        # safe in-place.
        x = x.clamp_(0, 1)
        if self.patch_size > 1:
            n, t, c, h, w = x.shape
            x = F.pixel_shuffle(x.reshape(n * t, c, h, w), self.patch_size)
            x = x.reshape(n, t, x.shape[1], x.shape[2], x.shape[3])
        if first_decode:
            x = x[:, self.frames_to_trim :]
        return x
