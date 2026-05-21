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

"""LR projector and supporting modules for the FlashVSR encoder.

Hosts :class:`Causal_LQ4x_Proj`: a causal 3D-conv projector with an
optional :class:`CUDAGraphWrapper` around the streaming forward, plus
:class:`PixelShuffle3d` and the :class:`RMS_norm` / :class:`CausalConv3d`
building blocks the projector composes. Wrapped in
:mod:`flashvsr.encoder` for the
:class:`flashdreams.infra.encoder.StreamingEncoder` interface used by
``StreamInferencePipeline``.

:class:`PixelShuffle3d` is also re-used by the TC decoder
(:mod:`flashvsr.decoder.network`) to pack the bicubic conditioning into
latent channels; the decoder needs the ``autopad_first_frame=True`` +
``out_layout="frame_first"`` mode while the projector needs the default
``out_layout="channel_first"`` mode.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import torch
import torch.nn as nn
from einops import rearrange

from flashdreams.infra.cuda_graph import CUDAGraphWrapper, set_or_copy
from flashdreams.recipes.wan.autoencoder.vae import (
    CausalConv3d as _WanCausalConv3d,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    RMS_norm,
)

CACHE_T = 2
"""Number of conv-tail frames retained per streaming cache slot. Matches
the ``CACHE_T`` constant in the shared Wan VAE."""

PixelShuffle3dLayout = Literal["channel_first", "frame_first"]
"""Output layout of :class:`PixelShuffle3d`; see the class docstring."""

CausalConv3d = partial(_WanCausalConv3d, pad_mode="replicate")
"""Replicate-padded variant of the shared Wan VAE ``CausalConv3d``. The
FlashVSR projector pads with replicate (vs the Wan VAE's zero-pad);
``RMS_norm`` is reused as-is because every projector call uses the Wan
VAE's default ``bias=False``."""


class PixelShuffle3d(nn.Module):
    """3D pixel shuffle with optional first-frame replicate-pad.

    ``out_layout``:

    - ``"channel_first"`` (default; projector use): emits
      ``(B, C * ff * hh * ww, F, H, W)`` -- the standard channels-first
      layout consumed by ``Causal_LQ4x_Proj``'s ``Conv3d``.
    - ``"frame_first"`` (decoder use): emits
      ``(B, F, C * ff * hh * ww, H, W)`` -- the per-frame layout the TC
      decoder concatenates onto the latent.

    ``autopad_first_frame``:

    - ``False`` (default; projector use): assumes ``F % ff == 0`` and
      raises an einops error otherwise. The projector aligns ``F`` upstream.
    - ``True`` (decoder use): pad-left-replicates frame 0 to make ``F``
      divisible by ``ff``. Matches the legacy TC decoder behavior on
      cold-start chunks where the cond's frame count isn't a clean
      multiple of 4.
    """

    def __init__(
        self,
        ff: int,
        hh: int,
        ww: int,
        *,
        out_layout: PixelShuffle3dLayout = "channel_first",
        autopad_first_frame: bool = False,
    ):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww
        self.out_layout = out_layout
        self.autopad_first_frame = autopad_first_frame

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.autopad_first_frame and x.shape[2] % self.ff != 0:
            pad = self.ff - x.shape[2] % self.ff
            first_frame = x[:, :, 0:1].repeat(1, 1, pad, 1, 1)
            x = torch.cat([first_frame, x], dim=2)
        if self.out_layout == "channel_first":
            return rearrange(
                x,
                "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w",
                ff=self.ff,
                hh=self.hh,
                ww=self.ww,
            )
        return rearrange(
            x,
            "b c (f ff) (h hh) (w ww) -> b f (c ff hh ww) h w",
            ff=self.ff,
            hh=self.hh,
            ww=self.ww,
        )


@dataclass
class Causal_LQ4x_Proj_Cache:
    """Per-rollout streaming cache for :class:`Causal_LQ4x_Proj`."""

    cache: dict[str, torch.Tensor | None]
    """``{"conv1": ..., "conv2": ...}`` causal-conv tail slots, each of
    shape ``[B, C, CACHE_T, H, W]`` once primed; ``None`` until the first
    forward fills the slot."""


class Causal_LQ4x_Proj(nn.Module):
    """Causal projector with optional CUDA-graph capture of the streaming forward.

    With ``use_cuda_graph=True`` the cache-fill chunk runs through
    ``CUDAGraphWrapper.drain`` (the ``cache_x is None`` branch in
    ``CausalConv3d.forward`` makes the filling and steady traces structurally
    different — capturing the filling trace would bake the wrong control flow).
    Subsequent same-shape chunks go through ``wrapper.__call__`` for
    warmup -> capture -> replay. Slot-pointer tracking auto-resets the wrapper
    when the bound cache identity changes (new external cache, in-place dict
    rebind, etc.) so callers never need to call ``wrapper.reset()`` directly.
    Mirrors ``flashdreams.recipes.taehv.impl.TAEHV``'s in-module wrapper.
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1536,
        layer_num: int = 30,
        use_cuda_graph: bool = False,
        use_compile: bool = False,
        compile_mode: str = "max-autotune-no-cudagraphs",
        compile_dynamic: bool | None = None,
        warmup_iters: int = 1,
    ):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(
            in_dim * self.ff * self.hh * self.ww,
            self.hidden_dim1,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(
            self.hidden_dim1,
            self.hidden_dim2,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)]
        )

        # ``torch.compile`` rebind happens *before* wrapper construction so the
        # wrapper captures the compiled callable. ``CUDAGraphWrapper.drain``
        # then covers Inductor's lazy autotune on the first wrapper call (see
        # ``flashdreams/infra/cuda_graph.py:174``). Default mode is
        # ``max-autotune-no-cudagraphs`` to avoid double-stacking CUDA graphs
        # against the wrapper's own capture; ``compile_mode`` /
        # ``compile_dynamic`` exist so tests can opt into cheaper Inductor
        # settings (e.g. ``mode="default"`` to skip the multi-second autotune)
        # without bypassing the production constructor path.
        #
        # The compiled callable lives on a separate ``_stream_forward``
        # attribute -- rebinding ``self.stream_forward_efficient`` directly
        # would shadow the method declaration on the class and confuse type
        # checkers (the rebound callable's signature drops ``self``).
        self._use_compile = use_compile
        self._stream_forward: Callable[
            [torch.Tensor, Causal_LQ4x_Proj_Cache], list[torch.Tensor]
        ] = (
            torch.compile(
                self.stream_forward_efficient,
                mode=compile_mode,
                dynamic=compile_dynamic,
            )
            if use_compile
            else self.stream_forward_efficient
        )

        self._use_cuda_graph = use_cuda_graph
        self._proj_wrapper: CUDAGraphWrapper | None = (
            CUDAGraphWrapper(self._stream_forward, warmup_iters=warmup_iters)
            if use_cuda_graph
            else None
        )
        # Tracks the Python ``id()`` of the bound cache's two slot tensors. Any
        # change (new cache, replaced dict, freshly cleared slots) forces a
        # ``wrapper.reset()`` because captured kernels reference the prior
        # storage pointers. Sentinel value before the first call.
        self._wrapper_slot_id: tuple[int, int] = (id(None), id(None))

    def stream_forward_efficient(
        self,
        video: torch.Tensor,
        cache: Causal_LQ4x_Proj_Cache,
    ):
        if video.shape[2] % 4 != 0:
            n_left_padding = 4 - (video.shape[2] % 4)
            first_frame = video[:, :, :1, :, :].repeat(1, 1, n_left_padding, 1, 1)
            video = torch.cat([first_frame, video], dim=2)

        x = self.pixel_shuffle(video)
        # Snapshot the conv1 streaming tail before running the conv. The view
        # into ``x`` stays valid until ``set_or_copy`` reads it; the
        # ``copy_`` keeps the slot's storage pointer fixed for graph replay.
        new_tail1 = x[:, :, -CACHE_T:, :, :]
        x = self.conv1(x, cache.cache["conv1"])
        set_or_copy(cache.cache, "conv1", new_tail1)
        x = self.norm1(x)
        x = self.act1(x)
        new_tail2 = x[:, :, -CACHE_T:, :, :]
        x = self.conv2(x, cache.cache["conv2"])
        set_or_copy(cache.cache, "conv2", new_tail2)
        x = self.norm2(x)
        x = self.act2(x)

        out_x = rearrange(x, "b c f h w -> b (f h w) c")
        return [layer(out_x) for layer in self.linear_layers]

    def forward_streaming(
        self, video: torch.Tensor, cache: Causal_LQ4x_Proj_Cache
    ) -> list[torch.Tensor]:
        """Dispatch the streaming projector forward against the optional CUDA graph.

        Routing matches the TAEHV decode pattern (``flashdreams/recipes/taehv/impl.py``):

        - Filling phase (either slot is ``None``) routes to ``wrapper.drain``,
          which stages inputs into the static buffers but runs the callable
          eagerly. The ``cache_x is None`` branch in ``CausalConv3d.forward``
          differs from the steady-state branch, so capturing the filling
          trace would bake in the wrong control flow.
        - Steady phase routes to ``wrapper.__call__``, which warms up,
          captures, and replays.

        Slot-pointer tracking detects any change to the bound cache's slot
        identities (new external cache, in-place dict rebind, freshly cleared
        slots) and invalidates the captured graph automatically.
        """
        if self._proj_wrapper is None:
            # ``self._stream_forward`` is the (optionally torch.compile'd)
            # bound version of ``stream_forward_efficient``; see ``__init__``.
            return self._stream_forward(video, cache)

        slot_id = (id(cache.cache.get("conv1")), id(cache.cache.get("conv2")))
        if slot_id != self._wrapper_slot_id:
            self._proj_wrapper.reset()

        is_filling = cache.cache["conv1"] is None or cache.cache["conv2"] is None
        call = self._proj_wrapper.drain if is_filling else self._proj_wrapper
        out = call(video, cache)
        # Re-read after the call so first-write allocations (drain phase) and
        # any future shape-driven reallocs are picked up on the next dispatch.
        self._wrapper_slot_id = (
            id(cache.cache["conv1"]),
            id(cache.cache["conv2"]),
        )
        return out

    def create_external_cache(self) -> Causal_LQ4x_Proj_Cache:
        return Causal_LQ4x_Proj_Cache(cache={"conv1": None, "conv2": None})
