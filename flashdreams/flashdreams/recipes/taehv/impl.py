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
from flashdreams.infra.cuda_graph import CUDAGraphWrapper, set_or_copy
from flashdreams.infra.decoder import StreamingDecoderCache
from flashdreams.recipes.taehv.checkpoint import (
    StateDictTransform,
    compose,
    legacy_to_blocks_keys,
    truncate_oversize_tgrow_weights_from_blocks,
)


@dataclass
class TAEHVCache(StreamingDecoderCache):
    """Streaming decoder cache; one slot per ``MemBlock`` keyed by ``id(module)``.

    Each slot holds the last input frame of the previous chunk, used as
    rolled-in left context. Slot storage addresses are stable after the
    first chunk so CUDA-graph replay can write through them in place.
    """

    dec_state: Dict[int, torch.Tensor] = field(default_factory=dict)


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
        """Apply with streaming left-context: prepend the previous chunk's
        last frame to ``x``, run the conv stack, save the new last frame.

        ``state[id(self)]`` must already exist; :meth:`Decoder.initialize_state`
        pre-allocates it as a ``[batch, 1, C, H, W]`` zero tensor on the
        first chunk (bit-equivalent to the legacy ``F.pad`` zero-pad path).
        Keeping the dict-state lookup branchless lets the compiled decoder
        be one ``torch.compile``-stable graph across the AR0 -> AR1 transition.
        """
        key = id(self)
        bt, c, h, w = x.shape
        t = bt // batch
        x5 = x.view(batch, t, c, h, w)
        prev = state[key]
        past = torch.cat([prev, x5[:, :-1]], dim=1)
        out = self.forward(x, past.reshape(bt, c, h, w))
        set_or_copy(state, key, x5[:, -1:])
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

    @torch.no_grad()
    def initialize_state(
        self,
        z_shape: tuple[int, int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        state: Dict[int, torch.Tensor],
    ) -> None:
        """Populate ``state`` with one zero ``[B, 1, C_i, H_i, W_i]`` tensor per
        :class:`MemBlock`.

        Walks ``self.blocks`` once eagerly with a synthetic zero input to
        derive each ``MemBlock``'s input shape; the conv outputs are
        discarded. After this call every ``MemBlock``'s ``id()`` key is in
        ``state`` so :meth:`MemBlock.cache_step` never raises and Dynamo's
        first compiled trace already sees the populated branch (no AR0 -> AR1
        recompile).

        Zero ``prev`` makes ``cat([zeros, x5[:, :-1]], dim=1)`` in
        ``cache_step`` bit-equivalent to the legacy
        ``F.pad(x5, (.., 1, 0))[:, :t]`` first-chunk path, so AR0 outputs
        are unchanged.
        """
        batch, t, c_z, h_z, w_z = z_shape
        x = torch.zeros(batch * t, c_z, h_z, w_z, dtype=dtype, device=device)
        for blk in self.blocks:
            if isinstance(blk, MemBlock):
                bt_x, c_x, h_x, w_x = x.shape
                t_x = bt_x // batch
                state[id(blk)] = torch.zeros(
                    batch, 1, c_x, h_x, w_x, dtype=dtype, device=device
                )
                # Advance x's shape by running the block with a zero past;
                # output values are discarded.
                past = (
                    state[id(blk)]
                    .expand(-1, t_x, -1, -1, -1)
                    .reshape(bt_x, c_x, h_x, w_x)
                )
                x = blk.forward(x, past)
            else:
                x = blk(x)


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

    Subclasses that need to mutate the module tree before the weights
    are loaded (e.g. FlashVSR's identity-deepening of ``decoder.blocks``)
    pass ``checkpoint_path=None`` to ``super().__init__`` -- which stops
    after the meta construction -- and then call
    :meth:`load_from_checkpoint` once the mutation is done. See
    ``integrations/flashvsr/flashvsr/decoder/network.py`` for the live
    example.

    Per-checkpoint key remaps and shape patches live in
    :mod:`flashdreams.recipes.taehv.checkpoint`; pass them in via the
    ``state_dict_transform`` kwarg (typically declared next to the
    checkpoint URL in the consuming config -- see ``TeahvVAEDecoderConfig``,
    ``FlashVSRDecoderConfig``). When left at its default ``None``,
    :meth:`load_from_checkpoint` applies a generic default: a
    ``decoder.<i>.*`` → ``decoder.blocks.<i>.*`` rewrite plus a
    model-aware ``TGrow`` truncation walked off the live
    ``decoder.blocks``.

    Set ``torch.backends.cudnn.benchmark = True`` at process start for ~5%
    extra on the eager seed/tail chunks.
    """

    TEMPORAL_COMPRESSION_RATIO = 4
    SPATIAL_COMPRESSION_RATIO = 8

    SUPPORTED_MODEL_TYPES = ("wan21", "wan22")

    # Concrete type so ``self.decoder`` access doesn't go through
    # ``nn.Module.__getattr__``'s ``Tensor | Module``.
    decoder: "Decoder"

    def __init__(
        self,
        checkpoint_path: str | None = "taew2_1.pth",
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
        patch_size: int = 1,
        latent_channels: int = 16,
        channels: tuple[int, int, int, int] = (256, 128, 64, 64),
        clamp_output: bool = True,
        model_type: str = "wan21",
        use_cuda_graph: bool = True,
        use_compile: bool = False,
        warmup_iters: int = 2,
        state_dict_transform: StateDictTransform | None = None,
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
        self.channels = channels
        self.clamp_output = clamp_output
        # Frames the decoder drops from the front of its first chunk output
        # (matches the legacy 2 ** sum(time_upscale) - 1 formula).
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1

        # Build on meta so only the checkpoint allocates real memory.
        with torch.device("meta"):
            self.decoder = Decoder(
                n_f=self.channels,
                latent_channels=latent_channels,
                image_channels=self.image_channels,
                patch_size=patch_size,
                decoder_time_upscale=decoder_time_upscale,
                decoder_space_upscale=decoder_space_upscale,
                act_func=act_func,
            )

        # Runtime knobs consumed by ``load_from_checkpoint``; stashed here
        # so subclasses that defer the load (``checkpoint_path=None``)
        # inherit the same wrapper wiring without re-plumbing every flag.
        self._use_cuda_graph = use_cuda_graph
        self._use_compile = use_compile
        self._warmup_iters = warmup_iters
        self._decoder_wrapper: CUDAGraphWrapper | None = None

        if checkpoint_path is not None:
            self.load_from_checkpoint(
                checkpoint_path, state_dict_transform=state_dict_transform
            )

    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        state_dict_transform: StateDictTransform | None = None,
    ) -> None:
        """Load weights and wire up the decode runtime.

        Runs the loaded state dict through ``state_dict_transform``,
        assigns the result into the meta-built decoder via
        ``load_state_dict(..., assign=True)``, switches the module to
        eval / no-grad, applies ``torch.compile``, and constructs the
        :class:`~flashdreams.infra.cuda_graph.CUDAGraphWrapper`
        according to the flags captured at ``__init__``. See
        :class:`TAEHV` for the deferred-load contract used by
        tree-mutating subclasses. Must be called at most once per
        instance.

        Args:
            checkpoint_path: Path / URL handed to
                :func:`~flashdreams.core.checkpoint.load.load_checkpoint`.
            state_dict_transform: Per-checkpoint state-dict remap (see
                :mod:`flashdreams.recipes.taehv.checkpoint`). ``None``
                applies the generic default:
                :func:`~flashdreams.recipes.taehv.checkpoint.legacy_to_blocks_keys`
                composed with
                :func:`~flashdreams.recipes.taehv.checkpoint.truncate_oversize_tgrow_weights_from_blocks`
                walked off the live ``self.decoder.blocks``.
        """
        if state_dict_transform is None:
            # Generic default: structural key rewrite + model-aware TGrow
            # shape patch. Built off the live ``decoder.blocks`` so subclass
            # mutations (e.g. FlashVSR's identity deepening) are reflected
            # in the discovered ``TGrow`` indices.
            state_dict_transform = compose(
                legacy_to_blocks_keys,
                truncate_oversize_tgrow_weights_from_blocks(self.decoder.blocks),
            )
        sd = load_checkpoint(checkpoint_path)
        sd = state_dict_transform(sd)
        # assign=True: meta params become the checkpoint tensors directly;
        # strict=False: silently drop encoder-only weights.
        self.load_state_dict(sd, strict=False, assign=True)

        self.eval().requires_grad_(False)

        if self._use_compile:
            self.decoder = compile_module(self.decoder)

        self._decoder_wrapper = (
            CUDAGraphWrapper(self.decoder, warmup_iters=self._warmup_iters)
            if self._use_cuda_graph
            else None
        )

    @property
    def _decoder_call(self) -> Callable[..., torch.Tensor]:
        """Steady-state decoder entry point (wrapper if present, else decoder).

        Implemented as a property so the lookup sidesteps
        ``nn.Module.__setattr__``'s submodule auto-registration. A plain
        ``self._decoder_call = self.decoder`` fallback would re-register
        ``decoder`` under a second name and duplicate every key in
        ``state_dict``.
        """
        return (
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
        self,
        z: torch.Tensor,
        cache: Optional[TAEHVCache] = None,
        **_: object,
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
        if first_decode:
            # Pre-populate state with zeros so Dynamo's first decoder trace
            # already sees ``state[key]`` as Tensor; without this, AR1 would
            # recompile when the dict transitions from empty to populated.
            b, t, c_z, h_z, w_z = z.shape
            self.decoder.initialize_state(
                (b, t, c_z, h_z, w_z), z.dtype, z.device, state
            )
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
        if self.clamp_output:
            x = x.clamp_(0, 1)
        if self.patch_size > 1:
            n, t, c, h, w = x.shape
            x = F.pixel_shuffle(x.reshape(n * t, c, h, w), self.patch_size)
            x = x.reshape(n, t, x.shape[1], x.shape[2], x.shape[3])
        if first_decode:
            x = x[:, self.frames_to_trim :]
        return x
