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

"""FlashVSR encoder: bicubic upres + per-block LR-latent projector.

Wraps :class:`Causal_LQ4x_Proj` (in :mod:`.network`) for the
:class:`flashdreams.infra.encoder.StreamingEncoder` interface; one
``forward()`` call bicubic-upsamples a chunk of LR frames, center-crops
the upres to the largest 128-multiple, and runs the streaming
projector. The (cropped) bicubic upres is side-stashed on the cache so
the pipeline can forward it to the decoder (TC decoder ``cond`` +
color-corrector AdaIN reference). See the sibling ``README.md`` for the
streaming chunk contract.

The bicubic-then-128-multiple-crop step mirrors upstream FlashVSR's
``compute_scaled_and_target_dims`` / ``upscale_then_center_crop``
helpers (in ``examples/WanVSR/infer_flashvsr_v1.1_tiny.py``): the
projector's 16x spatial pixel-shuffle and the DiT's 8-window patchify
together require ``target_H % 128 == 0`` and ``target_W % 128 == 0``,
so any spillover from ``input * scale`` is symmetric-trimmed away
post-bicubic. This loses at most 127 pixels per axis; the LR
equivalent is ``< 128 / scale`` LR pixels per axis.

Cold-start chunks rely on the bicubic-vs-pad commutativity: bicubic is
spatial-only, so ``bicubic(replicate_pad_left(lowres)) ==
replicate_pad_left(bicubic(lowres))``, which licenses storing the
un-padded bicubic upres and letting the decoder's ``PixelShuffle3d``
replicate-pad frame 0 to match the legacy padded path byte-for-byte.
Set ``FLASHVSR_DEV_ASSERT=1`` to enable a runtime check of this on
cold-start chunks (off by default).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)
from flashdreams.infra.profiler import EventProfiler, record_event
from flashvsr.constants import (
    FLASHVSR_CHUNK_FRAME_TARGETS,
    FLASHVSR_FRAMES_PER_DIT_ITER,
)
from flashvsr.encoder.network import (
    Causal_LQ4x_Proj,
    Causal_LQ4x_Proj_Cache,
)

_DEV_ASSERT = os.environ.get("FLASHVSR_DEV_ASSERT", "0") == "1"
"""Opt-in cold-start bicubic-vs-pad equivalence check. Set
``FLASHVSR_DEV_ASSERT=1`` to enable; production stays fast."""

__all__ = [
    "FlashVSREncoder",
    "FlashVSREncoderConfig",
    "FlashVSREncoderCache",
]


@dataclass(kw_only=True)
class FlashVSREncoderConfig(EncoderConfig):
    """Configuration for :class:`FlashVSREncoder`.

    Pipeline contract: one encoder call ingests N raw low-resolution
    frames at ``input_H x input_W`` and returns the per-block LR latent
    slices covering ``N // 4`` latent frames (= ``N // 8`` DiT iterations).
    Allowed N is the legacy
    ``_CHUNK_TARGET = {5: 8, 13: 16, 8: 8, 16: 16}`` table at every AR
    step. The cold-start sizes (5 / 13) are pad-left replicated to 8 / 16
    so the projector's 4-frame causal stride aligns; the steady sizes
    (8 / 16) pass through unchanged.

    Output dimensions follow upstream FlashVSR's "bicubic then
    center-crop to a 128-multiple" rule: bicubic upsamples to
    ``(input_H * scale, input_W * scale)`` and the encoder then
    center-crops to ``target_H = (input_H * scale) // 128 * 128`` /
    ``target_W = (input_W * scale) // 128 * 128``. The DiT and decoder
    both see the cropped target; pipelines built with non-aligned
    ``input_H`` / ``input_W`` work but lose up to ``128 - 1`` HR pixels
    per axis to the symmetric trim.
    """

    _target: type["FlashVSREncoder"] = field(default_factory=lambda: FlashVSREncoder)

    input_H: int = 540
    """Low-resolution input height in pixels."""

    input_W: int = 960
    """Low-resolution input width in pixels."""

    scale: Literal[2, 4] = 2
    """Output / input pixel scale factor."""

    projector_in_dim: int = 3
    """Projector input channels (raw RGB)."""

    projector_out_dim: int = 1536
    """Projector output dim per linear head; matches the DiT hidden width."""

    projector_layer_num: int = 1
    """Number of per-block linear heads in the projector. The shipped
    FlashVSR-v1.1 projector has only one head, so the LR injection lands on
    DiT block 0 alone (the per-block list ``input[i]`` is consumed by
    ``FlashVSRDiTNetwork.forward`` only when ``i < len(input)``)."""

    projector_checkpoint_path: str | None = None
    """Path / URL to the projector ``.ckpt``. ``None`` leaves the
    projector at its random init (only useful in tests)."""

    use_compile: bool = False
    """``torch.compile`` the projector's streaming forward."""

    use_cuda_graph: bool = False
    """Wrap the projector in ``CUDAGraphWrapper`` for steady-state replay.

    Defaults to ``False`` and matches :attr:`FlashVSRDecoderConfig.use_cuda_graph`.
    :func:`build_flashvsr_v1_1` hard-codes both knobs to ``True`` for
    production wiring; when ``use_compile`` is also ``True``, the wrapper's
    ``drain`` step is what absorbs Inductor's lazy triton autotunes against
    the same staged buffers capture will use (otherwise capture would fail
    with ``cudaErrorStreamCaptureUnsupported``). This default only matters
    when assembling sub-configs by hand outside the builder."""

    dtype: torch.dtype = torch.bfloat16
    """Projector compute dtype. ``bfloat16`` matches FlashVSR-tiny weights."""


@dataclass(kw_only=True)
class FlashVSREncoderCache(StreamingEncoderCache):
    """Per-rollout encoder cache.

    Holds the projector's internal causal-conv tail buffer plus per-step
    side-channel slots that :meth:`FlashVSRPipeline.generate` reads
    between the encoder and the diffusion / decoder calls.

    Per-AR-step profiling lives on the parent
    :attr:`StreamInferencePipelineCache.event_profiler`; the pipeline
    forwards that single instance to :meth:`FlashVSREncoder.forward` as
    an explicit kwarg, so we do not duplicate the slot here.
    """

    proj_cache: Causal_LQ4x_Proj_Cache
    """Projector causal-conv streaming state (``conv1`` / ``conv2`` tails)."""

    last_upres: Tensor | None = None
    """``[B, 3, T_unpadded, target_H, target_W]`` bicubic upres of the
    current chunk; un-padded so the decoder's color corrector references the
    user-visible frames only. Set by ``FlashVSREncoder.forward``; read by
    ``FlashVSRPipeline.generate``."""

    last_n_iters: int = 0
    """Number of internal DiT iterations the pipeline must run for the
    current chunk (= ``T_padded // 8``). Set by ``FlashVSREncoder.forward``
    so the pipeline doesn't have to re-derive it from the encoder output's
    token count. Equals ``1`` for an 8-frame chunk and ``2`` for a 16-frame
    chunk; matches the legacy ``n_iters = (T // 4) // 2``."""


class FlashVSREncoder(StreamingEncoder[FlashVSREncoderCache]):
    """Bicubic-upsample + ``Causal_LQ4x_Proj`` encoder for FlashVSR."""

    _CHUNK_FRAME_TARGETS: dict[int, int] = FLASHVSR_CHUNK_FRAME_TARGETS
    """``(raw_T -> padded_T)`` table accepted at every AR step. Cold-start
    sizes (5 / 13) get pad-left replicated to 8 / 16; steady sizes
    (8 / 16) pass through. Cold sizes are accepted at every step (not
    only step 0) so callers can re-prime a cache mid-stream after a
    scene cut. ``T_padded // FLASHVSR_FRAMES_PER_DIT_ITER`` is the
    number of internal DiT iterations the pipeline will run for the
    chunk."""

    def __init__(self, config: FlashVSREncoderConfig) -> None:
        super().__init__(config)
        self.config: FlashVSREncoderConfig = config
        # Bicubic upsamples to ``(scaled_H, scaled_W)`` (un-rounded
        # ``input * scale``) and then center-crops to ``(target_H,
        # target_W)`` -- the largest 128-multiple that fits. This mirrors
        # upstream's ``compute_scaled_and_target_dims`` /
        # ``upscale_then_center_crop`` helpers in
        # ``examples/WanVSR/infer_flashvsr_v1.1_tiny.py`` and keeps the
        # DiT's ``target % 128 == 0`` invariant satisfied for any input
        # dims (the projector's 16x pixel-shuffle + the DiT's 8-window
        # patchify together require the post-crop ``target % 128 == 0``).
        self.scaled_H = config.input_H * config.scale
        self.scaled_W = config.input_W * config.scale
        self.target_H = (self.scaled_H // 128) * 128
        self.target_W = (self.scaled_W // 128) * 128
        assert self.target_H > 0 and self.target_W > 0, (
            f"Scaled size {self.scaled_H}x{self.scaled_W} too small to crop to "
            f"a 128-multiple; need input_H * scale and input_W * scale to each "
            f"be >= 128 (got input_H={config.input_H}, input_W={config.input_W}, "
            f"scale={config.scale})."
        )

        projector = Causal_LQ4x_Proj(
            in_dim=config.projector_in_dim,
            out_dim=config.projector_out_dim,
            layer_num=config.projector_layer_num,
            use_cuda_graph=config.use_cuda_graph,
            use_compile=config.use_compile,
        )
        if config.projector_checkpoint_path is not None:
            projector.load_state_dict(
                load_checkpoint(config.projector_checkpoint_path),
                strict=True,
            )
        self.projector = projector.to(dtype=config.dtype)

    def initialize_autoregressive_cache(self, **_unused: Any) -> FlashVSREncoderCache:
        return FlashVSREncoderCache(
            proj_cache=self.projector.create_external_cache(),
            last_upres=None,
        )

    def forward(  # type: ignore[override]
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: FlashVSREncoderCache | None = None,
        event_profiler: EventProfiler | None = None,
    ) -> list[Tensor]:
        """Bicubic-upsample ``input`` and project it to per-block LR latents.

        Args:
            input: Low-resolution frames ``[B, 3, T, H, W]`` in ``[-1, 1]``.
                See :data:`_CHUNK_FRAME_TARGETS` for allowed ``T``;
                cold-start counts (5 / 13) get pad-left replicated to
                8 / 16.
            autoregressive_index: AR step index. Currently unused for
                input validation -- the legacy table accepts the same
                ``T`` values at every AR step -- but kept for parity with
                the legacy signature.
            cache: Per-rollout encoder cache. Updated in place: the
                projector causal-conv tail, ``last_upres`` and
                ``last_n_iters`` are set here.
            event_profiler: Optional per-AR-step profiler (forwarded by
                :meth:`FlashVSRPipeline.generate` from the parent
                :attr:`StreamInferencePipelineCache.event_profiler`).
                The encoder records the ``pad`` / ``bicubic`` / ``projector``
                sub-stages against it.

        Returns:
            ``list[Tensor]`` of per-block LR latents, one entry per
            projector linear head, each of shape
            ``[B, n_iters * len_t * pH * pW, dim]`` where
            ``n_iters = T_padded // FLASHVSR_FRAMES_PER_DIT_ITER``.
        """
        del autoregressive_index  # parity-only; see docstring
        assert cache is not None, "FlashVSREncoder requires a cache"
        B, _3, T_raw, H, W = input.shape
        assert (H, W) == (self.config.input_H, self.config.input_W), (
            f"input frames at {H}x{W} but encoder configured for "
            f"{self.config.input_H}x{self.config.input_W}"
        )

        target_T = self._CHUNK_FRAME_TARGETS.get(T_raw)
        if target_T is None:
            raise AssertionError(
                f"T={T_raw} not in supported chunk targets: "
                f"expected one of {sorted(self._CHUNK_FRAME_TARGETS)}."
            )
        n_left_padding = target_T - T_raw
        if n_left_padding > 0:
            # Replicate-pad on the **raw** input. Bicubic-upsampling first
            # then padding the upres along the temporal axis would also
            # work numerically -- the legacy code happened to pad first,
            # so we keep that ordering. The runtime assertion below
            # double-checks the equivalence on cold-start chunks during
            # development, gated behind an env flag so production stays
            # fast.
            input = F.pad(input, (0, 0, 0, 0, n_left_padding, 0), mode="replicate")
        T = target_T

        record_event(event_profiler, "pad")

        # Bicubic upsample to the un-rounded ``(scaled_H, scaled_W)``, then
        # center-crop to the 128-multiple ``(target_H, target_W)``. Mirrors
        # upstream's ``upscale_then_center_crop`` in
        # ``examples/WanVSR/infer_flashvsr_v1.1_tiny.py``: bicubic first so
        # the kernel's edge behaviour sees the full input, then symmetric
        # crop trims the non-128-aligned overhang (e.g. 832 -> 768 = 32 px
        # top + 32 px bottom for a 416-pixel-tall LR input at scale=2).
        # ``[B, 3, T, H, W] -> [B, 3, T, target_H, target_W]``.
        upres = (
            F.interpolate(
                input.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W),
                size=(self.scaled_H, self.scaled_W),
                mode="bicubic",
                align_corners=False,
            )
            .view(B, T, 3, self.scaled_H, self.scaled_W)
            .permute(0, 2, 1, 3, 4)
        )
        top = (self.scaled_H - self.target_H) // 2
        left = (self.scaled_W - self.target_W) // 2
        upres = upres[:, :, :, top : top + self.target_H, left : left + self.target_W]

        # Stash the un-padded upres for the decoder/color-corrector. Padding
        # only existed to keep the projector's 4-frame causal stride aligned;
        # downstream stages should see only the user-visible frames.
        cache.last_upres = upres[:, :, n_left_padding:, :, :]
        # Each DiT iter consumes ``FLASHVSR_FRAMES_PER_DIT_ITER`` raw frames
        # (= 2 latent frames after the projector's 4x temporal compression).
        # Mirrors the legacy ``n_iters = (T // 4) // 2``.
        cache.last_n_iters = T // FLASHVSR_FRAMES_PER_DIT_ITER

        if _DEV_ASSERT and n_left_padding > 0:
            # Cold-start equivalence check: replicate-padding the raw
            # lowres before bicubic must produce the same upres as
            # replicate-padding the bicubic of the un-padded lowres.
            # This is what licenses the (un-padded ``last_upres`` ->
            # decoder ``PixelShuffle3d`` frame-0 replicate) path to match
            # the legacy padded-then-bicubic path byte-for-byte.
            reconstructed = F.pad(
                cache.last_upres,
                (0, 0, 0, 0, n_left_padding, 0),
                mode="replicate",
            )
            assert torch.equal(reconstructed, upres), (
                "Cold-start padding equivalence check failed: "
                "bicubic(replicate_pad(lowres)) differs from "
                "replicate_pad(bicubic(lowres)). The decoder's PixelShuffle3d "
                "frame-0 replicate would no longer match the legacy padded path."
            )

        record_event(event_profiler, "bicubic")

        # Per-block LR latents: list of ``num_layers`` tensors, each
        # ``[B, n_iters * len_t * pH * pW, dim]``.
        out = self.projector.forward_streaming(upres, cache.proj_cache)

        record_event(event_profiler, "projector")

        return out
