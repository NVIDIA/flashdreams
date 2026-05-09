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
``flashdreams.infra.encoder.StreamingEncoder`` interface; one ``forward()`` call
bicubic-upsamples a chunk of LR frames and runs the streaming projector.
The bicubic upres is side-stashed on the cache so the pipeline can
forward it to the decoder (TC decoder ``cond`` + color-corrector AdaIN
reference). See ``README.md`` (sibling) for the streaming chunk contract.

The fast path is a hand-rolled CUDA extension
(:mod:`csrc/bicubic_pixelshuffle_cuda.cu`) that fuses the temporal
replicate-pad-left, bicubic upres, and the projector's spatial
pixel-shuffle into one kernel launch. It emits both the projector's
post-pixel-shuffle conv1 input and the un-padded BCTHW ``last_upres`` in
a single pass. Cold-start chunks rely on the bicubic-vs-pad
commutativity (bicubic is spatial-only, so
``bicubic(replicate_pad_left(lowres)) ==
replicate_pad_left(bicubic(lowres))``) so the kernel can fold the pad
into ``t_in = max(0, t_padded - n_left_padding)`` index math without
materialising the padded upres.

The eager PyTorch path is preserved as a fallback for non-CUDA hosts
and for hosts where the extension fails to build.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.cpp_extension import load as _load_cuda_extension

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.encoder import StreamingEncoder, StreamingEncoderCache
from flashdreams.infra.profiler import EventProfiler, record_event
from flashdreams.recipes.flashvsr.constants import (
    FLASHVSR_CHUNK_FRAME_TARGETS,
    FLASHVSR_FRAMES_PER_DIT_ITER,
)
from flashdreams.recipes.flashvsr.encoder.network import (
    Causal_LQ4x_Proj,
    Causal_LQ4x_Proj_Cache,
)

__all__ = [
    "FlashVSREncoder",
    "FlashVSREncoderConfig",
    "FlashVSREncoderCache",
]


_BICUBIC_PIXELSHUFFLE_EXTENSION = None
_BICUBIC_PIXELSHUFFLE_LOAD_ERROR: Optional[Exception] = None


def _load_bicubic_pixelshuffle_extension():
    """Lazy-load the fused bicubic + pixel-shuffle CUDA extension.

    Mirrors :func:`flashdreams.recipes.flashvsr.corrector._load_adain_cuda_extension`:
    the source bytes' sha256 is baked into the extension name so any edit
    to ``bicubic_pixelshuffle_cuda.cu`` invalidates the cached ``.so``
    under ``~/.cache/torch_extensions``. Returns ``None`` on load failure
    (CPU-only host, missing toolchain, etc.) -- callers fall back to the
    eager PyTorch path.
    """
    global _BICUBIC_PIXELSHUFFLE_EXTENSION, _BICUBIC_PIXELSHUFFLE_LOAD_ERROR
    if _BICUBIC_PIXELSHUFFLE_EXTENSION is not None:
        return _BICUBIC_PIXELSHUFFLE_EXTENSION
    if _BICUBIC_PIXELSHUFFLE_LOAD_ERROR is not None:
        return None

    source_path = Path(__file__).parent.parent / "csrc" / "bicubic_pixelshuffle_cuda.cu"
    csrc_checksum = hashlib.sha256(source_path.read_bytes()).hexdigest()[:8]
    # try:
    _BICUBIC_PIXELSHUFFLE_EXTENSION = _load_cuda_extension(
        name=f"flashvsr_bicubic_pixelshuffle_cuda_{csrc_checksum}",
        sources=[str(source_path)],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    # except Exception as exc:
    #     _BICUBIC_PIXELSHUFFLE_LOAD_ERROR = exc
    #     return None
    return _BICUBIC_PIXELSHUFFLE_EXTENSION


@dataclass(kw_only=True)
class FlashVSREncoderConfig(InstantiateConfig["FlashVSREncoder"]):
    """Configuration for :class:`FlashVSREncoder`.

    Pipeline contract: one encoder call ingests N raw low-resolution
    frames at ``input_H x input_W`` and returns the per-block LR latent
    slices covering ``N // 4`` latent frames (= ``N // 8`` DiT iterations).
    Allowed N is the legacy
    ``_CHUNK_TARGET = {5: 8, 13: 16, 8: 8, 16: 16}`` table at every AR
    step. The cold-start sizes (5 / 13) are pad-left replicated to 8 / 16
    so the projector's 4-frame causal stride aligns; the steady sizes
    (8 / 16) pass through unchanged.
    """

    _target: type["FlashVSREncoder"] = field(default_factory=lambda: FlashVSREncoder)

    input_H: int = 540
    """Low-resolution input height in pixels."""

    input_W: int = 960
    """Low-resolution input width in pixels."""

    scale: Literal[2, 4] = 2
    """Output / input pixel scale factor."""

    projector_in_dim: int = 3
    projector_out_dim: int = 1536
    projector_layer_num: int = 1
    """Number of per-block linear heads in the projector. The shipped
    FlashVSR-v1.1 projector has only one head, so the LR injection lands on
    DiT block 0 alone (the per-block list ``input[i]`` is consumed by
    ``FlashVSRDiTNetwork.forward`` only when ``i < len(input)``)."""

    projector_checkpoint_path: str | None = None

    use_compile: bool = False
    """``torch.compile`` the projector's streaming forward."""

    use_cuda_graph: bool = False
    """Wrap the projector in ``CUDAGraphWrapper`` for steady-state replay.

    Defaults to ``False`` and matches :class:`FlashVSRDecoderConfig.use_cuda_graph`.
    :func:`build_flashvsr_v1_1` hard-codes both knobs to ``True`` for
    production wiring; when ``use_compile`` is also ``True``, the wrapper's
    ``drain`` step is what absorbs Inductor's lazy triton autotunes against
    the same staged buffers capture will use (otherwise capture would fail
    with ``cudaErrorStreamCaptureUnsupported``). This default only matters
    when assembling sub-configs by hand outside the builder."""

    dtype: torch.dtype = torch.bfloat16


@dataclass(kw_only=True)
class FlashVSREncoderCache(StreamingEncoderCache):
    """Per-rollout encoder cache.

    Holds the projector's internal causal-conv tail buffer plus per-step
    side-channel slots that :class:`FlashVSRPipeline.generate` reads
    between the encoder and the diffusion / decoder calls.

    Per-AR-step profiling lives on the parent
    :class:`StreamInferencePipelineCache.event_profiler`; the pipeline
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

    # Mirrors the legacy ``_CHUNK_TARGET = {5: 8, 13: 16, 8: 8, 16: 16}``
    # from ``UltraFlashVSRUpsampler.forward``. Accepted at every AR step:
    # the cold-start sizes (5 / 13) get pad-left replicated to 8 / 16, the
    # steady sizes (8 / 16) pass through. Splitting cold vs steady would
    # break legacy callers that occasionally feed the cold-start sizes
    # mid-stream (e.g. when re-priming a cache after a scene cut).
    # ``T_padded // FLASHVSR_FRAMES_PER_DIT_ITER`` is the number of DiT
    # iterations the pipeline will run for this chunk.
    _CHUNK_FRAME_TARGETS: dict[int, int] = FLASHVSR_CHUNK_FRAME_TARGETS

    def __init__(self, config: FlashVSREncoderConfig) -> None:
        super().__init__(config)
        self.config: FlashVSREncoderConfig = config
        self.target_H = config.input_H * config.scale
        self.target_W = config.input_W * config.scale
        assert self.target_W % 128 == 0 and self.target_H % 128 == 0, (
            "Target resolution must be divisible by 128 (FlashVSR DiT requires "
            "post-patchify h/w divisible by the 8-window size and the projector "
            "uses an 16x spatial pixel-shuffle)."
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
                :class:`FlashVSRPipeline.generate` from the parent
                :class:`StreamInferencePipelineCache.event_profiler`).
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
        # Each DiT iter consumes ``FLASHVSR_FRAMES_PER_DIT_ITER`` raw frames
        # (= 2 latent frames after the projector's 4x temporal compression).
        # Mirrors the legacy ``n_iters = (T // 4) // 2``.
        cache.last_n_iters = target_T // FLASHVSR_FRAMES_PER_DIT_ITER

        if input.is_cuda:
            ext = _load_bicubic_pixelshuffle_extension()
        else:
            ext = None

        input = input.contiguous()
        if ext is not None:
            # Fast path: one kernel folds pad + permute+contiguous +
            # bicubic + permute-back + pixel-shuffle, emitting both the
            # projector's post-shuffle conv1 input and the un-padded
            # ``last_upres`` in a single launch.
            proj_input, last_upres = ext.bicubic_pixelshuffle_forward(
                input,
                target_T,
                self.target_H,
                self.target_W,
                n_left_padding,
            )
            # ``pad`` is folded into the kernel; record the synthetic event
            # so the per-AR-step profile keeps the same ``pad`` /
            # ``bicubic`` / ``projector`` breakdown shape.
            record_event(event_profiler, "pad")
            cache.last_upres = last_upres
            record_event(event_profiler, "bicubic")
            out = self.projector.forward_streaming_from_shuffled(
                proj_input, cache.proj_cache
            )
            record_event(event_profiler, "projector")
            return out
        else:
            raise RuntimeError("Bicubic pixelshuffle CUDA extension not found")

        # Eager PyTorch fallback (non-CUDA hosts, extension build failure).
        # Mirrors the pre-fusion code path; kept around so CPU smokes and
        # tests without an nvcc toolchain still exercise the encoder.
        if n_left_padding > 0:
            input = F.pad(input, (0, 0, 0, 0, n_left_padding, 0), mode="replicate")
        T = target_T

        record_event(event_profiler, "pad")

        upres = (
            F.interpolate(
                input.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W),
                size=(self.target_H, self.target_W),
                mode="bicubic",
                align_corners=False,
            )
            .view(B, T, 3, self.target_H, self.target_W)
            .permute(0, 2, 1, 3, 4)
        )
        # Un-padded upres for the decoder/color-corrector. The bicubic
        # is spatial-only, so the un-padded slice equals
        # ``bicubic(unpadded_lowres)`` exactly -- the same identity the
        # CUDA kernel relies on to skip materialising the padded upres.
        cache.last_upres = upres[:, :, n_left_padding:, :, :]

        record_event(event_profiler, "bicubic")

        out = self.projector.forward_streaming(upres, cache.proj_cache)

        record_event(event_profiler, "projector")

        return out
