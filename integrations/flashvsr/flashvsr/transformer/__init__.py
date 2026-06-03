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

"""Reference :class:`Transformer` subclass for the FlashVSR integration.

Mirrors the ``template/transformer/__init__.py`` role: this module owns
the :class:`Wan21Transformer` subclass (including its config) that wraps
the raw :class:`FlashVSRDiTNetwork` from :mod:`.network` and exposes the
streaming inference contract (``predict_flow``, autoregressive cache
lifecycle, KV-cache-aware patchify hook).

FlashVSR is single-branch only by design: the distilled checkpoint does
not ship negative-prompt embeddings, so the parent's CFG plumbing is
structurally bypassed. ``FlashVSRTransformerConfig.__post_init__``
asserts ``guidance_scale == 1.0`` and ``predict_flow`` defensively
asserts ``cache.network_cache_uncond is None`` to catch a misconfigured
cache (e.g. one built with a CFG-enabled transformer config and then
used here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
from torch import Tensor

from flashdreams.core.distributed.context_parallel import split_inputs_cp
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.recipes.wan.transformer.impl.network import WanDiTNetworkConfig
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerCache,
    Wan21TransformerConfig,
)
from flashvsr.transformer.network import (
    _SELF_ATTN_WINDOW,
    _SELF_ATTN_WINDOW_TOKENS,
    FlashVSRDiTNetwork,
    FlashVSRDiTNetworkConfig,
)

__all__ = [
    "FlashVSRTransformer",
    "FlashVSRTransformerConfig",
]


@dataclass(kw_only=True)
class FlashVSRTransformerConfig(Wan21TransformerConfig):
    """Configuration for :class:`FlashVSRTransformer`.

    Wraps :class:`Wan21TransformerConfig` with the FlashVSR-specific knobs:

    - ``topk_ratio``: top-k block budget multiplier (the legacy DiT computes
      ``topk = int(window_size**2 * topk_ratio) - 1``).
    - ``kv_ratio``: number of prior chunks retained in the streaming
      self-attention KV cache (the just-written chunk is also visible at
      attention time, so the buffer holds ``kv_ratio + 1`` chunks).
    - ``local_range``: spatial window radius for the local-block mask.
    - ``attention_mode``: ``"sparse"`` preserves the legacy block-sparse path;
      ``"full"`` uses the dense CP-aware Wan self-attention path.

    ``__post_init__`` enforces the parent's invariants and additionally:

    - Sets ``window_size_t = (kv_ratio + 1) * len_t`` and
      ``sink_size_t = 0`` so ``Wan21Transformer._build_network_cache``
      sizes the per-block ``BlockKVCache`` to the (kv_ratio+1)-chunk
      capacity FlashVSR expects.
    """

    _target: type["FlashVSRTransformer"] = field(  # type: ignore[assignment]
        default_factory=lambda: FlashVSRTransformer
    )

    network: WanDiTNetworkConfig = field(default_factory=FlashVSRDiTNetworkConfig)
    len_t: int = 2
    """FlashVSR processes 2 latent frames per DiT iteration."""

    topk_ratio: float = 2.0
    """Multiplier on the per-chunk window count squared; sets the top-k block budget."""

    kv_ratio: int = 3
    """Number of prior chunks retained in the streaming self-attention KV cache.

    The buffer holds ``kv_ratio + 1`` chunks at attention time -- the
    ``kv_ratio`` cached prior chunks plus the just-written current one.
    ``__post_init__`` translates this into
    ``window_size_t = (kv_ratio + 1) * len_t`` for the inherited
    ``Wan21Transformer._build_network_cache``, which sizes the per-block
    :class:`BlockKVCache` accordingly."""

    local_range: int = 11
    """Local-block window radius (in window units) for the draft mask."""

    attention_mode: Literal["sparse", "full"] = "sparse"
    """Self-attention implementation. Multi-GPU is supported only with ``"full"``."""

    use_cuda_graph: bool = False
    """Capture the **steady-state** DiT call into a CUDA graph and replay it.

    Mirrors :attr:`FlashVSREncoderConfig.use_cuda_graph` and
    :attr:`FlashVSRDecoderConfig.use_cuda_graph`; the inherited
    :attr:`Wan21TransformerConfig.compile_network` plays the role of
    those configs' ``use_compile``.

    When ``True``, ``predict_flow`` lazily wraps the DiT forward in a
    :class:`flashdreams.infra.cuda_graph.CUDAGraphWrapper`. Filling chunks
    (``internal_ar_idx < kv_ratio + 1``) still run eagerly because the
    K/V cache tensor's effective shape varies during fill. From the first
    steady-state call onwards the wrapper drains Inductor autotune for
    ``warmup_iters`` calls, then captures one graph and replays it for
    every subsequent steady-state call.

    Requires ``compile_network=True`` to give Inductor a clean graph to
    autotune; the wrapper will fail at capture time otherwise (lazy
    triton autotunes are illegal during capture).

    Phase 2 of the optimization plan in
    ``internal/upsampler/PERF_NOTES.md``. Empirical ceiling on H100 is
    ~5-10% off ``dit_concat`` (~30-60 ms);
    the headline ``dit_concat`` budget is dominated by the
    Triton block-sparse attention kernel which CUDA graphs do not affect."""

    def __post_init__(self) -> None:
        # ``Wan21TransformerConfig`` has no ``__post_init__`` of its own,
        # so there's nothing to chain into.
        assert self.guidance_scale == 1.0, (
            "FlashVSR does not support classifier-free guidance; "
            f"set guidance_scale=1.0 (got {self.guidance_scale})."
        )
        if self.attention_mode not in ("sparse", "full"):
            raise ValueError(
                f"Unsupported attention_mode={self.attention_mode!r}; "
                "expected 'sparse' or 'full'."
            )
        if isinstance(self.network, FlashVSRDiTNetworkConfig):
            self.network.attention_mode = self.attention_mode
        # FlashVSR's KV cache holds ``kv_ratio + 1`` chunks at attention time
        # (the cached prior chunks plus the just-written one). Map this onto
        # the parent's pre-patchify frame-window knob so the inherited
        # ``_build_network_cache`` sizes the buffer correctly.
        self.window_size_t = (self.kv_ratio + 1) * self.len_t
        self.sink_size_t = 0


class FlashVSRTransformer(Wan21Transformer):
    """Wan 2.1 transformer specialised for the FlashVSR streaming VSR DiT."""

    config: FlashVSRTransformerConfig
    network: FlashVSRDiTNetwork

    def __init__(self, config: FlashVSRTransformerConfig) -> None:
        if (
            config.attention_mode == "sparse"
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise ValueError(
                "FlashVSR attention_mode='sparse' is single-GPU only because "
                "Triton sparse attention is not context-parallel aware. Use "
                "attention_mode='full' for multi-GPU context parallelism."
            )
        super().__init__(config)

        # CUDA-graph wrapper for the steady-state DiT call. Lazy-initialised
        # on the first steady-state call in :meth:`predict_flow` so we don't
        # pay the wrapper-construction cost when ``use_cuda_graph`` is False.
        # Tied to a specific ``Wan21TransformerCache`` instance via
        # ``_captured_cache``; if a different cache is passed (new rollout
        # via ``initialize_autoregressive_cache``), we reset the wrapper so
        # the new cache's storage pointers get re-staged at re-capture time.
        self._cuda_graph_wrapper: Optional[CUDAGraphWrapper] = None
        self._captured_cache: Optional[Wan21TransformerCache] = None

    def finalize_kv_cache(self, *args: Any, **kwargs: Any) -> None:
        """No-op: FlashVSR keys its KV cache from the **noisy** forward."""

    def patchify_and_maybe_split_cp(self, x):  # type: ignore[override]
        """CP-split LR-latent lists; defer tensors to the standard path.

        ``FlashVSREncoder.forward`` returns the per-block low-resolution
        latent slices as a ``list[Tensor]`` already in ``[B, L, D]``
        post-patchify space (the projector emits them that way). The
        infra ``DiffusionModel.generate`` calls
        ``transformer.patchify_and_maybe_split_cp(input)`` unconditionally on
        the encoder output; lists therefore only need token-axis CP splitting.
        Plain tensor inputs (e.g. an unpatchified noisy latent in tests) still
        take the parent's rearrange + linear path.
        """
        if isinstance(x, list):
            return [split_inputs_cp(t, seq_dim=-2, cp_group=self._cp_group) for t in x]
        return super().patchify_and_maybe_split_cp(x)

    def _is_steady_state(self, ar_idx: int) -> bool:
        """Return True if the per-block KV cache is full at this AR index.

        With ``sink_size_t == 0`` and
        ``window_size_t == (kv_ratio + 1) * len_t``, the per-block cache
        becomes full after ``kv_ratio + 1`` calls (counted in **internal**
        AR steps -- the ones the transformer cache sees, i.e.
        ``autoregressive_index * n_iters + iter_idx``). From the next call
        onward, ``BlockKVCache.cached_k()`` returns the full buffer (shape
        ``[B, total_size, n_heads, head_dim]``) and all per-step DiT inputs
        are statically shaped -- the prerequisite for graph capture.
        """
        return ar_idx >= self.config.kv_ratio + 1

    def _compute_rope_freqs(self, cache: Wan21TransformerCache, ar_idx: int) -> Tensor:
        """FlashVSR temporal-RoPE: ``ar_idx==0`` uses offset 0, otherwise
        ``2 + ar_idx * 2``.

        The legacy WanModel keeps two distinct ``RotaryPositionEmbedding3D``
        instances (``rope_freq_first`` for ``ar_idx==0``, ``rope_freq_other``
        otherwise); a single instance with two different offsets is
        bit-equivalent because both legacy instances were constructed from
        identical inputs.
        """
        return cache.rope_adapter.shift_t(0 if ar_idx == 0 else 2 + ar_idx * 2)

    def _compute_topk(self, cache: Wan21TransformerCache) -> int:
        """Match the legacy ``WanModel`` top-k computation.

        ``topk = int(block_n_per_chunk ** 2 * topk_ratio) - 1`` where
        ``block_n_per_chunk = win[0] * h * w / 128 = 2 * pH * pW / 128``.
        """
        del cache  # patchified spatial dims live on ``self``, not the cache
        _kt, kh, kw = self.config.network.patch_size
        assert self._output_height is not None and self._output_width is not None, (
            "_compute_topk requires an initialized rollout; call "
            "initialize_autoregressive_cache(..., height=..., width=...) first."
        )
        pH = self._output_height // kh
        pW = self._output_width // kw
        block_n_per_chunk = (_SELF_ATTN_WINDOW[0] * pH * pW) // _SELF_ATTN_WINDOW_TOKENS
        return int(block_n_per_chunk * block_n_per_chunk * self.config.topk_ratio) - 1

    def _capturable_dit_forward(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        rope_freqs: Tensor,
        lq_latents: Optional[list[Tensor] | Tensor],
        *,
        cache,  # WanDiTNetworkCache; non-tensor, passes through CUDAGraphWrapper verbatim
        topk: int,
        f: int,
        h: int,
        w: int,
        local_range: int,
    ) -> Tensor:
        """The DiT forward as :class:`CUDAGraphWrapper` will see it.

        All tensor inputs are top-level positional args so
        ``CUDAGraphWrapper._stage`` copies them into static buffers on every
        call (including pre-capture warmup). The ``cache`` is passed as a
        non-tensor kwarg and threads through verbatim; the network mutates
        its per-block KV buffers in place against stable storage pointers.

        ``lq_latents`` accepts either a ``list[Tensor]`` (eager path:
        forwarded verbatim) or a single leading-dim ``Tensor`` of shape
        ``[num_layers, B, L_per_iter, dim]`` (capture path: staged into a
        static buffer on every wrapper call). The network's consumer
        indexes by block (``lq_latents[i]`` / ``len(lq_latents)``), which
        works identically on both representations.
        :meth:`predict_flow` decides which form to pass based on whether
        the wrapper is in use; passing a list to the wrapper would make
        the captured graph reference whatever list elements existed at
        capture time (wrong on subsequent replays).

        ``eager_mode=False`` because :meth:`predict_flow` runs
        ``before_update`` / ``after_update`` outside the captured region --
        their Python state advances every call, while the conditional GPU
        ops they issue (steady-state ``_roll_local_window_left``) execute
        eagerly. This costs ~600 us per chunk vs. capturing them inside
        and is the price of keeping cache Python state coherent across
        replay.
        """
        # ``current_chunk_idx`` is consulted only when ``eager_mode=True``,
        # which we never pass here; the per-block KV-cache lifecycle runs
        # in :meth:`predict_flow` outside the captured region. Pass 0 as a
        # placeholder so the kwarg is well-typed.
        return self.network(
            x=noisy_latent,
            timesteps=timestep,
            cache=cache,
            rope_freqs=rope_freqs,
            current_chunk_idx=0,
            eager_mode=False,
            block_extra_kwargs={
                "f": f,
                "h": h,
                "w": w,
                "topk": topk,
                "local_range": local_range,
            },
            lq_latents=lq_latents,
        )

    def _get_or_create_wrapper(self, cache: Wan21TransformerCache) -> CUDAGraphWrapper:
        """Lazy-init the cudagraph wrapper; reset on cache identity change.

        Captured kernels reference cache buffer storage pointers, so a
        fresh rollout (new ``initialize_autoregressive_cache`` call) yields
        new pointers and we must reset the wrapper to re-stage and
        re-capture against the new cache.
        """
        if self._cuda_graph_wrapper is None or cache is not self._captured_cache:
            if self._cuda_graph_wrapper is not None:
                self._cuda_graph_wrapper.reset()
            self._cuda_graph_wrapper = CUDAGraphWrapper(
                self._capturable_dit_forward, warmup_iters=2
            )
            self._captured_cache = cache
        return self._cuda_graph_wrapper

    def predict_flow(  # type: ignore[override]
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: Any = None,
        network_extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tensor:
        """Predict the FlashVSR flow at ``timestep``.

        Args:
            noisy_latent: Patchified noisy latent for this AR step
                (i.e. ``patchify_and_maybe_split_cp(...)``-output shape).
            timestep: Scalar timestep tensor; FlashVSR fixes this at 1000.
            cache: Per-rollout AR cache.
            input: Optional list of per-block patchified low-resolution
                latents from the LR projector. ``input[i]`` is added to the
                hidden state at the start of block ``i``.

        Returns:
            Tensor with the same shape as ``noisy_latent`` (the predicted flow).
        """
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0, (
            "Wan21TransformerCache.start(autoregressive_index) must be called "
            "before predict_flow (FlashVSRPipeline.generate runs it per "
            "internal iter)."
        )
        assert cache.network_cache_uncond is None, (
            "FlashVSR doesn't support CFG (guidance_scale must be 1.0); "
            "the CUDA-graph capture path also assumes single-network forward."
        )

        cfg = self.config
        rope_freqs = self._compute_rope_freqs(cache, ar_idx)
        topk = self._compute_topk(cache)
        use_capture = cfg.use_cuda_graph and self._is_steady_state(ar_idx)

        # Stack the per-block LR latent slices into a single tensor only
        # when going through the CUDA-graph wrapper. The wrapper stages
        # top-level tensors into static buffers that the captured graph
        # references on every replay; lists are forwarded verbatim, which
        # would make the captured graph reference whatever list elements
        # existed at capture time (wrong on subsequent replays). The
        # eager path can pass the list straight through -- the network's
        # ``lq_latents[i]`` / ``len(lq_latents)`` consumer works
        # identically on a list and a leading-dim tensor.
        if use_capture and isinstance(input, list):
            lq: Optional[list[Tensor] | Tensor] = torch.stack(input)
        else:
            lq = input  # list[Tensor] | Tensor | None

        # ``before_update`` is hoisted into :meth:`Wan21TransformerCache.start`
        # -- :meth:`FlashVSRPipeline.generate` calls ``cache.start(internal_ar_idx)``
        # per iter before ``predict_flow``, so we don't repeat it here. The
        # paired ``after_update`` is driven from outside this method too:
        # per-iter via ``cache.transformer_cache.finalize(internal_ar_idx)``
        # in the pipeline loop for non-final iters, and via the framework's
        # ``DiffusionModel.finalize`` -> ``final_state.cache.finalize(...)``
        # for the final iter (the pipeline stashes a ``FinalState`` with the
        # last internal iter index so the framework lands on the right one).
        # Keeping both update hooks outside the capturable region means their
        # Python bookkeeping (``_prev_chunk_idx``, ``_n_cached``) advances on
        # every call rather than being baked into the captured graph.
        #
        # Intentionally no try/finally around the forward: if it raises, the
        # iter's ``after_update`` never fires and the cache is left with
        # ``_curr_chunk_idx`` set so the next ``before_update`` fails loudly.
        # A thrown forward leaves the per-block KV buffer in an inconsistent
        # state, and the only safe recovery is :meth:`FlashVSRPipeline.initialize_cache`.
        network_cache = cache.network_cache

        _kt, kh, kw = cfg.network.patch_size
        assert self._output_height is not None and self._output_width is not None, (
            "predict_flow requires an initialized rollout; call "
            "initialize_autoregressive_cache(..., height=..., width=...) first."
        )
        pT = cfg.len_t // _kt
        pH = self._output_height // kh
        pW = self._output_width // kw

        dit_args = (noisy_latent, timestep, rope_freqs, lq)
        dit_kwargs = {
            "cache": network_cache,
            "topk": topk,
            "f": pT,
            "h": pH,
            "w": pW,
            "local_range": cfg.local_range,
        }
        if use_capture:
            flow = self._get_or_create_wrapper(cache)(*dit_args, **dit_kwargs)
        else:
            flow = self._capturable_dit_forward(*dit_args, **dit_kwargs)

        return flow
