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

"""AgiBot transformer cache and CUDA-graph state."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .network import (
        CosmosMoTNetwork,
        _MoTLayer,
        _MoTStreamCache,
    )

from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.transformer import TransformerAutoregressiveCache

from .. import action_format as _agibot
from ..patchify import (
    crop_latent,
    pad_latent,
    patchify_latent,
    unpatchify_latent,
)
from ..sdpa_backends import normalize_sdpa_backend
from .config import (
    DEFAULT_WINDOW_FRAMES,
    validate_cache_options,
)
from .rope import (
    action_mrope_ids,
    text_mrope_ids,
    vae_mrope_ids,
)


class HandTransformerCache(TransformerAutoregressiveCache):
    """Long-lived AgiBot KV, compile, graph, and per-step state.

    Args:
        net: The MoT network.
        text_ids: Packed caption tokens, or ``None`` for no text conditioning.
        init_latent: ``[hw, patch_dim]`` patchified clean frame 0.
        grid_hw: Transformer token grid ``(gh, gw)``.
        chunk_size: Latent frames generated per step.
        tcf: Actions per latent frame (VAE temporal compression).
        action_domain_id: Default embedding domain for legacy tensors and frame zero.
        fps: Conditioning frame rate used for mRoPE temporal scaling.
        window_frames: Frames of generated history to retain, rounded up to a
            whole number of chunks. ``0`` keeps everything, which makes each step
            slower than the last and is not allowed in ``block`` cache mode.
        logical_latent_hw: Content latent size when it is smaller than the grid
            implies. Defaults to the full transformer latent.
        native_gqa_attention: Let SDPA handle grouped-query attention instead of
            expanding K/V heads.
        sdpa_backend: Attention kernel to pin, or ``"default"`` to let PyTorch
            choose.
        cache_mode: One of :data:`CACHE_MODES`. ``block`` needs a bounded window
            and full chunks, since the ring buffer's block size is fixed.
        cache_refresh_mode: One of :data:`CACHE_REFRESH_MODES`. ``framewise``
            requires ``cache_mode="block"``.
        overlap_cache_update: Write the clean K/V back on a side CUDA stream so
            the frames can be decoded while the cache catches up. Requires
            ``cache_mode="block"``; a no-op off CUDA.
        compile_denoise: ``torch.compile`` one velocity evaluation. Requires
            ``cache_mode="block"``, whose fixed buffers give the stable shapes
            compile needs.
        compile_denoise_mode: One of
            :data:`~cosmos3_nano_sim_bimanual.transformer.config.COMPILE_DENOISE_MODES`.
        fixed_shape_cache_fill: During cache fill, pad history to its eventual
            steady-state shape and mask unwritten positions. This enables a
            separate static compiled/CUDA-graph path from the first chunk while
            retaining the unmasked attention path after saturation.
        valid_length_cache_attention: Keep full-size physical K/V buffers from
            the first chunk and pass their dynamic logical prefix length to fused
            varlen FlashAttention. This uses one static compiled/CUDA-graph path
            during fill and steady state without exposing fake history.
        cuda_graph_denoise: Capture the compiled velocity call into a replayable
            CUDA graph. Requires ``compile_denoise`` and a CUDA device. The ring
            buffer is allocated once so captured pointers stay valid across
            rollout resets.
        cuda_graph_warmup_iters: Eager calls before capture, to drain Inductor's
            lazy autotuning, which is illegal during capture.

    Raises:
        ValueError: If a mode is unknown, ``block`` is asked for without a
            window, or an option that depends on ``block``, ``compile_denoise``
            or CUDA is asked for without it.
    """

    def __init__(
        self,
        net: CosmosMoTNetwork,
        *,
        scene: Any,
        text_ids: Tensor | None,
        init_latent: Tensor,
        grid_hw: tuple[int, int],
        chunk_size: int = 4,
        tcf: int = _agibot.TEMPORAL_COMPRESSION_FACTOR,
        action_domain_id: int = _agibot.AGIBOT_DOMAIN_ID,
        fps: float = _agibot.FPS,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        logical_latent_hw: tuple[int, int] | None = None,
        native_gqa_attention: bool = False,
        sdpa_backend: str = "default",
        cache_mode: str = "cat",
        cache_refresh_mode: str = "chunk",
        overlap_cache_update: bool = False,
        compile_denoise: bool = False,
        compile_denoise_mode: str = "max-autotune-no-cudagraphs",
        fixed_shape_cache_fill: bool = False,
        valid_length_cache_attention: bool = False,
        cuda_graph_denoise: bool = False,
        cuda_graph_warmup_iters: int = 2,
        profile: bool = False,
    ) -> None:
        super().__init__()
        self.net = net
        self.scene = scene
        self.spec = net.config.spec
        self.device = net.embed_tokens.weight.device
        self.text_ids = text_ids
        self.gh, self.gw = grid_hw
        self.hw = self.gh * self.gw
        self.patch = self.spec.vfm.latent_patch_size
        self.channels = self.spec.vfm.vae_latent_channels
        self.transformer_h_lat = self.gh * self.patch
        self.transformer_w_lat = self.gw * self.patch
        self.h_lat, self.w_lat = (
            int(value)
            for value in (
                logical_latent_hw or (self.transformer_h_lat, self.transformer_w_lat)
            )
        )
        if (
            self.h_lat <= 0
            or self.w_lat <= 0
            or self.h_lat > self.transformer_h_lat
            or self.w_lat > self.transformer_w_lat
            or -(-self.h_lat // self.patch) != self.gh
            or -(-self.w_lat // self.patch) != self.gw
        ):
            raise ValueError(
                "logical latent geometry must fit and map to the transformer grid: "
                f"logical={self.h_lat}x{self.w_lat}, "
                f"transformer={self.transformer_h_lat}x{self.transformer_w_lat}, "
                f"patch={self.patch}, grid={self.gh}x{self.gw}"
            )
        self.n_text = int(text_ids.shape[0]) if text_ids is not None else 0
        self.chunk_size = int(chunk_size)
        self.tcf = int(tcf)
        self.action_domain_id = int(action_domain_id)
        self.fps = float(fps)
        self._init_latent = init_latent.to(self.device)
        self.action_dim = _agibot.MAX_ACTION_DIM
        self.stride = (
            (self.spec.vfm.base_fps / self.fps)
            if (self.spec.vfm.enable_fps_modulation and self.fps)
            else 1.0
        )

        _, offset = text_mrope_ids(self.n_text, 0.0)
        self._off = offset + self.spec.vfm.mrope_temporal_modality_margin

        self.native_gqa_attention = bool(native_gqa_attention)
        self.sdpa_backend = normalize_sdpa_backend(sdpa_backend)
        self.profile = bool(profile)
        self.last_timing: dict[str, float] = {}

        self.cache_mode = str(cache_mode)
        self.cache_refresh_mode = str(cache_refresh_mode)
        self.overlap_cache_update = bool(overlap_cache_update)
        self.compile_denoise = bool(compile_denoise)
        self.fixed_shape_cache_fill = bool(fixed_shape_cache_fill)
        self.valid_length_cache_attention = bool(valid_length_cache_attention)
        self.cuda_graph_denoise = bool(cuda_graph_denoise)
        validate_cache_options(
            cache_mode=self.cache_mode,
            cache_refresh_mode=self.cache_refresh_mode,
            window_frames=max(0, int(window_frames)),
            overlap_cache_update=self.overlap_cache_update,
            compile_denoise=self.compile_denoise,
            compile_denoise_mode=compile_denoise_mode,
            fixed_shape_cache_fill=self.fixed_shape_cache_fill,
            valid_length_cache_attention=self.valid_length_cache_attention,
            cuda_graph_denoise=self.cuda_graph_denoise,
            device=self.device.type,
        )
        # The cat path reallocates its cache every step, so a side stream would
        # race with that; only the fixed ring buffer can be written concurrently.
        self._cache_stream = (
            torch.cuda.Stream(device=self.device)
            if self.overlap_cache_update
            and self.device.type == "cuda"
            and torch.cuda.is_available()
            else None
        )
        self._pending_cache: list[tuple[torch.cuda.Event, tuple[Tensor, ...]]] = []

        self.compile_denoise_mode = compile_denoise_mode
        self.cuda_graph_warmup_iters = max(0, int(cuda_graph_warmup_iters))

        # Set before the frame-0 seed below, which reads them through _flags().
        self._per_frame_tokens = self.hw + self.tcf
        # Rounded up to whole chunks so the two cache modes span the same history:
        # the ring buffer can only evict a whole block, and frames arrive a chunk
        # at a time, so a chunk is the natural eviction granularity.
        requested_window = max(0, int(window_frames))
        self._window_frames = self.chunk_size * -(-requested_window // self.chunk_size)
        self._abs_frame = 0
        self._steps = 0
        self._ar_index: int | None = None
        self._actions: Tensor | None = None
        self._action_domain_ids: Tensor | None = None
        self._positions_current: Tensor | None = None
        self._is_gen: Tensor | None = None
        self._is_action: Tensor | None = None
        self._frames = 0
        self._vision_tokens = 0
        self._compiled_args: tuple[object, ...] | None = None
        self._compiled_fill_active = False
        self._compiled_masked_fill_active = False
        self._compiled_valid_length_active = False
        self._step_started = 0.0
        self._denoise_started = 0.0
        self._cache_wait_s = 0.0

        self.cache: list[dict[str, Tensor]] | None = None
        self.stream_cache: _MoTStreamCache | None = None
        text_positions = (
            text_mrope_ids(self.n_text, 0.0)[0].to(self.device)
            if text_ids is not None and self.n_text > 0
            else None
        )
        if self.cache_mode == "block":
            self.stream_cache = net.init_stream_cache(
                chunk_grid=(self.chunk_size, self.gh, self.gw),
                window_frames=self._window_frames,
                sink_frames=1,
                tokens_per_frame=self._per_frame_tokens,
            )
            if self.fixed_shape_cache_fill or self.valid_length_cache_attention:
                for gen in self.stream_cache.gen:
                    gen._k.zero_()
                    gen._v.zero_()
            if text_ids is not None and text_positions is not None:
                net.prefill_text_stream(
                    text_ids,
                    text_positions,
                    self.stream_cache,
                    native_gqa_attention=self.native_gqa_attention,
                )
        else:
            self.cache = net.new_kv_cache()
            if text_ids is not None and text_positions is not None:
                net.prefill_text_cache(
                    text_ids,
                    text_positions,
                    self.cache,
                    native_gqa_attention=self.native_gqa_attention,
                )

        # A normal bounded cache has separate growing-history and saturated
        # execution regimes. The valid-length path instead presents one fixed
        # physical K/V shape from the first chunk and changes only a device-side
        # logical prefix length, so fill and steady state share one callable.
        self._denoise_step_fn = self._build_denoise_step_fn(dynamic=False)
        self._denoise_fill_fn = (
            self._denoise_step_fn
            if self.valid_length_cache_attention
            else (
                self._build_denoise_step_fn(dynamic=not self.fixed_shape_cache_fill)
                if self.compile_denoise
                else self._denoise_step_fn
            )
        )
        self._denoise_graph = (
            CUDAGraphWrapper(
                self._denoise_step_fn,
                warmup_iters=self.cuda_graph_warmup_iters,
                capture_error_mode="thread_local",
            )
            if self.cuda_graph_denoise
            else None
        )
        self._denoise_fill_graph = (
            CUDAGraphWrapper(
                self._denoise_fill_fn,
                warmup_iters=self.cuda_graph_warmup_iters,
                capture_error_mode="thread_local",
            )
            if (
                self.cuda_graph_denoise
                and self.fixed_shape_cache_fill
                and not self.valid_length_cache_attention
            )
            else None
        )
        # The valid-length route has no saturation handoff. Legacy routes keep
        # their historical fill boundary and switch to the steady callable there.
        self._compile_start_step = (
            0
            if self.valid_length_cache_attention
            else (self._window_frames // self.chunk_size if self.compile_denoise else 0)
        )

        # Frame 0 is seeded with a null action whose values are zeroed in the
        # cache, so the start image conditions geometry without implying motion.
        seed_frame_zero = (
            self._seed_frame_stream
            if self.cache_mode == "block"
            else self._seed_frame_cat
        )
        seed_frame_zero(
            0,
            self._init_latent,
            torch.zeros(self.tcf, self.action_dim, device=self.device),
            null_action_values=True,
        )
        self._abs_frame = 1

    def _reset_rollout_state(self) -> None:
        """Reset mutable rollout state while preserving compiled CUDA graphs."""
        self._pending_cache.clear()
        self._abs_frame = 0
        self._steps = 0
        self._ar_index = None
        self._actions = None
        self._action_domain_ids = None
        self._compiled_args = None
        self.last_timing = {}
        text_positions = (
            text_mrope_ids(self.n_text, 0.0)[0].to(self.device)
            if self.text_ids is not None and self.n_text > 0
            else None
        )
        if self.cache_mode == "block":
            assert self.stream_cache is not None
            self.stream_cache.reset()
            if self.fixed_shape_cache_fill or self.valid_length_cache_attention:
                for gen in self.stream_cache.gen:
                    gen._k.zero_()
                    gen._v.zero_()
        else:
            self.cache = self.net.new_kv_cache()
            if self.text_ids is not None and text_positions is not None:
                self.net.prefill_text_cache(
                    self.text_ids,
                    text_positions,
                    self.cache,
                    native_gqa_attention=self.native_gqa_attention,
                )

        seed_frame_zero = (
            self._seed_frame_stream
            if self.cache_mode == "block"
            else self._seed_frame_cat
        )
        seed_frame_zero(
            0,
            self._init_latent,
            torch.zeros(self.tcf, self.action_dim, device=self.device),
            null_action_values=True,
        )
        self._abs_frame = 1

    def reset_rollout(self) -> None:
        """Reset to frame zero while preserving compiled denoise graphs."""
        self.wait_for_pending_cache_update(block=True)
        self._reset_rollout_state()

    def start(self, autoregressive_index: int) -> None:
        """Start one generic diffusion-model step."""
        self._ar_index = autoregressive_index
        self.last_timing = {}
        self._step_started = time.perf_counter()
        self._cache_wait_s = self.wait_for_pending_cache_update()

    def prepare(self, action_chunk: Tensor, domain_ids: Tensor | None = None) -> None:
        """Validate controls and gather the history shared by every denoise pass."""
        rows = int(action_chunk.numel() // self.action_dim)
        if action_chunk.numel() % self.action_dim or rows == 0 or rows % self.tcf:
            raise ValueError(
                f"action chunk must contain n*{self.tcf} rows of {self.action_dim}, "
                f"got {tuple(action_chunk.shape)}"
            )
        frames = rows // self.tcf
        if frames > self.chunk_size:
            raise ValueError(
                f"action chunk requests {frames} latent frames; maximum is "
                f"{self.chunk_size}"
            )
        if self.cache_mode == "block" and frames != self.chunk_size:
            raise ValueError(
                f"cache_mode='block' requires a full {self.chunk_size}-frame chunk, "
                f"got {frames}; the ring buffer's block size is fixed at "
                "construction. Use cache_mode='cat' for partial chunks"
            )
        if domain_ids is None:
            domain_ids = torch.full(
                (rows,), self.action_domain_id, dtype=torch.long, device=self.device
            )
        elif not isinstance(domain_ids, Tensor) or domain_ids.dtype != torch.long:
            raise TypeError("action domain_ids must be an int64 Tensor")
        elif domain_ids.shape != (rows,):
            raise ValueError("action domain_ids must have one entry per action row")

        self._frames = frames
        self._actions = action_chunk.to(self.device, dtype=torch.float32).reshape(
            rows, self.action_dim
        )
        self._action_domain_ids = domain_ids.to(device=self.device).contiguous()
        self._positions_current = self._positions(self._abs_frame, frames)
        self._is_gen, self._is_action = self._flags(frames)
        self._vision_tokens = frames * self.hw
        self._compiled_args = None
        self._compiled_fill_active = False
        self._compiled_masked_fill_active = False
        self._compiled_valid_length_active = False

        if self.stream_cache is not None:
            # Reserve one block so every denoise pass overwrites the same slot;
            # the clean refresh commits it in postprocess_clean_latent().
            for gen in self.stream_cache.gen:
                gen.before_update(self._steps)
            if self.compile_denoise:
                cos, sin = self.net._mrope(self._positions_current)
                write_start = int(self.stream_cache.gen[0]._current_write_bounds()[0])
                self._compiled_fill_active = self._steps < self._compile_start_step
                history_valid_length = None
                if self.valid_length_cache_attention:
                    (
                        text_k,
                        text_v,
                        history_k,
                        history_v,
                        history_valid_length,
                    ) = self.net.gather_stream_history_valid_length(
                        self.stream_cache, write_start
                    )
                    history_mask = None
                    self._compiled_valid_length_active = True
                elif self._compiled_fill_active and self.fixed_shape_cache_fill:
                    (
                        text_k,
                        text_v,
                        history_k,
                        history_v,
                        history_mask,
                    ) = self.net.gather_stream_history_masked(
                        self.stream_cache, write_start
                    )
                    self._compiled_masked_fill_active = True
                else:
                    text_k, text_v, history_k, history_v = (
                        self.net.gather_stream_history(self.stream_cache, write_start)
                    )
                    history_mask = None
                self._compiled_args = (
                    cos,
                    sin,
                    self._actions,
                    self._action_domain_ids,
                    text_k,
                    text_v,
                    history_k,
                    history_v,
                    history_mask,
                    history_valid_length,
                )
        if self.profile:
            self._sync()
        self._denoise_started = time.perf_counter()

    def complete(self, clean_latent: Tensor) -> Tensor:
        """Refresh clean K/V before decode and finish per-step bookkeeping."""
        if self.profile:
            self._sync()
        denoised_at = time.perf_counter()
        clean = clean_latent.squeeze(0)
        assert self._actions is not None
        assert self._action_domain_ids is not None
        clean_tokens = self._to_tokens(clean.detach())
        if self.cache_mode == "block":
            cache_submit_s = self._submit_refresh(
                self._abs_frame,
                clean_tokens,
                self._actions,
                self._frames,
                self._action_domain_ids,
            )
        else:
            cache_submit_s = 0.0
            self._seed_chunk_cat(
                self._abs_frame,
                clean_tokens,
                self._actions,
                self._frames,
                self._action_domain_ids,
            )
        self._abs_frame += self._frames
        self._steps += 1
        self._prune_cache()
        if self.profile:
            self._sync()
        seeded_at = time.perf_counter()
        if self.profile:
            finished = time.perf_counter()
            self.last_timing = {
                "model_step_s": finished - self._step_started,
                "setup_s": self._denoise_started - self._step_started,
                "denoise_s": denoised_at - self._denoise_started,
                "cache_seed_prune_s": seeded_at - denoised_at,
                "cache_update_submit_s": cache_submit_s,
                "cache_update_wait_s": self._cache_wait_s,
                "output_s": finished - seeded_at,
                "cached_tokens": float(self._cached_tokens()),
                "compile_denoise_active": float(self._compiled_args is not None),
                "compile_denoise_fill_active": float(self._compiled_fill_active),
                "compile_denoise_masked_fill_active": float(
                    self._compiled_masked_fill_active
                ),
                "compile_denoise_valid_length_active": float(
                    self._compiled_valid_length_active
                ),
                "compile_denoise_static_active": float(
                    self._compiled_args is not None and not self._compiled_fill_active
                ),
                "cuda_graph_captured": float(
                    self._denoise_graph is not None
                    and getattr(self._denoise_graph, "_graph", None) is not None
                ),
                "cuda_graph_fill_captured": float(
                    self._denoise_fill_graph is not None
                    and getattr(self._denoise_fill_graph, "_graph", None) is not None
                ),
            }
        return clean_latent

    def finalize(self, autoregressive_index: int) -> None:
        """Finish generic lifecycle bookkeeping; K/V was refreshed pre-decode."""
        assert self._ar_index == autoregressive_index
        self._ar_index = None

    def _positions(self, start_frame: int, n_frames: int) -> Tensor:
        """Build interleaved ``[action, vision]`` mRoPE ids for ``n_frames``."""
        vision_ids, _ = vae_mrope_ids(
            n_frames,
            self.gh,
            self.gw,
            self._off,
            reset_spatial_indices=self.spec.vfm.mrope_reset_spatial_ids,
            fps=self.fps if self.spec.vfm.enable_fps_modulation else None,
            base_fps=self.spec.vfm.base_fps,
            temporal_compression_factor=self.tcf,
            start_frame_offset=start_frame,
        )
        action_ids = action_mrope_ids(
            n_frames, self._off, self.stride, self.tcf, start_frame=start_frame
        )
        vision_ids = vision_ids.reshape(3, n_frames, self.hw)
        action_ids = action_ids.reshape(3, n_frames, self.tcf)
        return torch.cat((action_ids, vision_ids), dim=2).reshape(3, -1).to(self.device)

    def _flags(self, n_frames: int) -> tuple[Tensor, Tensor]:
        """Return ``(is_gen, is_action)`` for the interleaved supertoken layout."""
        total = n_frames * self._per_frame_tokens
        is_gen = torch.ones(total, dtype=torch.bool, device=self.device)
        per_frame = torch.tensor(
            [True] * self.tcf + [False] * self.hw, device=self.device
        )
        return is_gen, per_frame.repeat(n_frames)

    def _build_denoise_step_fn(self, *, dynamic: bool):
        """Build one break-free velocity evaluation, compiled when requested.

        Only the velocity call is compiled; the generic scheduler owns the loop.

        The packing is written structurally rather than through the boolean-mask
        assignment :meth:`_assemble_input` uses, because Inductor lowers boolean
        assignment to ``index_put_``, which CUDA graphs cannot capture.
        """
        net = self.net
        device, dtype = self.device, net.config.dtype
        native_gqa = self.native_gqa_attention
        hidden = self.spec.llm.hidden_size
        # Block mode rejects partial chunks, so the frame count is fixed for the
        # session's lifetime and every shape below is static.
        tcf, hw, frames = self.tcf, self.hw, self.chunk_size
        to_tokens, to_logical = self._to_tokens, self._to_logical
        layers = self._layer_list()

        def denoise_step(
            latent: Tensor,
            timestep: Tensor,
            cos: Tensor,
            sin: Tensor,
            action_tokens: Tensor,
            action_domain_ids: Tensor,
            text_k: list[Tensor | None],
            text_v: list[Tensor | None],
            history_k: list[Tensor | None],
            history_v: list[Tensor | None],
            history_mask: Tensor | None,
            history_valid_length: Tensor | None,
        ) -> Tensor:
            vision = net.vae2llm(to_tokens(latent).to(device=device, dtype=dtype))
            vision = vision + net.embed_uniform_timestep(
                timestep, int(vision.shape[0])
            ).to(dtype)
            action = net.encode_action(action_tokens, domain_id=action_domain_ids)
            # One [action(tcf), vision(hw)] supertoken per frame, built by
            # concatenation so no position is ever written by index.
            h = torch.cat(
                (action.view(frames, tcf, hidden), vision.view(frames, hw, hidden)),
                dim=1,
            ).view(1, frames * (tcf + hw), hidden)
            for index, layer in enumerate(layers):
                h = layer.gen_compute(
                    h,
                    cos,
                    sin,
                    text_k[index],
                    text_v[index],
                    history_k[index],
                    history_v[index],
                    None,
                    cached_mask=history_mask,
                    cached_valid_length=history_valid_length,
                    native_gqa_attention=native_gqa,
                )
            h = net.norm_moe_gen(h)
            flow = net.llm2vae(h)[0].view(frames, tcf + hw, -1)
            return to_logical(flow[:, tcf:].reshape(frames * hw, -1), frames)

        if not self.compile_denoise:
            return denoise_step
        return torch.compile(
            denoise_step,
            # Dynamic history benefits from one reusable symbolic graph. Keep
            # it on Inductor's default mode: max-autotune is reserved for the
            # static steady-state graph, where shapes and pointers are stable.
            mode=(
                None
                if dynamic or self.compile_denoise_mode == "default"
                else self.compile_denoise_mode
            ),
            dynamic=dynamic,
        )

    def _clean_tokens(
        self,
        n_frames: int,
        vision: Tensor,
        action: Tensor,
        domain_ids: Tensor | None = None,
    ) -> Tensor:
        """Assemble ``n_frames`` of clean, timestep-free ``[action, vision]`` tokens."""
        is_gen, is_action = self._flags(n_frames)
        zeros = torch.zeros(n_frames * self.hw, device=self.device)
        return self.net._assemble_input(
            None,
            vision,
            zeros,
            is_gen,
            # Clean frames carry no timestep embedding.
            vision_noisy_mask=zeros,
            action_tokens=action,
            is_action=is_action,
            action_domain_id=self.action_domain_id
            if domain_ids is None
            else domain_ids,
        )

    def _causal_within(self, n_frames: int) -> Tensor:
        """Causal across frames, bidirectional inside each frame's supertoken."""
        frame = torch.arange(n_frames, device=self.device).repeat_interleave(
            self._per_frame_tokens
        )
        return frame[None, :] <= frame[:, None]

    def _seed_frame_cat(
        self,
        frame: int,
        clean_vision: Tensor,
        action: Tensor,
        *,
        null_action_values: bool = False,
        domain_ids: Tensor | None = None,
    ) -> None:
        """Append one clean ``[action, vision]`` supertoken to the growing cache."""
        _, is_action = self._flags(1)
        h = self._clean_tokens(1, clean_vision, action, domain_ids)
        self.net._layers_step(
            h,
            self._flags(1)[0],
            self._positions(frame, 1),
            self._cat_cache(),
            append=True,
            within_mask=self._causal_within(1),
            zero_value_mask=is_action if null_action_values else None,
            native_gqa_attention=self.native_gqa_attention,
        )

    def _seed_chunk_cat(
        self,
        start: int,
        clean_vision: Tensor,
        action: Tensor,
        n_frames: int,
        domain_ids: Tensor | None = None,
    ) -> None:
        """Append a chunk's clean K/V to the growing cache in ONE forward.

        A single causal-across-frames forward is equivalent to seeding the frames
        sequentially through a growing cache, since each frame still sees only
        the frames before it. Doing it per frame costs one full transformer pass
        each, which for a four-frame chunk is three more passes than the whole
        denoise loop saves.
        """
        self.net._layers_step(
            self._clean_tokens(n_frames, clean_vision, action, domain_ids),
            self._flags(n_frames)[0],
            self._positions(start, n_frames),
            self._cat_cache(),
            append=True,
            within_mask=self._causal_within(n_frames),
            native_gqa_attention=self.native_gqa_attention,
        )

    def _seed_frame_stream(
        self,
        frame: int,
        clean_vision: Tensor,
        action: Tensor,
        *,
        null_action_values: bool = False,
        domain_ids: Tensor | None = None,
    ) -> None:
        """Write one clean supertoken into the ring buffer's pinned sink region."""
        assert self.stream_cache is not None
        _, is_action = self._flags(1)
        h = self._clean_tokens(1, clean_vision, action, domain_ids)
        cos, sin = self.net._mrope(self._positions(frame, 1))
        within = self._causal_within(1)
        sink = int(h.shape[1])
        # Zeroing V for the null action mirrors the cat path's zero_value_mask:
        # future frames must not read motion out of the start image, while the
        # current forward still uses the raw projection.
        keep = (~is_action).view(1, 1, -1, 1)
        for index, layer in enumerate(self._layer_list()):
            h, k, v = layer.gen_compute_with_kv(
                h,
                cos,
                sin,
                self.stream_cache.text_k[index],
                self.stream_cache.text_v[index],
                None,
                None,
                within,
                native_gqa_attention=self.native_gqa_attention,
            )
            gen = self.stream_cache.gen[index]
            gen._k[:, :, :sink].copy_(k)
            gen._v[:, :, :sink].copy_(v * keep.to(v.dtype) if null_action_values else v)
            gen._n_cached = sink
            gen._prev_chunk_idx = -1
            gen._curr_chunk_idx = None

    def _stream_layers(
        self, h: Tensor, positions: Tensor, within_mask: Tensor | None
    ) -> Tensor:
        """Run the generation half of every layer against the ring buffer."""
        assert self.stream_cache is not None
        cos, sin = self.net._mrope(positions)
        for index, layer in enumerate(self._layer_list()):
            h = layer.gen_step(
                h,
                cos,
                sin,
                self.stream_cache.text_k[index],
                self.stream_cache.text_v[index],
                self.stream_cache.gen[index],
                within_mask,
                native_gqa_attention=self.native_gqa_attention,
            )
        return h

    def _submit_refresh(
        self,
        start: int,
        clean_vision: Tensor,
        action: Tensor,
        n_frames: int,
        domain_ids: Tensor,
    ) -> float:
        """Run the clean K/V write-back, on a side stream when one is configured.

        The rewrite only has to finish before the *next* step reads the cache, so
        overlapping it lets decode and presentation proceed in the meantime.

        Returns:
            Seconds spent submitting, which is not the cost of the write itself.
        """
        if self._cache_stream is None:
            self._refresh_chunk(start, clean_vision, action, n_frames, domain_ids)
            return 0.0

        submitted = time.perf_counter()
        # Held until the event fires so the allocator cannot reuse the memory the
        # side stream is still reading from.
        clean_ref, action_ref = clean_vision.detach(), action.detach()
        domain_ref = domain_ids.detach()
        ready, done = torch.cuda.Event(), torch.cuda.Event()
        torch.cuda.current_stream(self.device).record_event(ready)
        clean_ref.record_stream(self._cache_stream)
        action_ref.record_stream(self._cache_stream)
        domain_ref.record_stream(self._cache_stream)
        with torch.cuda.stream(self._cache_stream):
            self._cache_stream.wait_event(ready)
            self._refresh_chunk(start, clean_ref, action_ref, n_frames, domain_ref)
            done.record(self._cache_stream)
        self._pending_cache.append((done, (clean_ref, action_ref, domain_ref)))
        self._pending_cache = [
            entry for entry in self._pending_cache if not entry[0].query()
        ]
        return time.perf_counter() - submitted

    def wait_for_pending_cache_update(self, *, block: bool = False) -> float:
        """Order the current stream after any outstanding clean-cache write.

        Args:
            block: Also wait on the host, which a caller needs before reading the
                cache from Python rather than from a queued kernel.

        Returns:
            Seconds spent waiting.
        """
        if self._cache_stream is None or not self._pending_cache:
            return 0.0
        waited = time.perf_counter()
        current = torch.cuda.current_stream(self.device)
        current.wait_event(self._pending_cache[-1][0])
        if block:
            current.synchronize()
            self._pending_cache.clear()
        else:
            self._pending_cache = [
                entry for entry in self._pending_cache if not entry[0].query()
            ]
        return time.perf_counter() - waited

    def _refresh_chunk(
        self,
        start: int,
        clean_vision: Tensor,
        action: Tensor,
        n_frames: int,
        domain_ids: Tensor,
    ) -> None:
        """Overwrite the ring buffer's current block with finalized clean K/V."""
        assert self.stream_cache is not None
        if self.valid_length_cache_attention:
            self._refresh_chunk_valid_length(
                start, clean_vision, action, n_frames, domain_ids
            )
        elif self.cache_refresh_mode == "framewise":
            self._refresh_chunk_framewise(
                start, clean_vision, action, n_frames, domain_ids
            )
        else:
            self._stream_layers(
                self._clean_tokens(n_frames, clean_vision, action, domain_ids),
                self._positions(start, n_frames),
                self._causal_within(n_frames),
            )
        for gen in self.stream_cache.gen:
            gen.after_update(self._steps)

    def _refresh_chunk_valid_length(
        self,
        start: int,
        clean_vision: Tensor,
        action: Tensor,
        n_frames: int,
        domain_ids: Tensor,
    ) -> None:
        """Refresh clean K/V through the same fixed physical cache geometry.

        Each frame's queries attend to real history plus all tokens through that
        frame. Splitting only the attention call by frame exactly reproduces the
        block-causal mask while avoiding a new masked-SDPA shape and backend plan
        on every cache-fill step.
        """
        assert self.stream_cache is not None
        write_start = int(self.stream_cache.gen[0]._current_write_bounds()[0])
        (
            text_k,
            text_v,
            history_k,
            history_v,
            history_valid_length,
        ) = self.net.gather_stream_history_valid_length(self.stream_cache, write_start)
        h = self._clean_tokens(n_frames, clean_vision, action, domain_ids)
        cos, sin = self.net._mrope(self._positions(start, n_frames))
        current_tokens = n_frames * self._per_frame_tokens
        write_end = write_start + current_tokens
        for index, layer in enumerate(self._layer_list()):
            h, k, v = layer.gen_compute_with_kv(
                h,
                cos,
                sin,
                text_k[index],
                text_v[index],
                history_k[index],
                history_v[index],
                None,
                cached_valid_length=history_valid_length,
                causal_block_tokens=self._per_frame_tokens,
                native_gqa_attention=self.native_gqa_attention,
            )
            gen = self.stream_cache.gen[index]
            destination = gen._seq_slice(write_start, write_end)
            gen._k[destination].copy_(k)
            gen._v[destination].copy_(v)

    def _refresh_chunk_framewise(
        self,
        start: int,
        clean_vision: Tensor,
        action: Tensor,
        n_frames: int,
        domain_ids: Tensor,
    ) -> None:
        """Refresh one frame at a time, exposing only history that is already clean.

        The chunk refresh hides later frames behind a mask over a single forward.
        Here each frame instead attends to the real prefix, so no masking is
        needed and every frame sees finalized neighbours.
        """
        assert self.stream_cache is not None
        for offset in range(n_frames):
            h = self._clean_tokens(
                1,
                clean_vision[offset * self.hw : (offset + 1) * self.hw],
                action[offset * self.tcf : (offset + 1) * self.tcf],
                domain_ids[offset * self.tcf : (offset + 1) * self.tcf],
            )
            cos, sin = self.net._mrope(self._positions(start + offset, 1))
            for index, layer in enumerate(self._layer_list()):
                gen = self.stream_cache.gen[index]
                frame_start = (
                    int(gen._current_write_bounds()[0])
                    + offset * self._per_frame_tokens
                )
                frame_end = frame_start + self._per_frame_tokens
                history = gen._seq_slice(0, frame_start)
                h, k, v = layer.gen_compute_with_kv(
                    h,
                    cos,
                    sin,
                    self.stream_cache.text_k[index],
                    self.stream_cache.text_v[index],
                    gen._k[history] if frame_start > 0 else None,
                    gen._v[history] if frame_start > 0 else None,
                    None,
                    native_gqa_attention=self.native_gqa_attention,
                )
                gen._k[gen._seq_slice(frame_start, frame_end)].copy_(k)
                gen._v[gen._seq_slice(frame_start, frame_end)].copy_(v)

    def _prune_cache(self) -> None:
        """Drop the oldest frames beyond the window, keeping text and frame 0."""
        if self._window_frames <= 0 or self.cache is None:
            return
        generated = self._abs_frame - 1
        if generated <= self._window_frames:
            return
        head = self.n_text + self._per_frame_tokens
        tail = self._window_frames * self._per_frame_tokens
        for layer_cache in self.cache:
            k, v = layer_cache.get("k"), layer_cache.get("v")
            if k is None or v is None:
                continue
            layer_cache["k"] = torch.cat(
                (k[:, :, :head], k[:, :, -tail:]), dim=2
            ).contiguous()
            layer_cache["v"] = torch.cat(
                (v[:, :, :head], v[:, :, -tail:]), dim=2
            ).contiguous()

    def _to_tokens(self, logical_latent: Tensor) -> Tensor:
        """Pad the logical latent up to the transformer grid and patchify it."""
        padded = pad_latent(
            logical_latent,
            bottom=self.transformer_h_lat - self.h_lat,
            right=self.transformer_w_lat - self.w_lat,
        )
        return patchify_latent(padded, self.patch)

    def _to_logical(self, tokens: Tensor, n_frames: int) -> Tensor:
        """Unpatchify tokens and crop back to the logical content region."""
        latent = unpatchify_latent(
            tokens,
            channels=self.channels,
            grid_t=n_frames,
            grid_h=self.gh,
            grid_w=self.gw,
            patch_spatial=self.patch,
        )
        return crop_latent(latent, height=self.h_lat, width=self.w_lat)

    def _layer_list(self) -> list[_MoTLayer]:
        """The MoT layers. ``nn.ModuleList`` erases its element type."""
        return cast("list[_MoTLayer]", list(self.net.layers))

    def _cat_cache(self) -> list[dict[str, Tensor]]:
        """The growing cache, which only exists in ``cat`` mode."""
        assert self.cache is not None, "cat cache is absent in block mode"
        return self.cache

    def _sync(self) -> None:
        """Wait for queued GPU work so a timing boundary means something."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _cached_tokens(self) -> int:
        """Sequence length the next step will attend over, for diagnostics."""
        if self.stream_cache is not None:
            return self.n_text + int(self.stream_cache.gen[0]._n_cached)
        for layer_cache in self.cache or ():
            key = layer_cache.get("k")
            if key is not None:
                return int(key.shape[2])
        return 0
