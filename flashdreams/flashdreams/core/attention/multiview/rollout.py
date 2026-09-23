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

"""Model-independent autoregressive rollout for multi-view latent tokens."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, Protocol

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from flashdreams.core.attention.kvcache import TokenWindow
from flashdreams.core.attention.multiview.mask import AttentionPattern, AttentionScope
from flashdreams.core.attention.multiview.packing import (
    ClipGeometry,
    MemoryLayout,
    ar_chunk_plan,
    ar_chunk_range,
    build_chunk_metadata,
    control_window,
)
from flashdreams.core.attention.multiview.rope import (
    chunk_mrope_ids,
    vision_temporal_offset,
)

SDE_CHUNK_STRIDE = 1_000_003
"""Seed stride separating autoregressive chunks."""

SDE_STEP_STRIDE = 9_176
"""Seed stride separating re-noising steps within a chunk."""

ChunkMask = Tensor | BlockMask
"""Dense or block-sparse visibility mask for one chunk pass."""


class ControlSource(Protocol):
    """Provide view-major patch tokens for requested frame ranges."""

    def tokens(self, start: int, end: int) -> Tensor:
        """Return ``[V*(end-start)*S, D]`` tokens for ``[start, end)``."""


class MultiViewRolloutState(Protocol):
    """Checkpoint-specific state used by the generic rollout scheduler."""

    @property
    def memory_layout(self) -> MemoryLayout:
        """Describe the K/V memory currently visible to chunk masks."""

    @property
    def num_text_tokens(self) -> int:
        """Return the number of cached prompt tokens."""

    def top_up_control(self, needed_end: int) -> None:
        """Make control tokens through ``needed_end`` available in memory."""

    def predict_velocity(
        self,
        latent: Tensor,
        timestep: Tensor,
        positions: Tensor,
        *,
        mask: ChunkMask,
    ) -> Tensor:
        """Predict the flow for one noisy chunk pass."""

    def commit(
        self,
        denoised: Tensor,
        positions: Tensor,
        *,
        mask: ChunkMask,
        frames: tuple[int, int],
    ) -> None:
        """Replay and cache a clean chunk for later steps."""


class MultiViewRolloutModel(Protocol):
    """Network adapter required by :class:`ChunkRollout`."""

    @property
    def device(self) -> torch.device:
        """Return the inference device."""

    @property
    def dtype(self) -> torch.dtype:
        """Return the network activation dtype."""

    @property
    def token_width(self) -> int:
        """Return the width of one patchified latent token."""

    @property
    def latent_channels(self) -> int:
        """Return the number of VAE latent channels."""

    @property
    def latent_patch_size(self) -> int:
        """Return the spatial latent patch size."""

    @property
    def default_schedule(self) -> Sequence[float]:
        """Return the model's distilled sigma schedule."""

    def prepare_rollout(
        self,
        *,
        geometry: ClipGeometry,
        controls: Tensor | ControlSource,
        text_ids: Tensor,
        view_text_tokens: tuple[int, ...] | None,
        condition_tokens: Tensor | None,
        fps: float,
        history_slots: int,
        control_ranges: tuple[tuple[int, int], ...] | None,
        control_slot_frames: int | None,
        attention_pattern: AttentionPattern,
        attention_scope: AttentionScope,
        decomposed_temporal_window_seconds: float | None,
        view_offsets: bool,
        use_block_mask: bool,
    ) -> MultiViewRolloutState:
        """Prefill checkpoint-specific caches and return mutable rollout state."""


@dataclass(frozen=True)
class ChunkTrace:
    """Work and context observed by one autoregressive step."""

    start: int
    """First generated frame."""

    end: int
    """Exclusive generated-frame end."""

    denoising_passes: int
    """Number of network passes used by the sampler."""

    clean_pass: bool
    """Whether a clean replay committed this chunk to memory."""

    memory_tokens: int
    """Real cached tokens visible before this chunk was committed."""


@dataclass(frozen=True)
class Rollout:
    """Completed multi-view latent rollout."""

    latent: Tensor
    """Decoder-ready latents shaped ``[V, C, T, H, W]``."""

    tokens: Tensor
    """View-major patch tokens shaped ``[V*T*S, D]``."""

    trace: tuple[ChunkTrace, ...]
    """One trace entry per generated chunk."""


class ChunkRollout:
    """Generate a clip one autoregressive chunk at a time through a model seam."""

    def __init__(
        self,
        model: MultiViewRolloutModel,
        *,
        geometry: ClipGeometry,
        controls: Tensor | ControlSource,
        text_ids: Tensor,
        view_text_tokens: Sequence[int] | None = None,
        condition_tokens: Tensor | None = None,
        fps: float = 30.0,
        seed: int = 0,
        schedule: Sequence[float] | None = None,
        sample_type: Literal["sde", "ode"] = "sde",
        history_frames: int | None = None,
        token_frames: int | None = None,
        attention_pattern: AttentionPattern = "causal",
        attention_scope: AttentionScope = "all_views",
        decomposed_temporal_window_seconds: float | None = None,
        view_offsets: bool = True,
        use_block_mask: bool = False,
        mask_block_size: int | tuple[int, int] = 128,
    ) -> None:
        """Validate rollout geometry, allocate output, and prefill model state."""
        plan = ar_chunk_plan(geometry)
        if not plan.count:
            raise ValueError(
                f"all {geometry.frames_per_view} latent frame(s) are conditioning "
                "frames, so there is nothing to generate."
            )
        if sample_type not in {"sde", "ode"}:
            raise ValueError(
                f"sample_type must be 'sde' or 'ode'; got {sample_type!r}."
            )
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be finite and positive, got {fps}.")
        self._check_tokens(
            condition_tokens,
            frames=geometry.condition_frames,
            geometry=geometry,
            width=model.token_width,
        )

        raw_schedule = list(model.default_schedule if schedule is None else schedule)
        if raw_schedule and raw_schedule[-1] == 0.0:
            raw_schedule.pop()
        if not raw_schedule:
            raise ValueError("the denoising schedule needs at least one nonzero sigma.")
        if any(not math.isfinite(sigma) or sigma <= 0 for sigma in raw_schedule):
            raise ValueError("every denoising sigma must be finite and positive.")
        if any(current <= following for current, following in pairwise(raw_schedule)):
            raise ValueError("the denoising schedule must be strictly decreasing.")
        if history_frames is not None and history_frames < 1:
            raise ValueError("history_frames must be positive when supplied.")
        if token_frames is not None and token_frames < 1:
            raise ValueError("token_frames must be positive when supplied.")
        view_text_tokens = None if view_text_tokens is None else tuple(view_text_tokens)
        if view_text_tokens is not None:
            if len(view_text_tokens) != geometry.num_views:
                raise ValueError(
                    f"view_text_tokens has {len(view_text_tokens)} entries for "
                    f"{geometry.num_views} views."
                )
            if any(tokens < 0 for tokens in view_text_tokens):
                raise ValueError("view_text_tokens cannot contain negative counts.")
            if sum(view_text_tokens) != int(text_ids.shape[0]):
                raise ValueError(
                    f"view_text_tokens sums to {sum(view_text_tokens)}, but text_ids "
                    f"contains {text_ids.shape[0]} tokens."
                )

        committed = plan.committed_frames
        kept = committed if history_frames is None else min(history_frames, committed)
        history_slots = -(-kept // geometry.frames_per_chunk)
        window = (
            None if history_frames is None else control_window(geometry, history_slots)
        )
        streaming = bool(window and window[-1][1] < geometry.frames_per_view)
        control_ranges = tuple(window) if streaming and window else None

        self._model = model
        with torch.no_grad():
            self._state = model.prepare_rollout(
                geometry=geometry,
                controls=controls,
                text_ids=text_ids,
                view_text_tokens=view_text_tokens,
                condition_tokens=condition_tokens,
                fps=fps,
                history_slots=history_slots,
                control_ranges=control_ranges,
                control_slot_frames=(geometry.frames_per_chunk if streaming else None),
                attention_pattern=attention_pattern,
                attention_scope=attention_scope,
                decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
                view_offsets=view_offsets,
                use_block_mask=use_block_mask,
            )
        self._geometry = geometry
        self._fps = fps
        self._seed = seed
        self._schedule = raw_schedule
        self._sample_type = sample_type
        self._history_slots = history_slots
        self._view_text_tokens = view_text_tokens
        self._attention_pattern = attention_pattern
        self._attention_scope = attention_scope
        self._decomposed_temporal_window_seconds = decomposed_temporal_window_seconds
        self._view_offsets = view_offsets
        self._use_block_mask = use_block_mask
        self._mask_block_size = mask_block_size
        text_position_span = (
            self._state.num_text_tokens
            if view_text_tokens is None
            else max(view_text_tokens)
        )
        self._offset = vision_temporal_offset(text_position_span)

        minimum_token_frames = geometry.condition_frames + geometry.frames_per_chunk
        self._token_frames = (
            geometry.frames_per_view
            if token_frames is None
            else max(token_frames, minimum_token_frames)
        )
        self._tokens = TokenWindow(
            num_views=geometry.num_views,
            frames=self._token_frames,
            spatial=geometry.spatial_tokens,
            width=model.token_width,
            device=model.device,
            dtype=model.dtype,
        )
        if condition_tokens is not None:
            self._tokens.append(
                0,
                geometry.condition_frames,
                condition_tokens.to(device=model.device, dtype=model.dtype).reshape(
                    geometry.num_views,
                    geometry.condition_frames,
                    geometry.spatial_tokens,
                    model.token_width,
                ),
            )

        self._chunk_count = plan.count
        self._next = 0
        self._frame = geometry.condition_frames
        self._stopped = False
        self._keeps_trace = self._token_frames >= geometry.frames_per_view
        self._trace: list[ChunkTrace] = []

    @staticmethod
    def _check_tokens(
        tokens: Tensor | None,
        *,
        frames: int,
        geometry: ClipGeometry,
        width: int,
    ) -> None:
        if frames and tokens is None:
            raise ValueError(
                f"the geometry has {frames} conditioning frames, so "
                "condition_tokens is required."
            )
        if not frames and tokens is not None:
            raise ValueError(
                "the geometry has no conditioning frames, so condition_tokens is unused."
            )
        if tokens is None:
            return
        expected = (geometry.num_views * frames * geometry.spatial_tokens, width)
        if tuple(tokens.shape) != expected:
            raise ValueError(
                f"condition_tokens has shape {tuple(tokens.shape)}; expected {expected}."
            )

    @property
    def geometry(self) -> ClipGeometry:
        """Return this rollout's clip geometry."""
        return self._geometry

    @property
    def chunk_count(self) -> int:
        """Return the total number of chunks in the clip."""
        return self._chunk_count

    @property
    def chunks_done(self) -> int:
        """Return the number of chunks generated so far."""
        return self._next

    @property
    def warmup_chunks(self) -> int:
        """Return chunks generated before the bounded cache reaches capacity."""
        return self._history_slots

    @property
    def next_range(self) -> tuple[int, int] | None:
        """Return the next frame range, or ``None`` after completion or stop."""
        if self._stopped:
            return None
        return ar_chunk_range(self._geometry, self._frame)

    @property
    def is_finished(self) -> bool:
        """Return whether the rollout has no more chunks to generate."""
        return self.next_range is None

    def stop(self) -> None:
        """Refuse further chunks while preserving already generated frames."""
        self._stopped = True

    @torch.no_grad()
    def step(self) -> tuple[ChunkTrace, Tensor]:
        """Generate and return the next frame chunk."""
        chunk = self.next_range
        if chunk is None:
            if self._stopped:
                raise RuntimeError(
                    f"nothing left to generate after {self._next} of "
                    f"{self._chunk_count} chunk(s)."
                )
            raise RuntimeError(
                f"all {self._chunk_count} chunks are generated; call result()."
            )

        start, end = chunk
        frames = end - start
        count = self._geometry.num_views * frames * self._geometry.spatial_tokens
        self._state.top_up_control(end)
        memory = self._state.memory_layout
        positions = chunk_mrope_ids(
            self._geometry,
            chunk_start=start,
            chunk_frames=frames,
            fps=self._fps,
            temporal_offset=self._offset,
            view_offsets=self._view_offsets,
            device=self._model.device,
        )
        noisy_mask = self._mask(
            memory,
            chunk_start=start,
            chunk_frames=frames,
            pass_kind="noisy",
        )
        denoised = _fixed_step_denoise(
            lambda latent, timestep: self._state.predict_velocity(
                latent, timestep, positions, mask=noisy_mask
            ),
            _chunk_noise(
                count,
                self._model.token_width,
                seed=self._seed,
                start=start,
                device=self._model.device,
            ),
            schedule=self._schedule,
            sample_type=self._sample_type,
            sde_noise_fn=_sde_noise_fn(
                count,
                self._model.token_width,
                seed=self._seed,
                start=start,
                device=self._model.device,
            ),
        )
        self._tokens.append(
            start,
            end,
            denoised.to(self._model.dtype).reshape(
                self._geometry.num_views,
                frames,
                self._geometry.spatial_tokens,
                self._model.token_width,
            ),
        )

        commit = end < self._geometry.frames_per_view
        if commit:
            clean_mask = self._mask(
                memory,
                chunk_start=start,
                chunk_frames=frames,
                pass_kind="clean",
            )
            self._state.commit(
                denoised,
                positions,
                mask=clean_mask,
                frames=chunk,
            )

        trace = ChunkTrace(
            start=start,
            end=end,
            denoising_passes=len(self._schedule),
            clean_pass=commit,
            memory_tokens=memory.real_token_count,
        )
        if self._keeps_trace:
            self._trace.append(trace)
        self._next += 1
        self._frame = end
        return trace, self._tokens.read(start, end)

    def _mask(
        self,
        memory: MemoryLayout,
        *,
        chunk_start: int,
        chunk_frames: int,
        pass_kind: Literal["noisy", "clean", "control"],
    ) -> ChunkMask:
        metadata = build_chunk_metadata(
            self._geometry,
            memory,
            chunk_start=chunk_start,
            chunk_frames=chunk_frames,
            text_tokens=self._state.num_text_tokens,
            view_text_tokens=self._view_text_tokens,
            pass_kind=pass_kind,
            device=self._model.device,
        )
        return (
            metadata.block_mask(
                pattern=self._attention_pattern,
                scope=self._attention_scope,
                decomposed_temporal_window_seconds=(
                    self._decomposed_temporal_window_seconds
                ),
                block_size=self._mask_block_size,
            )
            if self._use_block_mask
            else metadata.mask(
                pattern=self._attention_pattern,
                scope=self._attention_scope,
                decomposed_temporal_window_seconds=(
                    self._decomposed_temporal_window_seconds
                ),
            )
        )

    def latent_for(self, start: int, end: int) -> Tensor:
        """Return decoder-ready latents for an available frame range."""
        if not 0 <= start < end <= self._geometry.frames_per_view:
            raise ValueError(
                f"[{start}, {end}) is not a frame range of a "
                f"{self._geometry.frames_per_view}-frame clip."
            )
        return _to_latent(
            self._tokens.read(start, end),
            self._geometry,
            channels=self._model.latent_channels,
            patch=self._model.latent_patch_size,
        )

    def result(self) -> Rollout:
        """Return the finished clip, refusing incomplete or evicted output."""
        if self._next < self._chunk_count:
            raise RuntimeError(
                f"{self._chunk_count - self._next} of {self._chunk_count} chunks "
                "are still to generate."
            )
        if self._token_frames < self._geometry.frames_per_view:
            raise RuntimeError(
                f"this rollout keeps {self._token_frames} of the clip's "
                f"{self._geometry.frames_per_view} frames, so there is no whole "
                "clip to return."
            )
        tokens = self._tokens.read(0, self._geometry.frames_per_view)
        return Rollout(
            latent=_to_latent(
                tokens,
                self._geometry,
                channels=self._model.latent_channels,
                patch=self._model.latent_patch_size,
            ),
            tokens=tokens.reshape(-1, self._model.token_width),
            trace=tuple(self._trace),
        )


def run_rollout(
    model: MultiViewRolloutModel,
    *,
    geometry: ClipGeometry,
    controls: Tensor | ControlSource,
    text_ids: Tensor,
    view_text_tokens: Sequence[int] | None = None,
    condition_tokens: Tensor | None = None,
    fps: float = 30.0,
    seed: int = 0,
    schedule: Sequence[float] | None = None,
    sample_type: Literal["sde", "ode"] = "sde",
    history_frames: int | None = None,
    attention_pattern: AttentionPattern = "causal",
    attention_scope: AttentionScope = "all_views",
    decomposed_temporal_window_seconds: float | None = None,
    view_offsets: bool = True,
    use_block_mask: bool = False,
    mask_block_size: int | tuple[int, int] = 128,
    on_chunk: Callable[[ChunkTrace, Tensor], None] | None = None,
) -> Rollout:
    """Build and drain a model-independent multi-view rollout."""
    rollout = ChunkRollout(
        model,
        geometry=geometry,
        controls=controls,
        text_ids=text_ids,
        view_text_tokens=view_text_tokens,
        condition_tokens=condition_tokens,
        fps=fps,
        seed=seed,
        schedule=schedule,
        sample_type=sample_type,
        history_frames=history_frames,
        attention_pattern=attention_pattern,
        attention_scope=attention_scope,
        decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
        view_offsets=view_offsets,
        use_block_mask=use_block_mask,
        mask_block_size=mask_block_size,
    )
    while not rollout.is_finished:
        trace, chunk = rollout.step()
        if on_chunk is not None:
            on_chunk(trace, chunk)
    return rollout.result()


def _fixed_step_denoise(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    noise: Tensor,
    *,
    schedule: Sequence[float],
    sample_type: Literal["sde", "ode"],
    sde_noise_fn: Callable[[int], Tensor],
) -> Tensor:
    """Run a fixed distilled schedule with deterministic per-step re-noising."""
    sigmas = [*schedule, 0.0]
    latent = noise.float()
    for step, (current, following) in enumerate(pairwise(sigmas)):
        timestep = torch.tensor(
            current * 1000.0,
            device=latent.device,
            dtype=torch.float32,
        )
        velocity = velocity_fn(latent, timestep).float()
        clean = latent - current * velocity
        if following == 0.0:
            latent = clean
        elif sample_type == "ode":
            latent = latent + (following - current) * velocity
        else:
            noise = sde_noise_fn(step).to(device=clean.device, dtype=clean.dtype)
            latent = (1.0 - following) * clean + following * noise
    return latent


def _chunk_noise(
    count: int,
    width: int,
    *,
    seed: int,
    start: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed + start)
    return torch.empty(count, width, device=device, dtype=torch.float32).normal_(
        generator=generator
    )


def _sde_noise_fn(
    count: int,
    width: int,
    *,
    seed: int,
    start: int,
    device: torch.device,
) -> Callable[[int], Tensor]:
    def noise(step: int) -> Tensor:
        step_seed = seed + start * SDE_CHUNK_STRIDE + step * SDE_STEP_STRIDE
        generator = torch.Generator(device=device).manual_seed(step_seed)
        return torch.empty(count, width, device=device, dtype=torch.float32).normal_(
            generator=generator
        )

    return noise


def _to_latent(
    tokens: Tensor,
    geometry: ClipGeometry,
    *,
    channels: int,
    patch: int,
) -> Tensor:
    """Unpatchify ``[V, T, S, D]`` tokens into ``[V, C, T, H, W]``."""
    views, frames, spatial, width = tokens.shape
    expected_spatial = geometry.patch_h * geometry.patch_w
    expected_width = patch * patch * channels
    if spatial != expected_spatial or width != expected_width:
        raise ValueError(
            f"token layout has spatial/width {(spatial, width)}; expected "
            f"{(expected_spatial, expected_width)}."
        )
    latent = tokens.reshape(
        views,
        frames,
        geometry.patch_h,
        geometry.patch_w,
        patch,
        patch,
        channels,
    )
    return latent.permute(0, 6, 1, 2, 4, 3, 5).reshape(
        views,
        channels,
        frames,
        geometry.patch_h * patch,
        geometry.patch_w * patch,
    )


__all__ = [
    "ChunkMask",
    "ChunkRollout",
    "ChunkTrace",
    "ControlSource",
    "MultiViewRolloutModel",
    "MultiViewRolloutState",
    "Rollout",
    "run_rollout",
]
