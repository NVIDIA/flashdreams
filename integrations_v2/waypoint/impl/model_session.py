# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waypoint model-session binding for the shared Action2V application."""

from __future__ import annotations

import threading
from typing import Any

import torch
from action2v import Action2VStep
from torch import Tensor

from flashdreams.runtime_v2.session_desc import SessionDesc
from waypoint import WAYPOINT_1_5, WaypointControl
from waypoint.impl.pipeline import WaypointInferencePipeline


class WaypointModelSession:
    """Own one Waypoint cache and deterministic RNG stream."""

    def __init__(
        self,
        *,
        pipeline: WaypointInferencePipeline,
        pipeline_lock: threading.Lock,
        session_desc: SessionDesc,
        seed_frames: Tensor,
        seed: int,
    ) -> None:
        self.pipeline = pipeline
        self.pipeline_lock = pipeline_lock
        self.session_desc = session_desc
        self.seed = seed
        dtype = pipeline.diffusion_model.dtype
        self.seed_frames = seed_frames.to(
            device=pipeline.device, dtype=dtype
        ).contiguous()
        self.seed_pixels = _seed_pixels(self.seed_frames)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        self.initial_rng_state = generator.get_state()
        self.rng_state = self.initial_rng_state.clone()
        with pipeline_lock:
            self.cache: Any | None = pipeline.initialize_cache(
                seed_pixels=self.seed_pixels
            )

    def step(self, step_index: int, action: Any) -> Action2VStep:
        """Generate and finalize one Waypoint action."""
        if not isinstance(action, WaypointControl):
            raise TypeError("Waypoint actions must be WaypointControl values.")
        cache = self._require_cache()
        with self.pipeline_lock:
            rng = self.pipeline.diffusion_model.rng
            if rng is None:
                raise RuntimeError("Waypoint pipeline must have a deterministic seed")
            rng.set_state(self.rng_state)
            video = self.pipeline.generate(step_index, cache, action)
            stats = self.pipeline.finalize(step_index, cache)
            self.rng_state = rng.get_state()
        return Action2VStep(
            frames=_presentation_frames(video, self.session_desc),
            metrics=dict(stats or {}),
        )

    def reset(self) -> None:
        """Re-establish the seed cache and deterministic RNG stream."""
        self.rng_state = self.initial_rng_state.clone()
        with self.pipeline_lock:
            self.cache = None
            self.cache = self.pipeline.initialize_cache(seed_pixels=self.seed_pixels)

    def close(self) -> None:
        """Release the rollout cache."""
        self.cache = None

    def _require_cache(self) -> Any:
        if self.cache is None:
            raise RuntimeError("Waypoint model session is closed")
        return self.cache


def _seed_pixels(seed_frames: Tensor) -> Tensor:
    return seed_frames.add(1.0).mul(0.5).unsqueeze(0)


def _presentation_frames(video: Tensor, session_desc: SessionDesc) -> Tensor:
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError(
            "Waypoint pipeline output must have [1, T, C, H, W] layout, got "
            f"{tuple(video.shape)}"
        )
    output = video[0]
    expected = (WAYPOINT_1_5.frames_per_action, 3)
    if tuple(output.shape[:2]) != expected:
        raise ValueError(
            f"Waypoint pipeline must emit {expected[0]} RGB frames, got "
            f"{tuple(output.shape)}"
        )
    target_size = (session_desc.video_height, session_desc.video_width)
    if tuple(output.shape[-2:]) != target_size:
        raise ValueError(
            "Waypoint pipeline output spatial size must match the session: expected "
            f"{target_size[1]}x{target_size[0]}, got "
            f"{output.shape[-1]}x{output.shape[-2]}"
        )
    return output.detach().contiguous()


__all__ = ["WaypointModelSession"]
