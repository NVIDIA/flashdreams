# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Taxi-only presentation alignment for causal world-model frames."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from omnidreams_game_engine.types import PresentedFrame


class CausalFrameAlignmentPresenter:
    """Pair generated RGB with the state that causally produced it.

    Omnidreams output-frame motion follows the preceding HD-map condition
    frame. Delaying the synchronized map, pose, and application metadata by
    one presented frame keeps overlays and BEV aligned with generated RGB.
    """

    def __init__(self, presenter: Any) -> None:
        self._presenter = presenter
        self._previous_source: PresentedFrame | None = None
        self._last_identity: tuple[int, int] | None = None
        self._last_aligned: PresentedFrame | None = None

    def prepare_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        """Prefetch the incoming frame without advancing alignment state."""
        prepare = getattr(self._presenter, "prepare_frame", None)
        if callable(prepare):
            prepare(frame, view_mode=view_mode)

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        """Present generated RGB with its causally matching synchronized data."""
        self._presenter.present_frame(
            self._aligned_frame(frame),
            view_mode=view_mode,
        )

    def acknowledge_scene_change(self, scene_path: object, variant: str) -> Any:
        """Clear buffered state before forwarding a scene change."""
        self._reset()
        return self._presenter.acknowledge_scene_change(scene_path, variant)

    def close(self) -> None:
        """Close the wrapped presenter."""
        self._presenter.close()

    def _aligned_frame(self, frame: PresentedFrame) -> PresentedFrame:
        identity = (id(frame), int(frame.timestamp_us))
        if identity == self._last_identity and self._last_aligned is not None:
            return self._last_aligned
        if (
            frame.model_rgb_host_uint8 is None
            or frame.vehicle_state is None
            or frame.rig_to_world is None
        ):
            self._reset()
            return frame

        previous = self._previous_source
        if previous is None or frame.timestamp_us <= previous.timestamp_us:
            aligned = frame
        else:
            aligned = replace(
                previous,
                model_rgb_host_uint8=frame.model_rgb_host_uint8,
                status_message=frame.status_message,
            )
        self._previous_source = frame
        self._last_identity = identity
        self._last_aligned = aligned
        return aligned

    def _reset(self) -> None:
        self._previous_source = None
        self._last_identity = None
        self._last_aligned = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._presenter, name)
