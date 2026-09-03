# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-action control passthrough for the Waypoint pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)
from waypoint.impl.controls import WaypointControl


@dataclass(kw_only=True)
class WaypointControlEncoderConfig(EncoderConfig):
    """Config for the Waypoint per-action control passthrough."""

    _target: type["WaypointControlEncoder"] = field(
        default_factory=lambda: WaypointControlEncoder
    )


class WaypointControlEncoder(StreamingEncoder[StreamingEncoderCache]):
    """Carry one public Waypoint control event into the diffusion model.

    The learned control embedding lives inside the checkpoint-compatible DiT.
    This streaming encoder exists only because FlashDreams reserves the
    pipeline encoder slot for per-action user input.
    """

    def initialize_autoregressive_cache(self) -> StreamingEncoderCache:
        """Return empty state because public controls have no temporal preprocessing."""
        return StreamingEncoderCache()

    def forward(
        self,
        input: WaypointControl,
        autoregressive_index: int = 0,
        cache: StreamingEncoderCache | None = None,
    ) -> WaypointControl:
        """Validate and return the control event unchanged.

        Args:
            input: Public keyboard, mouse, and wheel input for this action.
            autoregressive_index: Action index; accepted for the streaming interface.
            cache: Empty passthrough cache; accepted for the streaming interface.

        Returns:
            The same :class:`WaypointControl` instance.

        Raises:
            TypeError: ``input`` is not a public Waypoint control event.
        """
        del autoregressive_index, cache
        if not isinstance(input, WaypointControl):
            raise TypeError(
                f"Waypoint control encoder requires WaypointControl, got {type(input)}"
            )
        return input
