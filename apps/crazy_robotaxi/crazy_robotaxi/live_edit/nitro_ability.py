# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Nitro pickup: a timed speed boost inside the taxi physics tick.

Unlike the weather/skin items, nitro never touches the world-model state
machines — it is pure app-side physics. The seam is the per-frame
``integrate_fn`` the rollout passes to ``sample_chunk_trajectory``
(:func:`crazy_robotaxi.driving.integrate_taxi_vehicle`): while the boost is
active, :func:`integrate_with_nitro` hands the integrator a vehicle config
with ``max_accel_mps2`` and ``max_speed_mps`` multiplied by
``nitro_boost``, the boosted max speed hard-capped at
``nitro_max_speed_mps`` so the ego stays inside the world model's manifold
(the conditioning renders the faster ego plausibly up to highway speeds;
~16 m/s is the validated comfort zone on the shipped suburb map).

Activation is INSTANT: a pickup detected in chunk N boosts the very next
sampled physics tick (chunk N+1 at the pipeline's one-chunk pickup
latency) — no chunk-boundary state-machine handshake, because there is no
model-side state to swap. The timer runs on game time (the integrator's
accumulated ``dt_s``), and a second pickup while boosted RESETS it to the
full duration — no multiplicative stacking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from loguru import logger
from omnidreams_game_engine.config import VehicleConfig
from omnidreams_game_engine.types import DriverCommand, VehicleState

from crazy_robotaxi.live_edit.config import LiveEditItemsConfig

IntegrateFn = Callable[
    [VehicleState, DriverCommand, float, VehicleConfig], VehicleState
]

_TIMER_EPSILON_S = 1.0e-9
"""Treat sub-nanosecond residue as expired (dt accumulation float drift)."""


class NitroAbility:
    """Hold the nitro timer and produce the boosted vehicle config.

    One instance per application, reset per rollout (the same lifecycle as
    the item course, so a rollout reset always starts unboosted).
    """

    def __init__(self, config: LiveEditItemsConfig) -> None:
        self._config = config
        self._remaining_s = 0.0

    @property
    def boost(self) -> float:
        """The configured speed/acceleration multiplier."""
        return self._config.nitro_boost

    @property
    def active(self) -> bool:
        """Whether the boost currently applies to physics ticks."""
        return self._remaining_s > _TIMER_EPSILON_S

    @property
    def seconds_remaining(self) -> float:
        """Game seconds of boost left (0 when inactive; HUD countdown)."""
        return self._remaining_s if self.active else 0.0

    def activate(self) -> None:
        """Start the boost; a re-pickup while boosted resets the timer."""
        self._remaining_s = self._config.nitro_duration_s
        logger.info(
            f"[live-edit] nitro boost ON x{self._config.nitro_boost:.2f} "
            f"for {self._config.nitro_duration_s:.1f}s "
            f"(max {self._config.nitro_max_speed_mps:.1f} m/s)"
        )

    def reset(self) -> None:
        """Drop any active boost (rollout reset)."""
        self._remaining_s = 0.0

    def boosted_vehicle(self, vehicle: VehicleConfig) -> VehicleConfig:
        """The vehicle config with the nitro multiplier and ceiling applied."""
        return replace(
            vehicle,
            max_accel_mps2=vehicle.max_accel_mps2 * self._config.nitro_boost,
            max_speed_mps=min(
                vehicle.max_speed_mps * self._config.nitro_boost,
                self._config.nitro_max_speed_mps,
            ),
        )

    def vehicle_for_tick(self, vehicle: VehicleConfig, dt_s: float) -> VehicleConfig:
        """Consume one physics tick; return the config the tick should use."""
        if not self.active:
            return vehicle
        boosted = self.boosted_vehicle(vehicle)
        self._remaining_s -= dt_s
        if not self.active:
            logger.info("[live-edit] nitro boost expired")
        return boosted


def integrate_with_nitro(nitro: NitroAbility, integrate_fn: IntegrateFn) -> IntegrateFn:
    """Wrap a taxi integrator so active nitro boosts each tick's vehicle."""

    def integrate(
        state: VehicleState,
        command: DriverCommand,
        dt_s: float,
        vehicle: VehicleConfig,
    ) -> VehicleState:
        return integrate_fn(state, command, dt_s, nitro.vehicle_for_tick(vehicle, dt_s))

    return integrate
