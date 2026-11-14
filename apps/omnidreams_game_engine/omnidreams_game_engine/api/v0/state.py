# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoded per-frame engine state and the ``flu/0`` state vocabulary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from omnidreams_game_engine.types import DriverCommand, VehicleState

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ActorState:
    """One dynamic actor's box at one frame, in FLU world meters."""

    entity_id: str
    object_type: str
    """Renderer class, such as ``Car`` or ``Pedestrian``."""

    position_world: FloatArray
    """Box center, shape ``(3,)``."""

    orientation_xyzw: FloatArray
    """Unit quaternion, shape ``(4,)``."""

    dimensions_lwh: FloatArray
    """Length, width and height, shape ``(3,)``."""


@dataclass(frozen=True, slots=True)
class FrameState:
    """The engine's state at one model frame, decoded to FLU."""

    g: int
    """Model frame index within the epoch."""

    tick: int
    """Engine tick of the state."""

    rig_to_world: FloatArray
    """4x4 pose the condition renderer draws from (``rig_poses_world``)."""

    vehicle: VehicleState
    command: DriverCommand = field(default_factory=DriverCommand)
    """Controls the engine applied on this frame."""

    actors: tuple[ActorState, ...] = ()
    events: tuple[str, ...] = ()
    raw: Mapping[str, Any] | None = None
    """The wire state, kept for HUD, debug views and recordings."""


class StateDecoder(Protocol):
    """Turn one wire ``frame.state`` into a :class:`FrameState`."""

    @property
    def state_format(self) -> str:
        """Vocabulary this decoder reads (compared with ``hello.state_format``)."""
        ...

    def decode(self, state: Mapping[str, Any], *, g: int, tick: int) -> FrameState:
        """Decode one state; raise ``ValueError`` on an unreadable state."""
        ...


FLU_STATE_FORMAT = "flu/0"
"""``hello.state_format`` of the engine-neutral vocabulary of :class:`FluStateCodec`."""

_VEHICLE_FIELDS = (
    "x_m",
    "y_m",
    "z_m",
    "yaw_rad",
    "speed_mps",
    "steer_rad",
    "pitch_rad",
    "roll_rad",
    "velocity_x_mps",
    "velocity_y_mps",
    "yaw_rate_radps",
    "suspension_pitch_rad",
    "suspension_roll_rad",
    "suspension_pitch_rate_radps",
    "suspension_roll_rate_radps",
    "ragdoll_active",
)
_COMMAND_FIELDS = (
    "throttle",
    "brake",
    "steer",
    "stop",
    "handbrake",
    "reverse",
    "steer_is_direct",
    "manual_control",
)


class FluStateCodec:
    """The engine-neutral ``flu/0`` vocabulary: FLU meters and radians.

    ``{"rig_to_world": [16 row-major floats], "vehicle": {VehicleState
    fields}, "command": {DriverCommand fields}, "actors": [{ActorState
    fields}], "events": [...]}``. A missing vehicle position is the rig
    position; other missing fields take their defaults, and unknown keys are
    ignored. Floats are written with Python's shortest round-trip repr, so
    ``decode(encode(s))`` reproduces float32 poses and float64 vehicle state
    exactly.
    """

    @property
    def state_format(self) -> str:
        return FLU_STATE_FORMAT

    def encode(self, frame: FrameState) -> dict[str, Any]:
        """Return the wire state of ``frame``."""
        vehicle = {
            name: getattr(frame.vehicle, name)
            for name in _VEHICLE_FIELDS
            if getattr(frame.vehicle, name) is not None
        }
        command = {name: getattr(frame.command, name) for name in _COMMAND_FIELDS}
        return {
            "rig_to_world": [
                float(v) for v in np.asarray(frame.rig_to_world).reshape(16)
            ],
            "vehicle": vehicle,
            "command": command,
            "actors": [
                {
                    "entity_id": actor.entity_id,
                    "object_type": actor.object_type,
                    "position_world": [float(v) for v in actor.position_world],
                    "orientation_xyzw": [float(v) for v in actor.orientation_xyzw],
                    "dimensions_lwh": [float(v) for v in actor.dimensions_lwh],
                }
                for actor in frame.actors
            ],
            "events": list(frame.events),
        }

    def decode(self, state: Mapping[str, Any], *, g: int, tick: int) -> FrameState:
        try:
            pose = _floats(state["rig_to_world"], (4, 4))
            vehicle: dict[str, Any] = {
                name: bool(value) if name == "ragdoll_active" else float(value)
                for name, value in dict(state["vehicle"]).items()
                if name in _VEHICLE_FIELDS and value is not None
            }
            vehicle.setdefault("x_m", float(pose[0, 3]))
            vehicle.setdefault("y_m", float(pose[1, 3]))
            vehicle.setdefault("z_m", float(pose[2, 3]))
            vehicle.setdefault("yaw_rad", 0.0)
            vehicle.setdefault("speed_mps", 0.0)
            vehicle.setdefault("steer_rad", 0.0)
            command = dict(state.get("command") or {})
            return FrameState(
                g=g,
                tick=tick,
                rig_to_world=pose,
                vehicle=VehicleState(**vehicle),
                command=DriverCommand(
                    throttle=float(command.get("throttle", 0.0)),
                    brake=float(command.get("brake", 0.0)),
                    steer=float(command.get("steer", 0.0)),
                    stop=bool(command.get("stop", False)),
                    handbrake=bool(command.get("handbrake", False)),
                    reverse=bool(command.get("reverse", False)),
                    steer_is_direct=bool(command.get("steer_is_direct", False)),
                    manual_control=bool(command.get("manual_control", False)),
                ),
                actors=tuple(
                    ActorState(
                        entity_id=str(actor["entity_id"]),
                        object_type=str(actor.get("object_type", "Others")),
                        position_world=_floats(actor["position_world"], (3,)),
                        orientation_xyzw=_floats(
                            actor.get("orientation_xyzw", (0.0, 0.0, 0.0, 1.0)), (4,)
                        ),
                        dimensions_lwh=_floats(actor["dimensions_lwh"], (3,)),
                    )
                    for actor in state.get("actors") or ()
                ),
                events=tuple(str(event) for event in state.get("events") or ()),
                raw=state,
            )
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"Unreadable {FLU_STATE_FORMAT} state: {error!r}"
            ) from error


def _floats(values: Any, shape: tuple[int, ...]) -> FloatArray:
    return np.asarray(values, dtype=np.float32).reshape(shape)


__all__ = [
    "FLU_STATE_FORMAT",
    "ActorState",
    "FluStateCodec",
    "FrameState",
    "StateDecoder",
]
