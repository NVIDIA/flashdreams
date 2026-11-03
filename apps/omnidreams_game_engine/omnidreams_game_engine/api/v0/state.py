# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoded per-frame engine state and the state decoders.

``frame.state`` on the wire is in an engine's own vocabulary (named by
``hello.state_format``). A :class:`StateDecoder` turns it into a
:class:`FrameState` in FLU metres, the frame of the scene and the renderer.
Engine-specific decoders (such as a state line in Unreal units) live with
their model host and use :mod:`~omnidreams_game_engine.api.v0.units`. This
module defines one engine-neutral vocabulary, ``flu/0``, with an exact
round trip; the stub engine and in-process engines use it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from omnidreams_game_engine.types import DriverCommand, VehicleState

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ActorState:
    """One dynamic actor's box at one frame, in FLU world metres."""

    entity_id: str
    object_type: str
    """Renderer class, such as ``Car`` or ``Pedestrian``."""
    position_world: FloatArray
    """Box centre, shape ``(3,)``."""
    orientation_xyzw: FloatArray
    """Unit quaternion, shape ``(4,)``."""
    dimensions_lwh: FloatArray
    """Length, width and height, shape ``(3,)``."""


@dataclass(frozen=True, slots=True)
class FrameState:
    """The engine's state at one model frame, decoded to FLU."""

    g: int
    tick: int
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

_VEHICLE_FIELDS = tuple(item.name for item in fields(VehicleState))
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
    """The engine-neutral ``flu/0`` vocabulary: FLU metres and radians.

    ``{"rig_to_world": [16 row-major floats], "vehicle": {every VehicleState
    field}, "command": {every DriverCommand field}, "actors": [{"id", "type",
    "position", "orientation_xyzw", "size_lwh"}], "events": [...]}``.
    Floats are written with Python's shortest round-trip repr, so
    ``decode(encode(s))`` reproduces float32 poses and float64 vehicle state
    exactly.
    """

    @property
    def state_format(self) -> str:
        return FLU_STATE_FORMAT

    def encode(self, frame: FrameState) -> dict[str, Any]:
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
                    "id": actor.entity_id,
                    "type": actor.object_type,
                    "position": [float(v) for v in actor.position_world],
                    "orientation_xyzw": [float(v) for v in actor.orientation_xyzw],
                    "size_lwh": [float(v) for v in actor.dimensions_lwh],
                }
                for actor in frame.actors
            ],
            "events": list(frame.events),
        }

    def decode(self, state: Mapping[str, Any], *, g: int, tick: int) -> FrameState:
        try:
            pose = np.asarray(state["rig_to_world"], dtype=np.float32).reshape(4, 4)
            vehicle_fields = dict(state["vehicle"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Unreadable {FLU_STATE_FORMAT} state: {error}") from error
        unknown = set(vehicle_fields) - set(_VEHICLE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown vehicle fields {sorted(unknown)}")
        vehicle_fields.setdefault("x_m", float(pose[0, 3]))
        vehicle_fields.setdefault("y_m", float(pose[1, 3]))
        vehicle_fields.setdefault("z_m", float(pose[2, 3]))
        vehicle_fields.setdefault("yaw_rad", 0.0)
        vehicle_fields.setdefault("speed_mps", 0.0)
        vehicle_fields.setdefault("steer_rad", 0.0)
        values: dict[str, Any] = {
            name: bool(value) if name == "ragdoll_active" else _optional_float(value)
            for name, value in vehicle_fields.items()
        }
        vehicle = VehicleState(**values)
        command_fields = dict(state.get("command") or {})
        command = DriverCommand(
            throttle=float(command_fields.get("throttle", 0.0)),
            brake=float(command_fields.get("brake", 0.0)),
            steer=float(command_fields.get("steer", 0.0)),
            stop=bool(command_fields.get("stop", False)),
            handbrake=bool(command_fields.get("handbrake", False)),
            reverse=bool(command_fields.get("reverse", False)),
            steer_is_direct=bool(command_fields.get("steer_is_direct", False)),
            manual_control=bool(command_fields.get("manual_control", False)),
        )
        actors = tuple(
            ActorState(
                entity_id=str(actor["id"]),
                object_type=str(actor.get("type", "Others")),
                position_world=np.asarray(actor["position"], dtype=np.float32),
                orientation_xyzw=np.asarray(
                    actor.get("orientation_xyzw", (0.0, 0.0, 0.0, 1.0)),
                    dtype=np.float32,
                ),
                dimensions_lwh=np.asarray(actor["size_lwh"], dtype=np.float32),
            )
            for actor in state.get("actors") or ()
        )
        return FrameState(
            g=g,
            tick=tick,
            rig_to_world=pose,
            vehicle=vehicle,
            command=command,
            actors=actors,
            events=tuple(str(event) for event in state.get("events") or ()),
            raw=state,
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "FLU_STATE_FORMAT",
    "ActorState",
    "FluStateCodec",
    "FrameState",
    "StateDecoder",
]
