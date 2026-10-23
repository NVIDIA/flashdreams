# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional controller shaping for calibrated wrist targets.

The shaper validates wrist-proxy frames and their calibrated counterparts,
then integrates accepted motion into held absolute wrist targets.

This module deliberately depends only on NumPy and the Python standard library.
It is imported both as part of ``handdreams`` and as a sibling
module by the isolated calibration process.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np

if __package__:
    from handdreams.controllers.isaac_teleop_calibration import (
        quaternion_to_matrix,
        rotation_to_quaternion,
    )
else:
    from isaac_teleop_calibration import (  # ty: ignore[unresolved-import]
        quaternion_to_matrix,
        rotation_to_quaternion,
    )

SIDES = ("left", "right")

HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION = "handdreams-absolute-command-target-v1"
"""Participant-frame held wrist targets with endpoint-aligned ``gripper_open``."""

DEFAULT_MAX_POSITION_STEP_M: float | None = 0.5
DEFAULT_GRIPPER_SLEW_PER_ACTION = 0.25

_MAX_DISCONTINUITY_INTERVAL_NS = 100_000_000
_CACHE_LIMIT = 512

_PoseTuple = tuple[float, ...]


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _unit_interval_mapping(value: object, *, name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain left and right values")
    source = cast(Mapping[str, object], value)
    result: dict[str, float] = {}
    for side in SIDES:
        if side not in source:
            raise ValueError(f"{name} is missing {side}")
        item = _finite_number(source[side], name=f"{side} {name}")
        if not 0.0 <= item <= 1.0:
            raise ValueError(f"{side} {name} must be in [0, 1]")
        result[side] = item
    return MappingProxyType(result)


def _optional_positive(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class ControllerControlConfig:
    """Immutable session configuration for :class:`ControllerControlShaper`.

    ``gripper_slew_per_action`` travels with this boundary configuration, but
    is intentionally not applied to controller samples.  Slew is defined per
    model-consumed action row; applying it here would make its rate depend on
    source sampling and would apply it repeatedly to overlapping windows.
    Output ``gripper_open`` is therefore the desired, unslewed endpoint value.
    """

    scene_gripper_open: Mapping[str, float]
    calibration_trigger: Mapping[str, float]
    grip_threshold: float = 0.0
    stale_ms: float | None = None
    max_position_step_m: float | None = DEFAULT_MAX_POSITION_STEP_M
    input_position_gain: float = 1.0
    gripper_slew_per_action: float = DEFAULT_GRIPPER_SLEW_PER_ACTION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scene_gripper_open",
            _unit_interval_mapping(
                self.scene_gripper_open, name="scene gripper openness"
            ),
        )
        object.__setattr__(
            self,
            "calibration_trigger",
            _unit_interval_mapping(
                self.calibration_trigger, name="calibration trigger"
            ),
        )
        grip = _finite_number(self.grip_threshold, name="grip threshold")
        if not 0.0 <= grip <= 1.0:
            raise ValueError("grip threshold must be in [0, 1]")
        object.__setattr__(self, "grip_threshold", grip)
        object.__setattr__(
            self,
            "stale_ms",
            _optional_positive(self.stale_ms, name="stale timeout"),
        )
        object.__setattr__(
            self,
            "max_position_step_m",
            _optional_positive(self.max_position_step_m, name="maximum position step"),
        )
        gain = _finite_number(self.input_position_gain, name="input position gain")
        if gain <= 0.0:
            raise ValueError("input position gain must be positive")
        object.__setattr__(self, "input_position_gain", gain)
        slew = _finite_number(
            self.gripper_slew_per_action, name="gripper slew per action"
        )
        if slew < 0.0:
            raise ValueError("gripper slew per action must be non-negative")
        object.__setattr__(self, "gripper_slew_per_action", slew)

    def to_dict(self) -> dict[str, Any]:
        """Return a normalized JSON-compatible configuration mapping."""
        return {
            "scene_gripper_open": dict(self.scene_gripper_open),
            "calibration_trigger": dict(self.calibration_trigger),
            "grip_threshold": self.grip_threshold,
            "stale_ms": self.stale_ms,
            "max_position_step_m": self.max_position_step_m,
            "input_position_gain": self.input_position_gain,
            "gripper_slew_per_action": self.gripper_slew_per_action,
        }

    def to_payload(self) -> dict[str, Any]:
        """Alias used by the isolated JSON protocol."""
        return self.to_dict()

    @classmethod
    def from_payload(cls, payload: object) -> ControllerControlConfig:
        """Validate a JSON-decoded control configuration."""
        if not isinstance(payload, Mapping):
            raise ValueError("controller control configuration must be an object")
        try:
            return cls(**dict(payload))
        except TypeError as error:
            raise ValueError(
                f"invalid controller control configuration: {error}"
            ) from error


@dataclass(frozen=True, slots=True)
class _SideSnapshot:
    calibrated_anchor: _PoseTuple | None = None
    command_pose: _PoseTuple | None = None
    previous_grip: float = 0.0
    gripper_open: float = 1.0


@dataclass(frozen=True, slots=True)
class _SideOutcome:
    command_pose: _PoseTuple | None
    gripper_open: float


@dataclass(frozen=True, slots=True)
class _CachedSample:
    capture_ns: int
    input_digest: str
    outcomes: tuple[_SideOutcome, _SideOutcome]


@dataclass(frozen=True, slots=True)
class ControllerControlSnapshot:
    """Immutable state at one transaction boundary."""

    config: ControllerControlConfig | None
    generation: int
    last_capture_ns: int | None
    token: str
    _sides: tuple[_SideSnapshot, _SideSnapshot] = field(repr=False)
    _cache: tuple[_CachedSample, ...] = field(repr=False)

    @property
    def configured(self) -> bool:
        """Return whether this snapshot can shape controller windows."""
        return self.config is not None


class ControllerControlCandidate:
    """Prepared output and next state that can be committed exactly once.

    The candidate itself remains safe to cache after commit.  ``frames``
    returns a new deep copy on every access, preventing a caller from changing
    the transaction or another caller's response in place.
    """

    __slots__ = ("base_token", "token", "snapshot", "_frames")

    def __init__(
        self,
        *,
        base_token: str,
        token: str,
        snapshot: ControllerControlSnapshot,
        frames: Sequence[Mapping[str, Any]],
    ) -> None:
        self.base_token = base_token
        self.token = token
        self.snapshot = snapshot
        self._frames = tuple(deepcopy(dict(frame)) for frame in frames)

    @property
    def frames(self) -> list[dict[str, Any]]:
        """Return independent copies of the prepared model-facing frames."""
        return deepcopy(list(self._frames))


def _pose_tuple(pose: np.ndarray | None) -> _PoseTuple | None:
    if pose is None:
        return None
    return tuple(float(value) for value in pose.reshape(-1))


def _pose_array(pose: _PoseTuple | None) -> np.ndarray | None:
    if pose is None:
        return None
    return np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()


def _project_rotation(value: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(value)
    result = left @ right
    if np.linalg.det(result) < 0.0:
        left[:, -1] *= -1.0
        result = left @ right
    return result


def _matrix_to_quaternion(value: np.ndarray) -> list[float]:
    """Project a rotation and return its serialized XYZW quaternion."""
    matrix = _project_rotation(np.asarray(value, dtype=np.float64))
    return rotation_to_quaternion(matrix).tolist()


def _controller_pose(payload: object, *, name: str) -> np.ndarray | None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} controller must be an object")
    controller = cast(Mapping[str, Any], payload)
    if not bool(controller.get("tracked", True)):
        return None
    position = np.asarray(controller.get("position"), dtype=np.float64).reshape(-1)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"{name} controller position must contain 3 finite values")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_to_matrix(
        controller.get("orientation"), name=f"{name} controller orientation"
    )
    pose[:3, 3] = position
    return pose


def _analog(payload: Mapping[str, Any], key: str) -> float:
    result = _finite_number(payload.get(key, 0.0), name=f"controller {key}")
    return float(np.clip(result, 0.0, 1.0))


def _short_interval(previous_ns: int | None, current_ns: int) -> bool:
    return previous_ns is not None and (
        0 < current_ns - previous_ns <= _MAX_DISCONTINUITY_INTERVAL_NS
    )


def _per_hand_discontinuity(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    config: ControllerControlConfig,
    previous_ns: int | None,
    current_ns: int,
) -> bool:
    if not _short_interval(previous_ns, current_ns):
        return False
    if config.max_position_step_m is not None:
        physical_step = float(
            np.linalg.norm(current[:3, 3] - previous[:3, 3])
            / config.input_position_gain
        )
        if physical_step > config.max_position_step_m:
            return True
    return False


def _config_payload(config: ControllerControlConfig | None) -> object:
    return None if config is None else config.to_dict()


def _side_payload(side: _SideSnapshot) -> dict[str, object]:
    return {
        "calibrated_anchor": side.calibrated_anchor,
        "command_pose": side.command_pose,
        "previous_grip": side.previous_grip,
        "gripper_open": side.gripper_open,
    }


def _snapshot_token(
    *,
    config: ControllerControlConfig | None,
    generation: int,
    last_capture_ns: int | None,
    sides: tuple[_SideSnapshot, _SideSnapshot],
    cache: tuple[_CachedSample, ...],
) -> str:
    payload = {
        "config": _config_payload(config),
        "generation": generation,
        "last_capture_ns": last_capture_ns,
        "sides": {
            side: _side_payload(state) for side, state in zip(SIDES, sides, strict=True)
        },
        "cache": [
            {
                "capture_ns": sample.capture_ns,
                "input_digest": sample.input_digest,
                "outcomes": [
                    {
                        "command_pose": outcome.command_pose,
                        "gripper_open": outcome.gripper_open,
                    }
                    for outcome in sample.outcomes
                ],
            }
            for sample in cache
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _make_snapshot(
    *,
    config: ControllerControlConfig | None,
    generation: int,
    last_capture_ns: int | None,
    sides: tuple[_SideSnapshot, _SideSnapshot],
    cache: tuple[_CachedSample, ...],
) -> ControllerControlSnapshot:
    return ControllerControlSnapshot(
        config=config,
        generation=generation,
        last_capture_ns=last_capture_ns,
        token=_snapshot_token(
            config=config,
            generation=generation,
            last_capture_ns=last_capture_ns,
            sides=sides,
            cache=cache,
        ),
        _sides=sides,
        _cache=cache,
    )


def _input_digest(
    raw_frame: Mapping[str, Any], calibrated_frame: Mapping[str, Any]
) -> str:
    def controller(payload: object) -> object:
        if not isinstance(payload, Mapping):
            return None
        source = cast(Mapping[str, Any], payload)
        result: dict[str, object] = {"tracked": bool(source.get("tracked", True))}
        for key in ("position", "orientation"):
            value = source.get(key)
            if value is not None:
                array = np.asarray(value, dtype=np.float64).reshape(-1)
                result[key] = [float(item) for item in array]
        for key in ("trigger", "grip"):
            result[key] = _analog(source, key)
        return result

    payload = {
        "raw": {side: controller(raw_frame.get(side)) for side in SIDES},
        "calibrated": {side: controller(calibrated_frame.get(side)) for side in SIDES},
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _capture_ns(frame: Mapping[str, Any], *, label: str) -> int:
    value = frame.get("client_capture_ns")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} client_capture_ns must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} client_capture_ns must be non-negative")
    return result


def _outcome_frame(
    calibrated_frame: Mapping[str, Any], outcomes: tuple[_SideOutcome, _SideOutcome]
) -> dict[str, Any]:
    result = deepcopy(dict(calibrated_frame))
    for side, outcome in zip(SIDES, outcomes, strict=True):
        payload = calibrated_frame.get(side)
        if not isinstance(payload, Mapping):
            raise ValueError(f"calibrated {side} controller must be an object")
        adjusted = deepcopy(dict(payload))
        pose = _pose_array(outcome.command_pose)
        if pose is None:
            raise RuntimeError(f"{side} controller has no tracked command pose to hold")
        adjusted["position"] = pose[:3, 3].astype(np.float32).tolist()
        adjusted["orientation"] = _matrix_to_quaternion(pose[:3, :3])
        adjusted["gripper_open"] = outcome.gripper_open
        result[side] = adjusted
    return result


class ControllerControlShaper:
    """Integrate controller motion into held, model-facing wrist targets.

    ``prepare`` is non-mutating.  A caller can send its candidate to another
    process, then call ``commit`` only after that process accepts the window.
    Commit compares the candidate's base token with the current snapshot, so a
    concurrent or stale transaction cannot silently overwrite newer state.
    """

    def __init__(self, config: ControllerControlConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._state = _make_snapshot(
            config=None,
            generation=0,
            last_capture_ns=None,
            sides=(_SideSnapshot(), _SideSnapshot()),
            cache=(),
        )
        if config is not None:
            self.configure(config)

    @property
    def active(self) -> bool:
        """Return whether a control configuration is installed."""
        with self._lock:
            return self._state.config is not None

    def configure(
        self, config: ControllerControlConfig | Mapping[str, Any]
    ) -> ControllerControlSnapshot:
        """Install a validated configuration and reset all stream state."""
        normalized = (
            config
            if isinstance(config, ControllerControlConfig)
            else ControllerControlConfig(**dict(config))
        )
        sides = tuple(
            _SideSnapshot(gripper_open=float(normalized.scene_gripper_open[side]))
            for side in SIDES
        )
        with self._lock:
            self._state = _make_snapshot(
                config=normalized,
                generation=self._state.generation + 1,
                last_capture_ns=None,
                sides=cast(tuple[_SideSnapshot, _SideSnapshot], sides),
                cache=(),
            )
            return self._state

    def reset(self) -> ControllerControlSnapshot:
        """Remove the active configuration and all accumulated control state."""
        with self._lock:
            self._state = _make_snapshot(
                config=None,
                generation=self._state.generation + 1,
                last_capture_ns=None,
                sides=(_SideSnapshot(), _SideSnapshot()),
                cache=(),
            )
            return self._state

    def snapshot(self) -> ControllerControlSnapshot:
        """Return the current immutable transaction snapshot."""
        with self._lock:
            return self._state

    def prepare(
        self,
        raw_frames: Sequence[Mapping[str, Any]],
        calibrated_frames: Sequence[Mapping[str, Any]],
    ) -> ControllerControlCandidate:
        """Prepare a complete ordered window without changing live state."""
        base = self.snapshot()
        config = base.config
        if config is None:
            raise RuntimeError("controller control shaping is not configured")
        if isinstance(raw_frames, (str, bytes)) or not isinstance(raw_frames, Sequence):
            raise ValueError("raw controller frames must be a sequence")
        if isinstance(calibrated_frames, (str, bytes)) or not isinstance(
            calibrated_frames, Sequence
        ):
            raise ValueError("calibrated controller frames must be a sequence")
        if len(raw_frames) != len(calibrated_frames):
            raise ValueError("raw and calibrated controller windows must align")
        if not raw_frames:
            raise ValueError("controller windows must not be empty")
        if any(not isinstance(frame, Mapping) for frame in raw_frames):
            raise ValueError("each raw controller frame must be an object")
        if any(not isinstance(frame, Mapping) for frame in calibrated_frames):
            raise ValueError("each calibrated controller frame must be an object")

        pairs: list[tuple[int, Mapping[str, Any], Mapping[str, Any], str]] = []
        previous_window_ns: int | None = None
        for raw, calibrated in zip(raw_frames, calibrated_frames, strict=True):
            raw_ns = _capture_ns(raw, label="raw")
            calibrated_ns = _capture_ns(calibrated, label="calibrated")
            if raw_ns != calibrated_ns:
                raise ValueError("raw and calibrated capture times must match")
            if previous_window_ns is not None and raw_ns < previous_window_ns:
                raise ValueError("controller window capture times must not decrease")
            previous_window_ns = raw_ns
            # Validate both sides even for cached samples.  A malformed retry
            # must not acquire authority merely by reusing a capture timestamp.
            for side in SIDES:
                raw_pose = _controller_pose(raw.get(side), name=f"raw {side}")
                calibrated_pose = _controller_pose(
                    calibrated.get(side), name=f"calibrated {side}"
                )
                if (raw_pose is None) != (calibrated_pose is None):
                    raise ValueError(
                        f"raw and calibrated {side} tracking state must match"
                    )
            pairs.append((raw_ns, raw, calibrated, _input_digest(raw, calibrated)))

        side_states = {side: base._sides[index] for index, side in enumerate(SIDES)}
        cache: OrderedDict[int, _CachedSample] = OrderedDict(
            (sample.capture_ns, sample) for sample in base._cache
        )
        outputs: list[dict[str, Any]] = []
        last_capture_ns = base.last_capture_ns
        new_samples = 0
        request_digests: list[tuple[int, str]] = []

        for capture_ns, raw, calibrated, digest in pairs:
            request_digests.append((capture_ns, digest))
            cached = cache.get(capture_ns)
            if cached is not None:
                if cached.input_digest != digest:
                    raise ValueError(
                        "controller sample reuses client_capture_ns with different input"
                    )
                outputs.append(_outcome_frame(calibrated, cached.outcomes))
                continue
            if last_capture_ns is not None and capture_ns <= last_capture_ns:
                # A jittered nearest-neighbour window can expose an older raw
                # sample for the first time. The server discards rows at or
                # before its committed capture time, so hold the current output
                # without rewinding state or making the stale sample authoritative.
                held_outcomes = (
                    _SideOutcome(
                        side_states["left"].command_pose,
                        side_states["left"].gripper_open,
                    ),
                    _SideOutcome(
                        side_states["right"].command_pose,
                        side_states["right"].gripper_open,
                    ),
                )
                outputs.append(_outcome_frame(calibrated, held_outcomes))
                continue

            raw_poses = {
                side: _controller_pose(raw.get(side), name=f"raw {side}")
                for side in SIDES
            }
            calibrated_poses = {
                side: _controller_pose(calibrated.get(side), name=f"calibrated {side}")
                for side in SIDES
            }
            stale = bool(
                last_capture_ns is not None
                and config.stale_ms is not None
                and capture_ns - last_capture_ns > int(config.stale_ms * 1e6)
            )
            next_states: dict[str, _SideSnapshot] = {}
            outcomes: list[_SideOutcome] = []

            for side in SIDES:
                raw_payload = cast(Mapping[str, Any], raw[side])
                tracked = raw_poses[side] is not None
                trigger = _analog(raw_payload, "trigger")
                current_grip = _analog(raw_payload, "grip")
                previous = side_states[side]
                gripper_open = previous.gripper_open
                if tracked:
                    gripper_open = float(
                        np.clip(
                            config.scene_gripper_open[side]
                            - (trigger - config.calibration_trigger[side]),
                            0.0,
                            1.0,
                        )
                    )

                previous_calibrated = _pose_array(previous.calibrated_anchor)
                current_calibrated = calibrated_poses[side]
                discontinuous = bool(
                    current_calibrated is not None
                    and previous_calibrated is not None
                    and _per_hand_discontinuity(
                        previous_calibrated,
                        current_calibrated,
                        config=config,
                        previous_ns=last_capture_ns,
                        current_ns=capture_ns,
                    )
                )
                command = _pose_array(previous.command_pose)
                if command is None and current_calibrated is not None:
                    command = current_calibrated.copy()
                elif (
                    command is not None
                    and current_calibrated is not None
                    and previous_calibrated is not None
                    and not stale
                    and not discontinuous
                    and previous.previous_grip >= config.grip_threshold
                    and current_grip >= config.grip_threshold
                ):
                    delta = np.linalg.inv(previous_calibrated) @ current_calibrated
                    command = command @ delta
                    command[:3, :3] = _project_rotation(command[:3, :3])
                    command[3] = (0.0, 0.0, 0.0, 1.0)

                keep_anchor = tracked and not stale
                next_state = _SideSnapshot(
                    calibrated_anchor=(
                        _pose_tuple(current_calibrated) if keep_anchor else None
                    ),
                    command_pose=_pose_tuple(command),
                    previous_grip=current_grip,
                    gripper_open=gripper_open,
                )
                next_states[side] = next_state
                outcomes.append(
                    _SideOutcome(
                        command_pose=next_state.command_pose,
                        gripper_open=gripper_open,
                    )
                )

            side_states = next_states
            outcome_tuple = cast(tuple[_SideOutcome, _SideOutcome], tuple(outcomes))
            sample = _CachedSample(capture_ns, digest, outcome_tuple)
            cache[capture_ns] = sample
            cache.move_to_end(capture_ns)
            while len(cache) > _CACHE_LIMIT:
                cache.popitem(last=False)
            outputs.append(_outcome_frame(calibrated, outcome_tuple))
            last_capture_ns = capture_ns
            new_samples += 1

        next_snapshot = _make_snapshot(
            config=config,
            generation=base.generation + (1 if new_samples else 0),
            last_capture_ns=last_capture_ns,
            sides=cast(
                tuple[_SideSnapshot, _SideSnapshot],
                tuple(side_states[side] for side in SIDES),
            ),
            cache=tuple(cache.values()),
        )
        candidate_payload = {
            "base_token": base.token,
            "snapshot_token": next_snapshot.token,
            "samples": request_digests,
        }
        candidate_token = hashlib.sha256(
            json.dumps(
                candidate_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        return ControllerControlCandidate(
            base_token=base.token,
            token=candidate_token,
            snapshot=next_snapshot,
            frames=outputs,
        )

    def commit(
        self, candidate: ControllerControlCandidate
    ) -> ControllerControlSnapshot:
        """Atomically publish a prepared candidate, rejecting stale bases.

        Repeating commit for a candidate whose exact target snapshot is already
        current is idempotent, which makes an acknowledged protocol retry safe.
        """
        if not isinstance(candidate, ControllerControlCandidate):
            raise TypeError("control candidate has an incompatible type")
        with self._lock:
            if self._state.token == candidate.snapshot.token:
                return self._state
            if self._state.token != candidate.base_token:
                raise RuntimeError("controller control candidate has a stale base")
            self._state = candidate.snapshot
            return self._state

    def abort(self, candidate: ControllerControlCandidate) -> None:
        """Discard a prepared candidate; preparation never changed live state."""
        if not isinstance(candidate, ControllerControlCandidate):
            raise TypeError("control candidate has an incompatible type")
