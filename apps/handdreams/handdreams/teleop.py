# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-scoped access to the Isaac Teleop controller bridge."""

from __future__ import annotations

import copy
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from handdreams.controllers import quest_controller
from handdreams.controllers.isaac_teleop_calibration import (
    HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION,
)
from handdreams.controllers.isaac_teleop_calibration_client import (
    PreparedControlWindow,
)
from handdreams.controllers.isaac_teleop_client import (
    IsaacTeleopBridge,
)
from handdreams.controllers.isaac_teleop_control import (
    HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
    ControllerControlConfig,
)
from handdreams.controllers.quest_pose import (
    HANDDREAMS_WRIST_PROXY_POSE_CONVENTION,
)


@dataclass(frozen=True, slots=True)
class PreparedTeleopWindow:
    """Absolute command targets plus their bridge-owned transaction receipt."""

    frames: list[dict[str, Any]]
    receipt: PreparedControlWindow
    final_gripper_open: tuple[tuple[str, float], ...] = ()


class TeleopControllerPipeline(Protocol):
    """Small injectable boundary used by the V2 model loop and CPU tests."""

    def start(self) -> None: ...

    def transform(self, frame: dict[str, Any]) -> dict[str, Any]: ...

    def calibrate(
        self,
        neutral_frame: dict[str, Any],
        *,
        scene_gripper_open: Mapping[str, float],
        position_gain: float,
        grip_threshold: float,
        stale_ms: float | None,
        gripper_slew_per_action: float,
        participant_basis: Any | None = None,
    ) -> None: ...

    def prepare_window(
        self, frames: list[dict[str, Any]], *, command_sequence: int
    ) -> PreparedTeleopWindow: ...

    def commit(self, prepared: PreparedTeleopWindow) -> None: ...

    def abort(self, prepared: PreparedTeleopWindow) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class IsaacTeleopControllerPipeline:
    """Validate Isaac wrist poses and own isolated calibration/control state.

    Both controller sources apply the grip-to-wrist offset before delivery.
    The bridge's calibration/control RPC owns
    participant recentering, movement gain, clutching, discontinuity handling,
    tracking-loss holds, and trigger-to-gripper mapping. The model loop only
    receives committed absolute command targets.
    """

    def __init__(
        self,
        *,
        python_executable: str,
        startup_timeout_s: float,
        bridge_factory: type[IsaacTeleopBridge] = IsaacTeleopBridge,
    ) -> None:
        self._bridge = bridge_factory(
            python_executable=python_executable,
            startup_timeout_s=startup_timeout_s,
        )
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._calibrated = False
        self._gripper_slew_per_action = 0.0
        self._committed_gripper_open: dict[str, float] | None = None

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Isaac Teleop controller pipeline is closed")
            if self._started:
                return
            self._bridge.start()
            self._started = True

    def _require_started(self) -> None:
        if self._closed or not self._started:
            raise RuntimeError("Isaac Teleop controller pipeline is not running")

    def transform(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Validate and copy one canonical wrist frame without changing its pose."""
        self._require_started()
        convention = frame.get("pose_convention")
        if convention != HANDDREAMS_WRIST_PROXY_POSE_CONVENTION:
            raise ValueError(f"unsupported controller pose convention {convention!r}")

        validated = quest_controller.validate_controller_frame(frame)
        # The validator intentionally rebuilds the wire schema. Preserve V2
        # source identity and other non-controller metadata out of band.
        for key, value in frame.items():
            if key not in {*quest_controller.SIDES, "client_capture_ns"}:
                validated[key] = copy.deepcopy(value)
        validated["pose_convention"] = HANDDREAMS_WRIST_PROXY_POSE_CONVENTION
        return validated

    def calibrate(
        self,
        neutral_frame: dict[str, Any],
        *,
        scene_gripper_open: Mapping[str, float],
        position_gain: float,
        grip_threshold: float,
        stale_ms: float | None,
        gripper_slew_per_action: float,
        participant_basis: Any | None = None,
    ) -> None:
        """Install participant calibration and source-owned command shaping."""
        self._require_started()
        validated = quest_controller.validate_controller_frame(neutral_frame)
        basis = (
            quest_controller.controller_separation_basis(validated)
            if participant_basis is None
            else quest_controller.validate_participant_basis(participant_basis)
        )
        calibration: dict[str, Any] = {
            side: {
                "position": list(validated[side]["position"]),
                "orientation": list(validated[side]["orientation"]),
                "trigger": float(validated[side].get("trigger", 0.0)),
            }
            for side in ("left", "right")
        }
        calibration["participant_basis"] = basis.tolist()
        calibration["position_gain"] = float(position_gain)
        with self._lock:
            gripper_baseline = dict(
                self._committed_gripper_open
                if self._committed_gripper_open is not None
                else scene_gripper_open
            )
        control = ControllerControlConfig(
            scene_gripper_open={
                side: float(gripper_baseline[side]) for side in ("left", "right")
            },
            calibration_trigger={
                side: float(calibration[side]["trigger"]) for side in ("left", "right")
            },
            grip_threshold=grip_threshold,
            stale_ms=stale_ms,
            input_position_gain=position_gain,
            gripper_slew_per_action=gripper_slew_per_action,
        ).to_payload()
        state = self._bridge.configure_calibration(calibration, control=control)
        if state is None:
            raise RuntimeError("Isaac Teleop did not acknowledge calibration")
        if state.pose_space != HANDDREAMS_PARTICIPANT_WRIST_POSE_CONVENTION:
            raise RuntimeError(
                f"Isaac Teleop returned incompatible pose space {state.pose_space!r}"
            )
        if state.control_space != HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION:
            raise RuntimeError(
                "Isaac Teleop did not enable absolute command-target shaping"
            )
        with self._lock:
            self._gripper_slew_per_action = float(gripper_slew_per_action)
            # Recentring changes the pose basis, not the last accepted model
            # gripper. A session reset clears this baseline explicitly.
            if self._committed_gripper_open is None:
                self._committed_gripper_open = {
                    side: float(scene_gripper_open[side]) for side in ("left", "right")
                }
            self._calibrated = True

    def prepare_window(
        self, frames: list[dict[str, Any]], *, command_sequence: int
    ) -> PreparedTeleopWindow:
        """Prepare a stateful command window without advancing bridge state."""
        self._require_started()
        if not self._calibrated:
            raise RuntimeError("Isaac Teleop controller pipeline is not calibrated")
        receipt = self._bridge.prepare_window(
            frames,
            command_sequence=command_sequence,
        )
        shaped = copy.deepcopy(receipt.frames)
        with self._lock:
            committed = self._committed_gripper_open
            limit = self._gripper_slew_per_action
            if committed is None:
                raise RuntimeError("Isaac Teleop gripper state is not calibrated")
            current = dict(committed)
        for side in ("left", "right"):
            if shaped:
                shaped[0][side]["gripper_open"] = current[side]
            value = current[side]
            for frame in shaped[1:]:
                desired = float(frame[side]["gripper_open"])
                if not math.isfinite(desired) or not 0.0 <= desired <= 1.0:
                    raise ValueError(
                        f"{side} command gripper_open must be finite and in [0, 1]"
                    )
                if limit == 0.0:
                    value = desired
                else:
                    value += max(-limit, min(limit, desired - value))
                frame[side]["gripper_open"] = value
            current[side] = value
        return PreparedTeleopWindow(
            frames=shaped,
            receipt=receipt,
            final_gripper_open=tuple(sorted(current.items())),
        )

    def commit(self, prepared: PreparedTeleopWindow) -> None:
        self._require_started()
        self._bridge.commit_window(prepared.receipt)
        if prepared.final_gripper_open:
            with self._lock:
                self._committed_gripper_open = dict(prepared.final_gripper_open)

    def abort(self, prepared: PreparedTeleopWindow) -> None:
        self._require_started()
        self._bridge.abort_window(prepared.receipt)

    def reset(self) -> None:
        """Clear calibration while retaining the isolated service."""
        if self._closed or not self._started:
            self._calibrated = False
            self._committed_gripper_open = None
            return
        self._bridge.configure_calibration(None)
        with self._lock:
            self._calibrated = False
            self._committed_gripper_open = None

    def close(self) -> None:
        """Stop the isolated bridge exactly once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._started = False
            self._calibrated = False
            self._committed_gripper_open = None
        self._bridge.stop()


__all__ = [
    "IsaacTeleopControllerPipeline",
    "PreparedTeleopWindow",
    "TeleopControllerPipeline",
]
