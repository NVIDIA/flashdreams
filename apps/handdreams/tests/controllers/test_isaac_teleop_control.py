# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional controller shaping for calibrated wrist targets."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from handdreams.controllers.isaac_teleop_control import (
    HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION,
    ControllerControlConfig,
    ControllerControlShaper,
)
from handdreams.controllers.sampler import ControlWindowSampler
from handdreams.input import OverlapWindowBuffer

pytestmark = pytest.mark.ci_cpu

_IDENTITY_QUATERNION = [0.0, 0.0, 0.0, 1.0]


def _quaternion_z(degrees: float) -> list[float]:
    half = np.deg2rad(degrees) * 0.5
    return [0.0, 0.0, float(np.sin(half)), float(np.cos(half))]


def _controller(
    position: tuple[float, float, float],
    *,
    orientation: list[float] | None = None,
    trigger: float = 0.2,
    grip: float = 1.0,
    tracked: bool = True,
) -> dict:
    return {
        "position": list(position),
        "orientation": list(orientation or _IDENTITY_QUATERNION),
        "trigger": trigger,
        "grip": grip,
        "tracked": tracked,
        "axes": [0.1, -0.2],
        "metadata": {"keep": [1, 2]},
    }


def _frame(
    capture_ns: int,
    *,
    left: tuple[float, float, float] = (-0.2, 1.0, -0.5),
    right: tuple[float, float, float] = (0.2, 1.0, -0.5),
    trigger: float = 0.2,
    grip: float = 1.0,
    tracked: bool = True,
) -> dict:
    return {
        "left": _controller(left, trigger=trigger, grip=grip, tracked=tracked),
        "right": _controller(right, trigger=trigger, grip=grip, tracked=tracked),
        "client_capture_ns": capture_ns,
        "top_level_metadata": {"source": "test", "nested": [3, 4]},
    }


def _config(**kwargs: object) -> ControllerControlConfig:
    values: dict[str, object] = {
        "scene_gripper_open": {"left": 0.8, "right": 0.7},
        "calibration_trigger": {"left": 0.2, "right": 0.1},
    }
    values.update(kwargs)
    return ControllerControlConfig.from_payload(values)


def _calibrated(raw: list[dict], *, gain: float = 1.0) -> list[dict]:
    result = copy.deepcopy(raw)
    for frame in result:
        for side in ("left", "right"):
            origin = -0.2 if side == "left" else 0.2
            position = np.asarray(frame[side]["position"], dtype=np.float64)
            position[0] = origin + gain * (position[0] - origin)
            frame[side]["position"] = position.tolist()
    return result


def _x(frame: dict, side: str) -> float:
    return float(frame[side]["position"][0])


def test_config_round_trip_and_output_convention() -> None:
    config = _config(
        grip_threshold=0.4,
        stale_ms=200.0,
        max_position_step_m=0.5,
        input_position_gain=1.5,
        gripper_slew_per_action=0.02,
    )

    assert ControllerControlConfig.from_payload(config.to_payload()) == config
    assert (
        HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION
        == "handdreams-absolute-command-target-v1"
    )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"grip_threshold": 2.0}, "grip threshold"),
        ({"stale_ms": 0.0}, "stale timeout"),
        ({"max_position_step_m": -1.0}, "maximum position step"),
        ({"input_position_gain": 0.0}, "input position gain"),
        ({"gripper_slew_per_action": -0.1}, "gripper slew"),
    ],
)
def test_config_rejects_invalid_values(override: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _config(**override)


def test_motion_is_integrated_and_inputs_and_metadata_remain_immutable() -> None:
    raw = [
        _frame(0),
        _frame(1_000_000, right=(0.3, 1.0, -0.5)),
    ]
    calibrated = _calibrated(raw, gain=1.5)
    original_raw = copy.deepcopy(raw)
    original_calibrated = copy.deepcopy(calibrated)
    shaper = ControllerControlShaper(_config(input_position_gain=1.5))

    candidate = shaper.prepare(raw, calibrated)
    shaper.commit(candidate)
    output = candidate.frames
    output[1]["right"]["metadata"]["keep"].append(9)
    output[1]["top_level_metadata"]["nested"].append(9)

    assert _x(output[0], "right") == pytest.approx(0.2)
    assert _x(output[1], "right") == pytest.approx(0.35)
    assert raw == original_raw
    assert calibrated == original_calibrated
    assert calibrated[1]["right"]["metadata"] == {"keep": [1, 2]}
    assert calibrated[1]["top_level_metadata"]["nested"] == [3, 4]


def test_clutch_requires_both_endpoints_and_reanchors_while_released() -> None:
    raw = [
        _frame(0, grip=0.0),
        _frame(1_000_000, right=(0.4, 1.0, -0.5), grip=1.0),
        _frame(2_000_000, right=(0.5, 1.0, -0.5), grip=1.0),
    ]
    shaper = ControllerControlShaper(_config(grip_threshold=0.5))

    candidate = shaper.prepare(raw, _calibrated(raw))
    shaper.commit(candidate)
    output = candidate.frames

    assert [_x(frame, "right") for frame in output] == pytest.approx([0.2, 0.2, 0.3])


def test_tracking_loss_and_first_reacquisition_hold_the_command() -> None:
    raw = [
        _frame(0),
        _frame(1_000_000, right=(0.4, 1.0, -0.5)),
        _frame(2_000_000, right=(0.8, 1.0, -0.5)),
        _frame(3_000_000, right=(0.9, 1.0, -0.5)),
    ]
    raw[1]["right"]["tracked"] = False
    calibrated = _calibrated(raw)
    shaper = ControllerControlShaper(_config())

    candidate = shaper.prepare(raw, calibrated)
    shaper.commit(candidate)
    output = candidate.frames

    assert [_x(frame, "right") for frame in output] == pytest.approx(
        [0.2, 0.2, 0.2, 0.3]
    )
    assert output[1]["right"]["tracked"] is False


def test_stale_sample_drops_anchor_then_reacquires_before_motion() -> None:
    raw = [
        _frame(0),
        _frame(200_000_000, right=(0.5, 1.0, -0.5)),
        _frame(201_000_000, right=(0.6, 1.0, -0.5)),
        _frame(202_000_000, right=(0.7, 1.0, -0.5)),
    ]
    shaper = ControllerControlShaper(_config(stale_ms=100.0))

    candidate = shaper.prepare(raw, _calibrated(raw))
    shaper.commit(candidate)
    output = candidate.frames

    assert [_x(frame, "right") for frame in output] == pytest.approx(
        [0.2, 0.2, 0.2, 0.3]
    )


def test_position_discontinuity_holds_row_but_reanchors_endpoint() -> None:
    raw = [
        _frame(0),
        _frame(1_000_000, right=(1.0, 1.0, -0.5)),
        _frame(2_000_000, right=(1.05, 1.0, -0.5)),
    ]
    shaper = ControllerControlShaper(_config())

    candidate = shaper.prepare(raw, _calibrated(raw))
    shaper.commit(candidate)
    output = candidate.frames

    assert [_x(frame, "right") for frame in output] == pytest.approx([0.2, 0.2, 0.25])


def test_discontinuity_threshold_uses_physical_distance_before_input_gain() -> None:
    raw = [_frame(0), _frame(1_000_000, right=(0.6, 1.0, -0.5))]
    calibrated = _calibrated(raw, gain=2.0)
    shaper = ControllerControlShaper(_config(input_position_gain=2.0))

    candidate = shaper.prepare(raw, calibrated)
    shaper.commit(candidate)
    output = candidate.frames

    assert _x(output[1], "right") == pytest.approx(1.0)


def test_rotation_is_integrated_without_a_step_limit() -> None:
    raw = [_frame(0), _frame(1_000_000), _frame(2_000_000)]
    for side in ("left", "right"):
        raw[1][side]["orientation"] = _quaternion_z(150.0)
        raw[2][side]["orientation"] = _quaternion_z(155.0)
    shaper = ControllerControlShaper(_config())

    candidate = shaper.prepare(raw, _calibrated(raw))
    shaper.commit(candidate)
    output = candidate.frames

    assert output[1]["right"]["orientation"] == pytest.approx(
        _quaternion_z(150.0), abs=1e-6
    )
    assert output[2]["right"]["orientation"] == pytest.approx(
        _quaternion_z(155.0), abs=1e-6
    )


def test_gripper_is_scene_relative_endpoint_aligned_and_unslewed() -> None:
    raw = [
        _frame(0, trigger=0.2),
        _frame(1_000_000, trigger=0.7),
    ]
    # A tiny slew limit is retained for later model-row shaping, not applied at
    # the controller sample boundary.
    shaper = ControllerControlShaper(_config(gripper_slew_per_action=0.01))

    candidate = shaper.prepare(raw, _calibrated(raw))
    shaper.commit(candidate)
    output = candidate.frames

    assert output[0]["left"]["gripper_open"] == pytest.approx(0.8)
    assert output[1]["left"]["gripper_open"] == pytest.approx(0.3)
    assert output[0]["right"]["gripper_open"] == pytest.approx(0.6)
    assert output[1]["right"]["gripper_open"] == pytest.approx(0.1)


def test_overlapping_windows_are_idempotent_by_capture_time() -> None:
    all_raw = [
        _frame(0),
        _frame(1_000_000, right=(0.3, 1.0, -0.5)),
        _frame(2_000_000, right=(0.4, 1.0, -0.5)),
        _frame(3_000_000, right=(0.5, 1.0, -0.5)),
    ]
    all_calibrated = _calibrated(all_raw)
    shaper = ControllerControlShaper(_config())

    first = shaper.prepare(all_raw[:3], all_calibrated[:3])
    shaper.commit(first)
    generation = shaper.snapshot().generation
    second = shaper.prepare(all_raw[1:], all_calibrated[1:])
    shaper.commit(second)
    repeated = shaper.prepare(all_raw[1:], all_calibrated[1:])
    shaper.commit(repeated)

    assert [_x(frame, "right") for frame in first.frames] == pytest.approx(
        [0.2, 0.3, 0.4]
    )
    assert [_x(frame, "right") for frame in second.frames] == pytest.approx(
        [0.3, 0.4, 0.5]
    )
    assert [_x(frame, "right") for frame in repeated.frames] == pytest.approx(
        [0.3, 0.4, 0.5]
    )
    assert shaper.snapshot().generation == generation + 1


def test_shifted_resampling_grid_holds_current_historical_sample() -> None:
    """Nearest-neighbour jitter must not rewind the live command integrator."""
    first_raw = [
        _frame(0),
        _frame(2_000_000, right=(0.4, 1.0, -0.5)),
    ]
    shaper = ControllerControlShaper(_config())
    first = shaper.prepare(first_raw, _calibrated(first_raw))
    shaper.commit(first)

    # A later resampling pass exposes timestamp 1 between the already accepted
    # timestamps 0 and 2. It was not delivered out of order by the producer; it
    # simply lost the previous nearest-neighbour tie.
    shifted_raw = [
        _frame(1_000_000, right=(0.3, 1.0, -0.5)),
        first_raw[1],
        _frame(3_000_000, right=(0.5, 1.0, -0.5)),
    ]
    candidate = shaper.prepare(shifted_raw, _calibrated(shifted_raw))
    shaper.commit(candidate)
    output = candidate.frames

    # The historical hole holds the current command; timestamp 2 retains its
    # exact cached result and timestamp 3 advances live state once.
    assert [_x(frame, "right") for frame in output] == pytest.approx([0.4, 0.4, 0.5])
    assert shaper.snapshot().last_capture_ns == 3_000_000


def test_72hz_capture_jitter_survives_36hz_resampling() -> None:
    """A realistic skipped capture must not poison controller delivery."""
    capture_interval_ns = int(1e9 / 72.0)
    sampled_indices = [0, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 32]
    sampler = ControlWindowSampler(rate_hz=36.0)
    shaper = ControllerControlShaper(_config())
    starting_generation = shaper.snapshot().generation
    shaped_windows: list[list[dict]] = []
    poses = OverlapWindowBuffer()
    next_control_ns = 0
    last_control_ns = -1

    for index in sampled_indices:
        frame = _frame(
            index * capture_interval_ns,
            right=(0.2 + index * 0.001, 1.0, -0.5),
        )
        assert sampler.add(frame)
        while next_control_ns <= frame["client_capture_ns"]:
            selected = sampler.sample_at(next_control_ns)
            next_control_ns += sampler.interval_ns
            if selected is not None and selected["client_capture_ns"] > last_control_ns:
                poses.append(selected)
                last_control_ns = selected["client_capture_ns"]
        while poses.ready(4):
            window = poses.peek(4)
            candidate = shaper.prepare(window, _calibrated(window))
            shaper.commit(candidate)
            shaped_windows.append(candidate.frames)
            poses.consume(4)

    expected_indices = [[0, 2, 5, 7, 9], [9, 11, 13, 15, 17], [17, 19, 21, 23, 25]]
    assert [
        [frame["client_capture_ns"] for frame in window] for window in shaped_windows
    ] == [
        [index * capture_interval_ns for index in window] for window in expected_indices
    ]
    assert shaper.snapshot().generation == starting_generation + 3
    assert shaper.snapshot().last_capture_ns == 25 * capture_interval_ns


def test_cached_retry_can_refresh_metadata_without_advancing_control() -> None:
    raw = [_frame(0), _frame(1_000_000, right=(0.3, 1.0, -0.5))]
    calibrated = _calibrated(raw)
    shaper = ControllerControlShaper(_config())
    first = shaper.prepare(raw, calibrated)
    shaper.commit(first)
    retried = copy.deepcopy(calibrated)
    retried[1]["top_level_metadata"]["source"] = "retry"

    candidate = shaper.prepare(raw, retried)
    shaper.commit(candidate)
    output = candidate.frames

    assert output[1]["top_level_metadata"]["source"] == "retry"
    assert _x(output[1], "right") == pytest.approx(0.3)


def test_reused_timestamp_with_different_semantic_input_is_rejected() -> None:
    raw = [_frame(0)]
    calibrated = _calibrated(raw)
    shaper = ControllerControlShaper(_config())
    candidate = shaper.prepare(raw, calibrated)
    shaper.commit(candidate)
    changed = [_frame(0, right=(0.3, 1.0, -0.5))]

    with pytest.raises(ValueError, match="different input"):
        shaper.prepare(changed, _calibrated(changed))


def test_uncached_historical_sample_holds_without_advancing_snapshot() -> None:
    shaper = ControllerControlShaper(_config())
    current = [_frame(10, right=(0.4, 1.0, -0.5))]
    current_candidate = shaper.prepare(current, _calibrated(current))
    shaper.commit(current_candidate)
    before = shaper.snapshot()
    older = [_frame(5, right=(0.9, 1.0, -0.5))]

    candidate = shaper.prepare(older, _calibrated(older))

    assert _x(candidate.frames[0], "right") == pytest.approx(0.4)
    assert candidate.snapshot.token == before.token
    assert candidate.snapshot.generation == before.generation
    assert candidate.snapshot.last_capture_ns == before.last_capture_ns
    assert shaper.commit(candidate).token == before.token


def test_prepare_commit_abort_and_candidate_frames_are_transactional() -> None:
    shaper = ControllerControlShaper(_config())
    raw = [_frame(0), _frame(1_000_000, right=(0.3, 1.0, -0.5))]
    calibrated = _calibrated(raw)
    before = shaper.snapshot()

    candidate = shaper.prepare(raw, calibrated)
    changed_copy = candidate.frames
    changed_copy[1]["right"]["position"][0] = 99.0

    assert shaper.snapshot().token == before.token
    assert _x(candidate.frames[1], "right") == pytest.approx(0.3)
    shaper.abort(candidate)
    assert shaper.snapshot().token == before.token
    committed = shaper.commit(candidate)
    assert committed.token == candidate.snapshot.token
    assert shaper.commit(candidate).token == committed.token


def test_stale_candidate_cannot_overwrite_a_newer_commit() -> None:
    shaper = ControllerControlShaper(_config())
    first = [_frame(0)]
    candidate_a = shaper.prepare(first, _calibrated(first))
    candidate_b = shaper.prepare(first, _calibrated(first))
    shaper.commit(candidate_a)

    # Identical candidates have the same target snapshot and are safe protocol
    # retries rather than conflicting writes.
    assert shaper.commit(candidate_b).token == candidate_a.snapshot.token

    next_raw = [_frame(1, right=(0.3, 1.0, -0.5))]
    stale = shaper.prepare(next_raw, _calibrated(next_raw))
    newer_raw = [_frame(2, right=(0.4, 1.0, -0.5))]
    newer = shaper.prepare(newer_raw, _calibrated(newer_raw))
    shaper.commit(newer)
    with pytest.raises(RuntimeError, match="stale base"):
        shaper.commit(stale)


def test_window_validation_rejects_misaligned_or_decreasing_times() -> None:
    shaper = ControllerControlShaper(_config())
    raw = [_frame(0), _frame(1)]
    calibrated = _calibrated(raw)
    calibrated[1]["client_capture_ns"] = 2
    with pytest.raises(ValueError, match="capture times must match"):
        shaper.prepare(raw, calibrated)

    decreasing = [_frame(1), _frame(0)]
    with pytest.raises(ValueError, match="must not decrease"):
        shaper.prepare(decreasing, _calibrated(decreasing))


def test_window_reuses_identical_nearest_neighbor_sample() -> None:
    shaper = ControllerControlShaper(_config())
    repeated = [_frame(0), _frame(0), _frame(1)]

    candidate = shaper.prepare(repeated, _calibrated(repeated))
    shaper.commit(candidate)
    output = candidate.frames

    assert len(output) == 3
    assert output[0] == output[1]
    assert shaper.snapshot().last_capture_ns == 1


def test_first_untracked_sample_fails_closed() -> None:
    shaper = ControllerControlShaper(_config())
    raw = [_frame(0, tracked=False)]

    with pytest.raises(RuntimeError, match="no tracked command pose"):
        shaper.prepare(raw, _calibrated(raw))


def test_module_loads_as_a_standalone_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = (
        Path(__file__).parents[2]
        / "handdreams"
        / "controllers"
        / "isaac_teleop_control.py"
    )
    monkeypatch.syspath_prepend(str(module_path.parent))
    spec = importlib.util.spec_from_file_location(
        "standalone_isaac_teleop_control", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    assert (
        module.HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION
        == HANDDREAMS_ABSOLUTE_COMMAND_TARGET_CONVENTION
    )
