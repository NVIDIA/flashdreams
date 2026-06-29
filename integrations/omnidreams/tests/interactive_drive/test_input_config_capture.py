# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Calibration inference for the configuration tool.

These are pure-function tests (no hardware): the same logic must classify a
steering wheel (pedals resting at axis max) and a game controller (triggers
resting at zero) correctly from captured raw samples.
"""

from __future__ import annotations

from omnidreams.interactive_drive.input.wheel_profiles import (
    AxisRange,
    Binding,
    DeviceSpec,
)
from omnidreams.interactive_drive.input_config.capture import (
    build_profile,
    infer_pedal_inverted,
    infer_steering_invert,
    peak_from_observed,
    select_axis_by_span,
)

# A wheel-like steering axis (16-bit) plus two trigger/pedal-like axes (8-bit).
RANGES = {
    0x00: AxisRange(0, 65535),
    0x02: AxisRange(0, 255),
    0x05: AxisRange(0, 255),
}


def test_steering_invert_depends_on_which_extreme_is_lower() -> None:
    # Full-left is the lower raw value -> sign must be flipped (invert).
    assert infer_steering_invert(left_raw=100, right_raw=60000) is True
    # Full-left is the higher raw value -> already correct (no invert).
    assert infer_steering_invert(left_raw=60000, right_raw=100) is False


def test_pedal_inverted_true_for_wheel_pedal() -> None:
    # Wheel pedal rests at axis maximum and falls toward zero when pressed.
    assert infer_pedal_inverted(rest_raw=65535, pressed_raw=0) is True


def test_pedal_inverted_false_for_controller_trigger() -> None:
    # Controller trigger rests at zero and rises when pressed.
    assert infer_pedal_inverted(rest_raw=0, pressed_raw=255) is False


def test_select_axis_by_span_picks_widest_relative_movement() -> None:
    # 0x00 span 5000/65535 (~7.6%) vs 0x02 span 255/255 (100%).
    observed = {0x00: AxisRange(30000, 35000), 0x02: AxisRange(0, 255)}
    assert select_axis_by_span(observed, RANGES) == 0x02


def test_select_axis_by_span_none_below_threshold() -> None:
    observed = {0x00: AxisRange(32000, 33000)}  # 1000/65535 ~1.5%
    assert select_axis_by_span(observed, RANGES) is None


def test_peak_from_observed_returns_far_extreme() -> None:
    # Controller trigger (rest 0): pressed peak is the high extreme.
    assert peak_from_observed(AxisRange(0, 255), reference=0) == 255
    # Wheel pedal (rest at max): pressed peak is the low extreme.
    assert peak_from_observed(AxisRange(0, 65535), reference=65535) == 0
    # Steering held left, center reference: the lower extreme is farther.
    assert peak_from_observed(AxisRange(100, 60000), reference=32767.5) == 100


def test_build_profile_assembles_axis_map_and_flags() -> None:
    profile = build_profile(
        name="my-gamepad",
        display_name="Game controller",
        devices=(DeviceSpec(detection_patterns=("Generic Gamepad",)),),
        axis_map={
            "steering": Binding(0, 0x00),
            "throttle": Binding(0, 0x05),
            "brake": Binding(0, 0x02),
        },
        invert_steering=False,
        inverted_pedals=False,
        ffb_enabled=False,
        ffb_gain=0.0,
        is_default=False,
        reverse_buttons=(Binding(0, 304),),
        reset_buttons=(Binding(0, 305),),
        exit_buttons=(Binding(0, 306),),
    )
    assert profile.axis_map == {
        "steering": Binding(0, 0x00),
        "throttle": Binding(0, 0x05),
        "brake": Binding(0, 0x02),
    }
    assert profile.inverted_pedals is False
    assert profile.ffb_enabled is False
    assert profile.detection_patterns == ("Generic Gamepad",)
    assert profile.reverse_buttons == (Binding(0, 304),)
    assert profile.reset_buttons == (Binding(0, 305),)
    assert profile.exit_buttons == (Binding(0, 306),)


def test_build_profile_supports_axes_split_across_devices() -> None:
    # Steering on the wheel base (device 0), pedals on a separate device 1.
    profile = build_profile(
        name="wheel-plus-pedals",
        display_name="Wheel + pedals",
        devices=(
            DeviceSpec(detection_patterns=("Wheel Base",)),
            DeviceSpec(detection_patterns=("USB Pedals",)),
        ),
        axis_map={
            "steering": Binding(0, 0x00),
            "throttle": Binding(1, 0x00),
            "brake": Binding(1, 0x01),
        },
        invert_steering=False,
        inverted_pedals=False,
        ffb_enabled=True,
        ffb_gain=0.5,
        is_default=False,
    )
    assert len(profile.devices) == 2
    assert profile.axis_map["steering"].device == 0
    assert profile.axis_map["throttle"].device == 1
    assert profile.axis_map["brake"].device == 1
