# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Profile serialization, round-tripping, and the user profiles directory.

The configuration tool's output must parse back to an identical profile via
the same loader the demo runtime uses, so the round-trip guarantee is the
key correctness contract here.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from omnidreams.interactive_drive.input.wheel_profiles import (
    WheelProfile,
    apply_steering_curve,
    delete_profile_file,
    load_wheel_profile_files,
    load_wheel_profiles,
    name_match_strength,
    profile_filename,
    save_wheel_profile,
    update_profile_file,
    user_wheel_profiles_dir,
    wheel_profile_to_yaml_dict,
)


def _wheel_profile() -> WheelProfile:
    return WheelProfile(
        name="my-wheel",
        display_name="Racing wheel",
        detection_patterns=("Generic Racing Wheel",),
        axis_map={"steering": 0x00, "throttle": 0x02, "brake": 0x05},
        inverted_pedals=True,
        invert_steering=True,
        ffb_enabled=True,
        ffb_gain=0.6,
        threshold=0.12,
        is_default=True,
        reverse_buttons=(294,),
        reset_buttons=(300,),
        exit_buttons=(307,),
        steering_range=0.7,
        steering_deadzone=0.1,
    )


def _controller_profile() -> WheelProfile:
    return WheelProfile(
        name="my-gamepad",
        display_name="Game controller",
        detection_patterns=("Generic Gamepad", "Wireless Controller"),
        axis_map={"steering": 0x00, "throttle": 0x05, "brake": 0x02},
        inverted_pedals=False,
        invert_steering=False,
        ffb_enabled=False,
        ffb_gain=0.0,
        threshold=0.12,
        is_default=False,
    )


@pytest.mark.parametrize("profile", [_wheel_profile(), _controller_profile()])
def test_round_trip_save_then_load(profile, tmp_path) -> None:
    save_wheel_profile(profile, tmp_path)
    loaded = load_wheel_profiles(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == profile


def test_yaml_dict_has_loader_schema_shape() -> None:
    data = wheel_profile_to_yaml_dict(_wheel_profile())
    assert set(data) == {
        "name",
        "display_name",
        "is_default",
        "detection_patterns",
        "axis_map",
        "pedal",
        "invert_steering",
        "ffb",
        "threshold",
        "reverse_buttons",
        "reset_buttons",
        "exit_buttons",
        "steering_range",
        "steering_deadzone",
    }
    assert set(data["axis_map"]) == {"steering", "throttle", "brake"}
    assert data["pedal"] == {"inverted": True}
    assert data["ffb"] == {"enabled": True, "gain": 0.6}
    assert data["reverse_buttons"] == [294]
    assert data["reset_buttons"] == [300]
    assert data["exit_buttons"] == [307]
    assert data["steering_range"] == 0.7
    assert data["steering_deadzone"] == 0.1


def test_load_missing_directory_is_empty(tmp_path) -> None:
    assert load_wheel_profiles(tmp_path / "does-not-exist") == ()


def test_profile_filename_is_slugged() -> None:
    assert profile_filename("My Wheel!") == "my-wheel.yaml"
    assert profile_filename("   ") == "profile.yaml"


def test_user_dir_follows_cache_env(monkeypatch, tmp_path) -> None:
    from omnidreams import scenes

    monkeypatch.setattr(scenes, "FLASHDREAMS_CACHE_DIR", tmp_path)
    assert user_wheel_profiles_dir() == tmp_path / "interactive-drive" / "wheels"


def test_apply_steering_curve_scale_and_deadzone() -> None:
    assert apply_steering_curve(1.0, scale=0.5) == 0.5
    assert apply_steering_curve(-1.0, scale=0.5) == -0.5
    # Inside the deadzone reads as zero; just past it rescales from zero.
    assert apply_steering_curve(0.05, deadzone=0.1) == 0.0
    assert apply_steering_curve(1.0, deadzone=0.1) == pytest.approx(1.0)
    assert apply_steering_curve(0.55, deadzone=0.1) == pytest.approx(0.5)
    # Output stays clamped to [-1, 1].
    assert apply_steering_curve(1.0, scale=2.0) == 1.0


def test_name_match_strength_prefers_exact() -> None:
    # The reported bug: a "Wireless Controller" profile must not bind the
    # sibling motion-sensor node whose name merely contains the pattern.
    assert name_match_strength("Wireless Controller", ["Wireless Controller"]) == 2
    assert (
        name_match_strength(
            "Wireless Controller Motion Sensors", ["Wireless Controller"]
        )
        == 1
    )
    assert (
        name_match_strength("Wireless Controller Touchpad", ["Wireless Controller"])
        == 1
    )
    # Substring patterns still work for longer names; case-insensitive.
    assert name_match_strength("Generic Racing Wheel FFB", ["Racing Wheel"]) == 1
    assert name_match_strength("wireless controller", ["Wireless Controller"]) == 2
    assert name_match_strength("Something Else", ["Wireless Controller"]) == 0


def test_profile_file_management_round_trip(tmp_path) -> None:
    profile = _wheel_profile()
    path = save_wheel_profile(profile, tmp_path)
    files = load_wheel_profile_files(tmp_path)
    assert len(files) == 1
    loaded_path, loaded = files[0]
    assert loaded_path == path
    assert loaded == profile
    update_profile_file(path, replace(loaded, display_name="Renamed"))
    assert load_wheel_profile_files(tmp_path)[0][1].display_name == "Renamed"
    delete_profile_file(path)
    assert load_wheel_profile_files(tmp_path) == ()
