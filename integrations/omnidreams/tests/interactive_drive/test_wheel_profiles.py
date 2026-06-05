# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Profile serialization, round-tripping, and the user profiles directory.

The configuration tool's output must parse back to an identical profile via
the same loader the demo runtime uses, so the round-trip guarantee is the
key correctness contract here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from omnidreams.interactive_drive.input.wheel_profiles import (
    FF_AUTOCENTER,
    FF_CONSTANT,
    AutocenterFFB,
    AxisRange,
    Binding,
    ConstantForceFFB,
    DeviceSpec,
    EvdevDevice,
    WheelProfile,
    apply_steering_curve,
    create_ffb_backend,
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
        devices=(DeviceSpec(detection_patterns=("Generic Racing Wheel",)),),
        axis_map={
            "steering": Binding(0, 0x00),
            "throttle": Binding(0, 0x02),
            "brake": Binding(0, 0x05),
        },
        inverted_pedals=True,
        invert_steering=True,
        ffb_enabled=True,
        ffb_gain=0.6,
        threshold=0.12,
        is_default=True,
        reverse_buttons=(Binding(0, 294),),
        reset_buttons=(Binding(0, 300),),
        exit_buttons=(Binding(0, 307),),
        steering_range=0.7,
        steering_deadzone=0.1,
    )


def _controller_profile() -> WheelProfile:
    return WheelProfile(
        name="my-gamepad",
        display_name="Game controller",
        devices=(
            DeviceSpec(
                detection_patterns=("Generic Gamepad", "Wireless Controller"),
                display_name="Gamepad",
            ),
        ),
        axis_map={
            "steering": Binding(0, 0x00),
            "throttle": Binding(0, 0x05),
            "brake": Binding(0, 0x02),
        },
        inverted_pedals=False,
        invert_steering=False,
        ffb_enabled=False,
        ffb_gain=0.0,
        threshold=0.12,
        is_default=False,
    )


def _multi_device_profile() -> WheelProfile:
    """A wheel base (steering) plus a separate-brand pedal set (throttle/brake)."""
    return WheelProfile(
        name="wheel-plus-pedals",
        display_name="Wheel + separate pedals",
        devices=(
            DeviceSpec(detection_patterns=("Fanatec CSL DD",), display_name="Base"),
            DeviceSpec(
                detection_patterns=("Heusinkveld Pedals",), display_name="Pedals"
            ),
        ),
        axis_map={
            "steering": Binding(0, 0x00),
            "throttle": Binding(1, 0x00),
            "brake": Binding(1, 0x01),
        },
        inverted_pedals=False,
        invert_steering=False,
        ffb_enabled=True,
        ffb_gain=0.6,
        ffb_mode="constant_force",
        is_default=False,
        reverse_buttons=(Binding(0, 294),),
    )


@pytest.mark.parametrize(
    "profile",
    [_wheel_profile(), _controller_profile(), _multi_device_profile()],
)
def test_round_trip_save_then_load(profile, tmp_path) -> None:
    save_wheel_profile(profile, tmp_path)
    loaded = load_wheel_profiles(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == profile


def test_legacy_single_device_yaml_migrates_to_one_device(tmp_path) -> None:
    # Profiles written before multi-device used top-level detection_patterns
    # and bare int axis/button codes. They must load as a single device with
    # device-0 bindings so existing user profiles keep working.
    legacy = {
        "name": "legacy-wheel",
        "display_name": "Legacy wheel",
        "detection_patterns": ["Old Wheel"],
        "axis_map": {"steering": 0, "throttle": 1, "brake": 2},
        "pedal": {"inverted": True},
        "ffb": {"enabled": True, "gain": 0.5},
        "reverse_buttons": [294],
    }
    (tmp_path / "legacy.yaml").write_text(yaml.safe_dump(legacy), encoding="utf-8")
    (profile,) = load_wheel_profiles(tmp_path)
    assert profile.devices == (DeviceSpec(detection_patterns=("Old Wheel",)),)
    assert profile.axis_map == {
        "steering": Binding(0, 0),
        "throttle": Binding(0, 1),
        "brake": Binding(0, 2),
    }
    assert profile.reverse_buttons == (Binding(0, 294),)
    # The compat accessor still exposes the primary device's patterns.
    assert profile.detection_patterns == ("Old Wheel",)


def test_round_trip_preserves_explicit_ffb_mode(tmp_path) -> None:
    # An explicit mode must survive save/load so a Fanatec keeps constant force.
    profile = replace(_wheel_profile(), ffb_mode="constant_force")
    save_wheel_profile(profile, tmp_path)
    loaded = load_wheel_profiles(tmp_path)
    assert loaded[0].ffb_mode == "constant_force"


@pytest.mark.parametrize(
    ("mode", "features", "expected"),
    [
        # Explicit modes ignore the advertised effects.
        ("autocenter", frozenset({FF_CONSTANT}), AutocenterFFB),
        ("constant_force", frozenset({FF_AUTOCENTER}), ConstantForceFFB),
        # "auto" prefers autocenter, falls back to constant force, else no-op.
        ("auto", frozenset({FF_AUTOCENTER, FF_CONSTANT}), AutocenterFFB),
        ("auto", frozenset({FF_CONSTANT}), ConstantForceFFB),
        ("auto", frozenset(), AutocenterFFB),
    ],
)
def test_create_ffb_backend_resolution(mode, features, expected) -> None:
    assert isinstance(create_ffb_backend(mode, features), expected)


def test_yaml_dict_has_loader_schema_shape() -> None:
    data = wheel_profile_to_yaml_dict(_wheel_profile())
    assert set(data) == {
        "name",
        "display_name",
        "is_default",
        "devices",
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
    assert data["devices"] == [
        {"display_name": "", "detection_patterns": ["Generic Racing Wheel"]}
    ]
    assert set(data["axis_map"]) == {"steering", "throttle", "brake"}
    assert data["axis_map"]["steering"] == {"device": 0, "code": 0x00}
    assert data["pedal"] == {"inverted": True}
    assert data["ffb"] == {"enabled": True, "gain": 0.6, "mode": "auto"}
    assert data["reverse_buttons"] == [{"device": 0, "code": 294}]
    assert data["reset_buttons"] == [{"device": 0, "code": 300}]
    assert data["exit_buttons"] == [{"device": 0, "code": 307}]
    assert data["steering_range"] == 0.7
    assert data["steering_deadzone"] == 0.1


def test_multi_device_yaml_shape() -> None:
    data = wheel_profile_to_yaml_dict(_multi_device_profile())
    assert [d["detection_patterns"] for d in data["devices"]] == [
        ["Fanatec CSL DD"],
        ["Heusinkveld Pedals"],
    ]
    # Pedals live on device 1, steering on device 0.
    assert data["axis_map"]["steering"] == {"device": 0, "code": 0x00}
    assert data["axis_map"]["throttle"] == {"device": 1, "code": 0x00}
    assert data["axis_map"]["brake"] == {"device": 1, "code": 0x01}


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


# --- runtime device resolution (demo.py) -------------------------------

_BASE = EvdevDevice(path=Path("/dev/input/event0"), name="Fanatec CSL DD")
_PEDALS = EvdevDevice(path=Path("/dev/input/event1"), name="Heusinkveld Pedals")


def test_resolve_profile_devices_maps_each_device(monkeypatch) -> None:
    from omnidreams.interactive_drive import demo

    # Pretend every bound axis exists on whichever device we query.
    monkeypatch.setattr(
        demo, "_query_axis_range", lambda path, code: AxisRange(0, 65535)
    )
    resolved = demo._resolve_profile_devices(_multi_device_profile(), (_BASE, _PEDALS))
    assert resolved == {0: _BASE.path, 1: _PEDALS.path}


def test_resolve_profile_devices_requires_steering(monkeypatch) -> None:
    from omnidreams.interactive_drive import demo

    monkeypatch.setattr(
        demo, "_query_axis_range", lambda path, code: AxisRange(0, 65535)
    )
    # Only the pedal set is connected; without the steering device, None.
    assert demo._resolve_profile_devices(_multi_device_profile(), (_PEDALS,)) is None


def test_resolve_profile_devices_degrades_when_extra_missing(monkeypatch) -> None:
    from omnidreams.interactive_drive import demo

    monkeypatch.setattr(
        demo, "_query_axis_range", lambda path, code: AxisRange(0, 65535)
    )
    # Steering device present, pedal device absent: pedals (index 1) are simply
    # left unresolved rather than failing the whole profile.
    resolved = demo._resolve_profile_devices(_multi_device_profile(), (_BASE,))
    assert resolved == {0: _BASE.path}
