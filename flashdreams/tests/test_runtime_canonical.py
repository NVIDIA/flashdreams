# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the raw-input to canonical-modality layer.

These cover the middle leg of ``raw input -> canonicalized input -> encoded
inference input``: applications consume canonical modalities, never raw device
events, so adding a device is a registration rather than an application change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from flashdreams.runtime import (
    CONDITIONING_PROMPT,
    DRIVER_COMMAND,
    CanonicalInputs,
    CanonicalModality,
    DeviceConverterSchema,
    InferenceInputSchema,
    InputCanonicalizer,
    InputField,
    InputMappingSchema,
    KeyboardToDriverCommand,
    LatestEventToModality,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
    check_mapping_compatibility,
)

pytestmark = pytest.mark.ci_cpu

KEYBOARD_SOURCE = UserInputSchema(
    capabilities=(
        UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"})),
        UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
    )
)
WHEEL_SOURCE = UserInputSchema(
    capabilities=(
        UserInputCapability(
            event_type="wheel_axis", payload_fields=frozenset({"axis", "value"})
        ),
    )
)
PROMPT_SOURCE = UserInputSchema(
    capabilities=(
        UserInputCapability(
            event_type="prompt_set", payload_fields=frozenset({"prompt"})
        ),
    )
)

# Written once against the canonical modality. It names no key and no axis.
STEERING_MAPPING = InputMappingSchema(
    name="driver-command-to-steering",
    consumes=(DRIVER_COMMAND,),
    produces_step=(InputField(name="steering"),),
)
STEERING_MODEL = InferenceInputSchema(step_fields=(InputField(name="steering"),))

WINDOW = TimeWindow(start_s=0.0, end_s=1.0)
NEXT_WINDOW = TimeWindow(start_s=1.0, end_s=2.0)


class WheelToDriverCommand:
    """Minimal wheel converter standing in for a real evdev profile."""

    def __init__(self, *, priority: int = 10) -> None:
        self._steer = 0.0
        self._seen = False
        self._schema = DeviceConverterSchema(
            name="wheel-to-driver-command",
            produces=DRIVER_COMMAND,
            device_kind="wheel",
            priority=priority,
            consumes=(
                UserInputCapability(
                    event_type="wheel_axis",
                    payload_fields=frozenset({"axis", "value"}),
                ),
            ),
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        return self._schema

    def reset(self) -> None:
        self._steer = 0.0
        self._seen = False

    def convert(
        self, user_inputs: UserInputs, window: TimeWindow
    ) -> Mapping[str, Any] | None:
        del window
        for event in user_inputs.events:
            if event.event_type == "wheel_axis" and event.payload["axis"] == "steer":
                self._seen = True
                self._steer = float(event.payload["value"])
        if not self._seen:
            return None
        return DRIVER_COMMAND.value(
            {
                "throttle": 0.0,
                "brake": 0.0,
                "steer": self._steer,
                "stop": False,
                "reverse": False,
            }
        )


def _key(event_type: str, key: str, timestamp_s: float) -> UserInputEvent:
    return UserInputEvent(
        timestamp_s=timestamp_s, event_type=event_type, payload={"key": key}
    )


def _command(canonical: CanonicalInputs) -> Mapping[str, Any]:
    assert DRIVER_COMMAND.name in canonical.per_step
    return canonical.per_step[DRIVER_COMMAND.name]


# --- per-step conditioning ----------------------------------------------


def test_keyboard_edges_become_canonical_driver_command() -> None:
    canonicalizer = InputCanonicalizer([KeyboardToDriverCommand()])
    inputs = UserInputs(events=(_key("key_down", "w", 0.1),))

    canonical = canonicalizer.canonicalize(
        inputs, window=WINDOW, source_schema=KEYBOARD_SOURCE
    )

    assert _command(canonical)["throttle"] == 1.0
    assert _command(canonical)["steer"] == 0.0
    assert canonical.metadata["canonical_sources"]["driver_command"] == "keyboard"


def test_key_aliases_are_normalized() -> None:
    canonicalizer = InputCanonicalizer([KeyboardToDriverCommand()])
    inputs = UserInputs(events=(_key("key_down", "ArrowLeft", 0.1),))

    canonical = canonicalizer.canonicalize(
        inputs, window=WINDOW, source_schema=KEYBOARD_SOURCE
    )

    assert _command(canonical)["steer"] == 1.0


def test_held_key_still_emits_in_a_window_with_no_events() -> None:
    """Edge-triggered HID must become level-triggered per-step conditioning."""
    canonicalizer = InputCanonicalizer([KeyboardToDriverCommand()])
    inputs = UserInputs(events=(_key("key_down", "w", 0.1),))
    canonicalizer.canonicalize(inputs, window=WINDOW, source_schema=KEYBOARD_SOURCE)

    quiet = canonicalizer.canonicalize(
        inputs, window=NEXT_WINDOW, source_schema=KEYBOARD_SOURCE
    )

    assert _command(quiet)["throttle"] == 1.0


def test_key_release_returns_to_neutral() -> None:
    canonicalizer = InputCanonicalizer([KeyboardToDriverCommand()])
    inputs = UserInputs(events=(_key("key_down", "a", 0.1), _key("key_up", "a", 1.5)))
    canonicalizer.canonicalize(inputs, window=WINDOW, source_schema=KEYBOARD_SOURCE)

    released = canonicalizer.canonicalize(
        inputs, window=NEXT_WINDOW, source_schema=KEYBOARD_SOURCE
    )

    assert _command(released)["steer"] == 0.0


def test_reset_drops_device_state() -> None:
    canonicalizer = InputCanonicalizer([KeyboardToDriverCommand()])
    inputs = UserInputs(events=(_key("key_down", "w", 0.1),))
    canonicalizer.canonicalize(inputs, window=WINDOW, source_schema=KEYBOARD_SOURCE)

    canonicalizer.reset()
    after = canonicalizer.canonicalize(
        UserInputs(), window=NEXT_WINDOW, source_schema=KEYBOARD_SOURCE
    )

    assert _command(after)["throttle"] == 0.0


# --- global conditioning ------------------------------------------------


def test_global_conditioning_is_emitted_only_when_it_changes() -> None:
    """A quiet window must not look like a repeated update request."""
    canonicalizer = InputCanonicalizer(
        [LatestEventToModality(modality=CONDITIONING_PROMPT, event_type="prompt_set")]
    )
    inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.2, event_type="prompt_set", payload={"prompt": "rain"}
            ),
        )
    )

    changed = canonicalizer.canonicalize(
        inputs, window=WINDOW, source_schema=PROMPT_SOURCE
    )
    quiet = canonicalizer.canonicalize(
        inputs, window=NEXT_WINDOW, source_schema=PROMPT_SOURCE
    )

    assert changed.has_global_change
    assert changed.global_conditioning["conditioning_prompt"]["prompt"] == "rain"
    assert not quiet.has_global_change


def test_global_converter_rejects_a_per_step_modality() -> None:
    with pytest.raises(ValueError, match="global conditioning"):
        LatestEventToModality(modality=DRIVER_COMMAND, event_type="wheel_axis")


def test_global_and_per_step_land_in_separate_slots() -> None:
    canonicalizer = InputCanonicalizer(
        [
            KeyboardToDriverCommand(),
            LatestEventToModality(
                modality=CONDITIONING_PROMPT, event_type="prompt_set"
            ),
        ]
    )
    source = UserInputSchema(
        capabilities=KEYBOARD_SOURCE.capabilities + PROMPT_SOURCE.capabilities
    )
    inputs = UserInputs(
        events=(
            _key("key_down", "w", 0.1),
            UserInputEvent(
                timestamp_s=0.2, event_type="prompt_set", payload={"prompt": "rain"}
            ),
        )
    )

    canonical = canonicalizer.canonicalize(inputs, window=WINDOW, source_schema=source)

    assert set(canonical.per_step) == {"driver_command"}
    assert set(canonical.global_conditioning) == {"conditioning_prompt"}


# --- device independence ------------------------------------------------


def test_mapping_written_against_a_modality_accepts_a_keyboard() -> None:
    canonicalizer = InputCanonicalizer([KeyboardToDriverCommand()])

    compatibility = check_mapping_compatibility(
        canonical_schema=canonicalizer.canonical_schema(KEYBOARD_SOURCE),
        model_schema=STEERING_MODEL,
        mapping_schema=STEERING_MAPPING,
    )

    assert compatibility.can_drive


def test_adding_a_device_needs_no_application_or_model_change() -> None:
    """A wheel is one register() call; mapping and model schemas are untouched."""
    canonicalizer = InputCanonicalizer([KeyboardToDriverCommand()])
    canonicalizer.register(WheelToDriverCommand())

    compatibility = check_mapping_compatibility(
        canonical_schema=canonicalizer.canonical_schema(WHEEL_SOURCE),
        model_schema=STEERING_MODEL,
        mapping_schema=STEERING_MAPPING,
    )
    assert compatibility.can_drive

    canonical = canonicalizer.canonicalize(
        UserInputs(
            events=(
                UserInputEvent(
                    timestamp_s=0.5,
                    event_type="wheel_axis",
                    payload={"axis": "steer", "value": -0.4},
                ),
            )
        ),
        window=WINDOW,
        source_schema=WHEEL_SOURCE,
    )
    assert _command(canonical)["steer"] == pytest.approx(-0.4)


def test_source_with_no_feedable_converter_supplies_no_modalities() -> None:
    canonicalizer = InputCanonicalizer([WheelToDriverCommand()])

    schema = canonicalizer.canonical_schema(KEYBOARD_SOURCE)

    assert schema.modalities == ()
    assert not schema.supports(DRIVER_COMMAND)
    assert canonicalizer.unavailable_converters(KEYBOARD_SOURCE)


def test_highest_priority_device_wins_when_both_are_present() -> None:
    canonicalizer = InputCanonicalizer(
        [KeyboardToDriverCommand(), WheelToDriverCommand()]
    )
    both = UserInputSchema(
        capabilities=KEYBOARD_SOURCE.capabilities + WHEEL_SOURCE.capabilities
    )

    canonical = canonicalizer.canonicalize(
        UserInputs(
            events=(
                _key("key_down", "a", 0.2),
                UserInputEvent(
                    timestamp_s=0.5,
                    event_type="wheel_axis",
                    payload={"axis": "steer", "value": -0.4},
                ),
            )
        ),
        window=WINDOW,
        source_schema=both,
    )

    assert canonical.metadata["canonical_sources"]["driver_command"] == "wheel"
    assert _command(canonical)["steer"] == pytest.approx(-0.4)


def test_preempted_device_keeps_its_state_current() -> None:
    """Keyboard state must not be stale when the wheel disappears."""
    canonicalizer = InputCanonicalizer(
        [KeyboardToDriverCommand(), WheelToDriverCommand()]
    )
    both = UserInputSchema(
        capabilities=KEYBOARD_SOURCE.capabilities + WHEEL_SOURCE.capabilities
    )
    inputs = UserInputs(
        events=(
            _key("key_down", "w", 0.2),
            UserInputEvent(
                timestamp_s=0.5,
                event_type="wheel_axis",
                payload={"axis": "steer", "value": -0.4},
            ),
        )
    )
    preempted = canonicalizer.canonicalize(inputs, window=WINDOW, source_schema=both)
    assert preempted.metadata["canonical_sources"]["driver_command"] == "wheel"

    keyboard_only = canonicalizer.canonicalize(
        inputs, window=NEXT_WINDOW, source_schema=KEYBOARD_SOURCE
    )

    assert keyboard_only.metadata["canonical_sources"]["driver_command"] == "keyboard"
    assert _command(keyboard_only)["throttle"] == 1.0


# --- registry -----------------------------------------------------------


def test_duplicate_converter_names_are_rejected() -> None:
    canonicalizer = InputCanonicalizer([KeyboardToDriverCommand()])

    with pytest.raises(ValueError, match="already registered"):
        canonicalizer.register(KeyboardToDriverCommand())


def test_converter_must_fill_the_declared_modality_payload() -> None:
    modality = CanonicalModality(
        name="steering_wheel", payload_fields=frozenset({"steer", "throttle"})
    )

    with pytest.raises(ValueError, match="requires payload fields"):
        modality.value({"steer": 0.0})


def test_new_modality_is_a_registration_not_a_core_change() -> None:
    pedals = CanonicalModality(
        name="pedal_state", payload_fields=frozenset({"throttle"})
    )

    class PedalsConverter:
        schema = DeviceConverterSchema(
            name="pedals",
            produces=pedals,
            device_kind="pedals",
            consumes=(
                UserInputCapability(
                    event_type="pedal_axis",
                    payload_fields=frozenset({"value"}),
                ),
            ),
        )

        def reset(self) -> None:
            return None

        def convert(
            self, user_inputs: UserInputs, window: TimeWindow
        ) -> Mapping[str, Any] | None:
            del window
            if not user_inputs.events:
                return None
            return pedals.value(
                {"throttle": float(user_inputs.events[-1].payload["value"])}
            )

    source = UserInputSchema(
        capabilities=(
            UserInputCapability(
                event_type="pedal_axis", payload_fields=frozenset({"value"})
            ),
        )
    )
    canonicalizer = InputCanonicalizer([PedalsConverter()])

    assert canonicalizer.canonical_schema(source).modalities == (pedals,)
    canonical = canonicalizer.canonicalize(
        UserInputs(
            events=(
                UserInputEvent(
                    timestamp_s=0.5, event_type="pedal_axis", payload={"value": 0.75}
                ),
            )
        ),
        window=WINDOW,
        source_schema=source,
    )
    assert canonical.per_step["pedal_state"]["throttle"] == pytest.approx(0.75)


def test_replaying_the_same_windows_reproduces_the_same_canonical_inputs() -> None:
    inputs = UserInputs(events=(_key("key_down", "w", 0.1), _key("key_down", "a", 1.2)))

    def run() -> list[dict[str, Any]]:
        canonicalizer = InputCanonicalizer([KeyboardToDriverCommand()])
        return [
            dict(
                _command(
                    canonicalizer.canonicalize(
                        inputs, window=window, source_schema=KEYBOARD_SOURCE
                    )
                )
            )
            for window in (WINDOW, NEXT_WINDOW)
        ]

    assert run() == run()
