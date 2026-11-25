# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keyboard state helpers shared by current applications."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SUPPORTED_KEYS = frozenset({"w", "a", "s", "d", "q", "e", "i", "k", "j", "l"})
DRIVING_SUPPORTED_KEYS = frozenset(
    {"w", "a", "s", "d", "up", "down", "left", "right", "space"}
)
KEY_ALIASES = {
    " ": "space",
    "arrowup": "w",
    "arrowleft": "a",
    "arrowdown": "s",
    "arrowright": "d",
}


def normalize_key(key: str) -> str:
    normalized = key.lower()
    if normalized.strip() == "spacebar":
        return "space"
    return KEY_ALIASES.get(normalized, normalized.strip())


@dataclass(slots=True)
class KeyboardState:
    pressed_keys: set[str] = field(default_factory=set)
    supported_keys: frozenset[str] = DEFAULT_SUPPORTED_KEYS
    _press_order: dict[str, int] = field(default_factory=dict)
    _press_counter: int = 0
    _pressed_sources: dict[str, set[str]] = field(default_factory=dict)

    def apply_event(self, *, event: str, key: str) -> bool:
        normalized_key = normalize_key(key)
        if normalized_key not in self.supported_keys:
            return False

        normalized_event = event.strip().lower()
        source_key = key.strip().lower()
        if normalized_event == "keydown":
            self._pressed_sources.setdefault(normalized_key, set()).add(source_key)
            self.pressed_keys.add(normalized_key)
            self._press_counter += 1
            self._press_order[normalized_key] = self._press_counter
            return True
        if normalized_event == "keyup":
            sources = self._pressed_sources.get(normalized_key)
            if sources is not None:
                sources.discard(source_key)
                if not sources:
                    self._pressed_sources.pop(normalized_key, None)
            if normalized_key not in self._pressed_sources:
                self.pressed_keys.discard(normalized_key)
                self._press_order.pop(normalized_key, None)
            return True
        return False

    def snapshot(self) -> frozenset[str]:
        return frozenset(self.pressed_keys)

    def _latest_pressed(self, keys: tuple[str, ...]) -> str | None:
        latest_key: str | None = None
        latest_idx = -1
        for key in keys:
            if key not in self.pressed_keys:
                continue
            idx = self._press_order.get(key, -1)
            if idx >= latest_idx:
                latest_idx = idx
                latest_key = key
        return latest_key

    def resolved_effective_keys(self) -> frozenset[str]:
        effective: set[str] = set()
        for key in (
            self._latest_pressed(("w", "s")),
            self._latest_pressed(("a", "d", "j", "l")),
            self._latest_pressed(("q", "e")),
            self._latest_pressed(("i", "k")),
        ):
            if key is not None:
                effective.add(key)
        return frozenset(key for key in effective if key in self.supported_keys)


__all__ = [
    "DEFAULT_SUPPORTED_KEYS",
    "DRIVING_SUPPORTED_KEYS",
    "KEY_ALIASES",
    "KeyboardState",
    "normalize_key",
]
