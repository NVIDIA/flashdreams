# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keyboard state helpers shared by runtime input canonicalizers."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SUPPORTED_KEYS = frozenset({"w", "a", "s", "d", "q", "e", "i", "k", "j", "l"})
DRIVING_SUPPORTED_KEYS = frozenset(
    {"w", "a", "s", "d", "up", "down", "left", "right", "space"}
)
WSAD_SUPPORTED_KEYS = frozenset({"w", "a", "s", "d"})
KEY_ALIASES = {
    " ": "space",
    "arrowup": "w",
    "arrowleft": "a",
    "arrowdown": "s",
    "arrowright": "d",
}


@dataclass(frozen=True, slots=True)
class ResetRequest:
    """Transport-neutral request to reset the realtime rollout."""

    reason: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class PromptRequest:
    """Transport-neutral prompt update request."""

    prompt: str
    negative_prompt: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ImageRequest:
    """Transport-neutral image update request."""

    data: bytes
    content_type: str
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class SparseInputSnapshot:
    """Sparse input state sampled at a realtime loop boundary."""

    timestamp_s: float
    pressed_keys: frozenset[str] = field(default_factory=frozenset)
    effective_keys: frozenset[str] = field(default_factory=frozenset)
    reset: ResetRequest | None = None
    prompt: PromptRequest | None = None
    image: ImageRequest | None = None


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

    def sparse_snapshot(self, *, timestamp_s: float) -> SparseInputSnapshot:
        return SparseInputSnapshot(
            timestamp_s=timestamp_s,
            pressed_keys=self.snapshot(),
            effective_keys=self.resolved_effective_keys(),
        )

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
    "ImageRequest",
    "KEY_ALIASES",
    "KeyboardState",
    "PromptRequest",
    "ResetRequest",
    "SparseInputSnapshot",
    "WSAD_SUPPORTED_KEYS",
    "normalize_key",
]
