# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-video backend selection for the unified demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .presets import PRESETS, T2VPreset


@dataclass(frozen=True, slots=True)
class T2VBackend:
    """One supported T2V model family and its app-owned presets."""

    key: str
    label: str
    default_preset_name: str
    preset_names: tuple[str, ...]

    def resolve_runner(self, preset_name: str | None = None) -> T2VPreset:
        """Return a preset without importing an integration package."""
        name = preset_name or self.default_preset_name
        if name not in self.preset_names:
            raise ValueError(
                f"Unknown {self.key} preset {name!r}. Available presets: "
                f"{', '.join(self.preset_names)}."
            )
        return PRESETS[name]


BACKENDS: dict[str, T2VBackend] = {
    "causal-forcing": T2VBackend(
        key="causal-forcing",
        label="Causal-Forcing (Wan 2.1)",
        default_preset_name="causal-forcing-wan2.1-t2v-1.3b-chunkwise",
        preset_names=(
            "causal-forcing-wan2.1-t2v-1.3b-chunkwise",
            "causal-forcing-wan2.1-t2v-1.3b-framewise",
        ),
    ),
    "cosmos-predict2": T2VBackend(
        key="cosmos-predict2",
        label="Cosmos Predict2",
        default_preset_name="cosmos2-t2v-2b-720p",
        preset_names=("cosmos2-t2v-2b-720p",),
    ),
    "self-forcing": T2VBackend(
        key="self-forcing",
        label="Self-Forcing (Wan 2.1)",
        default_preset_name="self-forcing-wan2.1-t2v-1.3b",
        preset_names=(
            "self-forcing-wan2.1-t2v-1.3b",
            "self-forcing-wan2.1-t2v-1.3b-taehv",
            "self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope",
        ),
    ),
}


def resolve_backend(value: str) -> T2VBackend:
    """Resolve a CLI/UI backend key."""
    try:
        return BACKENDS[value]
    except KeyError as exc:
        raise ValueError(
            f"Unknown backend {value!r}. Available backends: {', '.join(BACKENDS)}."
        ) from exc


def backend_choices() -> tuple[str, ...]:
    """Return stable CLI choices."""
    return tuple(BACKENDS)


def backend_metadata() -> list[dict[str, Any]]:
    """Return browser-safe backend names and app-owned presets."""
    return [
        {
            "key": backend.key,
            "label": backend.label,
            "default_preset": backend.default_preset_name,
            "presets": backend.preset_names,
        }
        for backend in BACKENDS.values()
    ]
