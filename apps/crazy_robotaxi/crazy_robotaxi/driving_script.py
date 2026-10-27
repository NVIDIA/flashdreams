# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timestamped input files for the interactive headless control interface."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from crazy_robotaxi.debug_input import control_event


@dataclass(frozen=True, slots=True)
class DrivingScript:
    """Commands queued before the gameplay-ready clock starts."""

    path: Path
    events: tuple[dict[str, Any], ...]


def load_driving_script(path: Path) -> DrivingScript:
    """Read the same timestamped event objects accepted by the live socket."""
    path = path.expanduser().resolve()
    document = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"events"}:
        raise ValueError(
            "Driving script requires an 'events' list of timestamped inputs"
        )
    events = document["events"]
    if not isinstance(events, list) or not 1 <= len(events) <= 100_000:
        raise ValueError("Driving script 'events' must contain 1 to 100,000 commands")
    ids = set()
    for event in events:
        control_event(event)
        if event["id"] in ids:
            raise ValueError("Driving script contains duplicate input ids")
        ids.add(event["id"])
    return DrivingScript(path, tuple(events))
