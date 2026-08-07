# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed demo-adapter discovery through Python entry points."""

from __future__ import annotations

from importlib import metadata
from typing import Any

from flashdreams.runtime.demo.spec import DemoAdapter

DEMO_ADAPTER_ENTRY_POINT_GROUP = "flashdreams.demo_adapters"
"""Entry-point group for installed model demo adapters."""


def discover_demo_adapters() -> dict[str, DemoAdapter]:
    """Load installed demo adapters keyed by stable model identity."""
    adapters: dict[str, DemoAdapter] = {}
    entry_points = metadata.entry_points()
    selected = (
        entry_points.select(group=DEMO_ADAPTER_ENTRY_POINT_GROUP)
        if hasattr(entry_points, "select")
        else entry_points.get(DEMO_ADAPTER_ENTRY_POINT_GROUP, ())
    )
    for entry_point in selected:
        adapter = _materialize_adapter(entry_point.load())
        if adapter.model_id in adapters:
            raise ValueError(f"Duplicate demo adapter model_id={adapter.model_id!r}.")
        adapters[adapter.model_id] = adapter
    return adapters


def _materialize_adapter(value: Any) -> DemoAdapter:
    candidate = value() if isinstance(value, type) or callable(value) else value
    required = (
        "model_id",
        "inference_input_schema",
        "inference_output_schema",
        "supported_routes",
        "prepare_session",
        "create_demo_runtime",
        "create_runtime",
        "list_sessions",
    )
    missing = tuple(name for name in required if not hasattr(candidate, name))
    if missing:
        raise TypeError(
            f"Demo adapter {type(candidate).__name__} is missing: {missing}."
        )
    return candidate


__all__ = ["DEMO_ADAPTER_ENTRY_POINT_GROUP", "discover_demo_adapters"]
