# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared local-window demo construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .replay import _require_supported_mode
from .spec import DemoAdapter, DemoSpec, LocalWindowOutputSpec


@runtime_checkable
class LocalWindowApp(Protocol):
    """A windowed demo that owns its own render loop.

    The same shape as the WebRTC server this sits beside: ``run`` blocks
    until the user closes the window, so the launcher hands off rather than
    driving frames itself.
    """

    def run(self) -> Any:
        """Open the window and drive frames until the demo exits."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class LocalWindowDemo:
    """Constructed local-window demo pieces, before or after running."""

    app: LocalWindowApp
    output: LocalWindowOutputSpec


def build_local_window_demo(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
) -> LocalWindowDemo:
    """Build the model-owned windowed app described by ``spec``.

    Raises:
        ValueError: ``spec`` does not select local-window output, or the
            adapter exposes no ``create_local_window_app`` factory.
    """
    if not isinstance(spec.output, LocalWindowOutputSpec):
        raise ValueError(
            "build_local_window_demo requires LocalWindowOutputSpec output."
        )
    _require_supported_mode(
        mode=spec.input_mode,
        supported=adapter.supported_input_modes(),
        label="input_mode",
    )
    _require_supported_mode(
        mode=spec.output.mode,
        supported=adapter.supported_output_modes(),
        label="output.mode",
    )

    factory = getattr(adapter, "create_local_window_app", None)
    if not callable(factory):
        raise ValueError(
            f"Adapter {type(adapter).__name__} does not provide "
            "create_local_window_app; local-window output needs a model-owned "
            "windowed app."
        )
    return LocalWindowDemo(app=factory(spec=spec), output=spec.output)


def run_local_window_demo(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
) -> LocalWindowDemo:
    """Build and run a local-window demo, blocking until the window closes."""
    demo = build_local_window_demo(spec=spec, adapter=adapter)
    demo.app.run()
    return demo


__all__ = [
    "LocalWindowApp",
    "LocalWindowDemo",
    "build_local_window_demo",
    "run_local_window_demo",
]
