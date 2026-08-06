# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared local-window demo construction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .replay import _require_supported_mode
from .spec import DemoAdapter, DemoSpec, LocalWindowOutputSpec

if TYPE_CHECKING:
    from flashdreams.serving.presentation import DisplayFrame


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

    app_factory = getattr(adapter, "create_local_window_app", None)
    if callable(app_factory):
        return LocalWindowDemo(app=app_factory(spec=spec), output=spec.output)

    overlay_factory = getattr(adapter, "create_local_window_overlay", None)
    if callable(overlay_factory):
        return LocalWindowDemo(
            app=_OverlayDrivenApp(
                overlay=overlay_factory(spec=spec),
                frames=adapter.local_window_frames(spec=spec),
                output=spec.output,
            ),
            output=spec.output,
        )

    raise ValueError(
        f"Adapter {type(adapter).__name__} supports local-window output but "
        "provides neither create_local_window_app nor "
        "create_local_window_overlay."
    )


@dataclass(slots=True)
class _OverlayDrivenApp:
    """Default loop for adapters that only supply chrome and a frame stream.

    Enough for a demo whose interaction is "show me frames as they arrive" --
    a text-to-video preview, or a model whose controls already reach it by
    another route. Adapters needing their own outer loop, such as one that
    switches scenes over a long-lived window, supply a full app instead.
    """

    overlay: Any
    frames: Iterable["DisplayFrame"]
    output: LocalWindowOutputSpec

    def run(self) -> None:
        from flashdreams.serving.presentation import (
            LocalWindowPresenter,
            WindowConfig,
        )

        presenter = LocalWindowPresenter(
            overlay=self.overlay,
            config=WindowConfig(
                width=self.output.width,
                height=self.output.height,
                title=self.output.title,
            ),
        )
        try:
            for frame in self.frames:
                if presenter.should_close:
                    break
                presenter.process_events()
                if presenter.should_close:
                    break
                presenter.present_frame(frame)
        finally:
            presenter.close()


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
