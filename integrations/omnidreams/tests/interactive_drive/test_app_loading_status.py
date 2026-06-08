# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import threading
from types import SimpleNamespace

from omnidreams.interactive_drive.app import InteractiveDriveApp


def _event(set_: bool) -> threading.Event:
    event = threading.Event()
    if set_:
        event.set()
    return event


def _app(
    *, optimizes: bool, model_ready: bool, first_chunk: bool
) -> InteractiveDriveApp:
    """Window-less app wired with just the state the loading status reads."""
    app = InteractiveDriveApp.__new__(InteractiveDriveApp)
    app._pipeline = SimpleNamespace(  # type: ignore[attr-defined]
        model_ready=_event(model_ready),
        first_chunk_produced=_event(first_chunk),
    )
    app._backend = SimpleNamespace(optimizes_on_first_chunk=optimizes)  # type: ignore[attr-defined]
    return app


def test_loading_status_world_model_phases() -> None:
    warming = _app(optimizes=True, model_ready=False, first_chunk=False)
    assert warming._loading_status_message() == "Loading world model..."

    # Warm but no first chunk yet -- the phase that previously misread "Loading
    # scene..." while the model was actually autotuning.
    optimizing = _app(optimizes=True, model_ready=True, first_chunk=False)
    assert optimizing._loading_status_message() == "Optimizing world model..."

    loaded = _app(optimizes=True, model_ready=True, first_chunk=True)
    assert loaded._loading_status_message() == "Loading scene..."


def test_loading_status_skips_optimize_phase_for_non_optimizing_backend() -> None:
    raster = _app(optimizes=False, model_ready=True, first_chunk=False)
    assert raster._loading_status_message() == "Loading scene..."
