# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import threading
from types import SimpleNamespace

from omnidreams.interactive_drive import demo
from omnidreams.interactive_drive.app import InteractiveDriveApp


def test_hud_loading_backend_build_pumps_until_ready(monkeypatch) -> None:
    args = SimpleNamespace()
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def prepare_config_and_backend(candidate_args):
        assert candidate_args is args
        entered.set()
        assert release.wait(timeout=1.0)
        return "config", "backend"

    class _Presenter:
        @property
        def should_close(self) -> bool:
            return False

        @property
        def pending_scene_change(self) -> None:
            return None

        def present_world_model_loading(self) -> None:
            nonlocal calls
            calls += 1
            if entered.is_set():
                release.set()

    monkeypatch.setattr(
        demo._cli, "prepare_config_and_backend", prepare_config_and_backend
    )
    monkeypatch.setattr(demo, "HUD_LOADING_PRESENT_INTERVAL_S", 0.001)

    assert demo._prepare_config_and_backend_with_hud_loading(args, _Presenter()) == (
        "config",
        "backend",
    )
    assert calls > 0


def test_app_blocking_cleanup_uses_runner() -> None:
    app = InteractiveDriveApp.__new__(InteractiveDriveApp)
    calls: list[str] = []

    def runner(cleanup) -> None:
        calls.append("runner")
        cleanup()

    app._blocking_work_runner = runner
    app._run_blocking_cleanup(lambda: calls.append("cleanup"))

    assert calls == ["runner", "cleanup"]


def test_main_handles_keyboard_interrupt_without_traceback(monkeypatch) -> None:
    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            return SimpleNamespace(
                synthetic_scene=True,
                stream_mjpeg=None,
                no_hud=True,
            )

    exits = 0

    def run(args: SimpleNamespace) -> None:
        del args
        raise KeyboardInterrupt

    def exit_after_keyboard_interrupt() -> None:
        nonlocal exits
        exits += 1

    monkeypatch.setattr(demo, "build_parser", lambda: _Parser())
    monkeypatch.setattr(demo._cli, "run", run)
    monkeypatch.setattr(
        demo, "_exit_after_keyboard_interrupt", exit_after_keyboard_interrupt
    )

    assert demo.main() == 130
    assert exits == 1
