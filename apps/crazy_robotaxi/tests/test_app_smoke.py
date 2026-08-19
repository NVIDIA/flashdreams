# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

import pytest
from omnidreams_game_engine import _sample_assets
from pyvirtualdisplay.display import Display

_WARMUP_SENTINEL = "[chunk-pipeline] warmup done"
_WORLD_MODEL_LATENCY_SENTINEL = "[flashdreams-session] continue block_index=1"
_WARMUP_TIMEOUT_S = 90.0
_WORLD_MODEL_TIMEOUT_S = 600.0
_LIVE_DURATION_S = 3.0
_SHUTDOWN_TIMEOUT_S = 15.0
_GAME_MAP = (
    Path(__file__).parents[1] / "crazy_robotaxi" / "maps" / "minimal_loop.robotaxi.yaml"
)


def _pump_stream(
    stream: IO[bytes],
    sink: list[str],
    lock: threading.Lock,
) -> None:
    for raw_line in iter(stream.readline, b""):
        line = raw_line.decode("utf-8", errors="replace")
        with lock:
            sink.append(line)
        if (
            "[presenter] device=" in line
            and _sample_assets.captured_presenter_device is None
        ):
            _sample_assets.captured_presenter_device = line.strip()
        sys.stderr.write(f"[app-smoke] {line}")
        sys.stderr.flush()


def _joined(sink: list[str], lock: threading.Lock) -> str:
    with lock:
        return "".join(sink)


def _has_sentinel(sink: list[str], lock: threading.Lock, sentinel: str) -> bool:
    with lock:
        return any(sentinel in line for line in sink)


def _wait_for_sentinel(
    *,
    process: subprocess.Popen[bytes],
    output_lines: list[str],
    output_lock: threading.Lock,
    sentinel: str,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"App exited before sentinel '{sentinel}' with code {process.returncode}:\n"
                f"{_joined(output_lines, output_lock)}"
            )
        if _has_sentinel(output_lines, output_lock, sentinel):
            return
        time.sleep(0.1)
    raise AssertionError(
        f"Did not observe '{sentinel}' within {timeout_s:.0f}s:\n"
        f"{_joined(output_lines, output_lock)}"
    )


def _run_raster_ui_smoke(map_path: Path) -> None:
    """Drive the full interactive_drive app subprocess under Xvfb against ``map_path``
    and assert it warms up, stays alive, and shuts down cleanly on SIGTERM.

    Does NOT validate raster output correctness - see
    ``test_raster_reference_image.py`` for that."""
    display = Display(backend="xvfb", size=(1280, 720), visible=False)
    display.start()
    try:
        env = os.environ.copy()
        assert "DISPLAY" in env, (
            "pyvirtualdisplay did not publish DISPLAY after start()"
        )

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "omnidreams_game_engine",
                # ``python -m omnidreams_game_engine`` now goes through the
                # demo wrapper which opens a pygame HUD by default and
                # only spawns the backend when the user clicks ``Load
                # Scene`` (or passes ``--auto-start``). The smoke
                # test exercises the bare backend, so opt out of the
                # HUD; the raster backend then prints the warmup
                # sentinel directly to this process's stdout.
                "--no-hud",
                "--map",
                str(map_path),
                "--backend",
                "raster",
                "--camera",
                "camera_front_wide_120fov",
                "--variant",
                "default",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        assert process.stdout is not None

        output_lines: list[str] = []
        output_lock = threading.Lock()
        reader = threading.Thread(
            target=_pump_stream,
            args=(process.stdout, output_lines, output_lock),
            name="interactive_drive-smoke-reader",
            daemon=True,
        )
        reader.start()

        try:
            _wait_for_sentinel(
                process=process,
                output_lines=output_lines,
                output_lock=output_lock,
                sentinel=_WARMUP_SENTINEL,
                timeout_s=_WARMUP_TIMEOUT_S,
            )

            live_deadline = time.monotonic() + _LIVE_DURATION_S
            while time.monotonic() < live_deadline:
                if process.poll() is not None:
                    raise AssertionError(
                        f"App exited while running with code {process.returncode}:\n"
                        f"{_joined(output_lines, output_lock)}"
                    )
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            reader.join(timeout=5.0)

        output = _joined(output_lines, output_lock)
        assert "Traceback (most recent call last)" not in output, (
            f"App logged a Python traceback:\n{output}"
        )
        assert process.returncode in (0, -signal.SIGTERM, -signal.SIGKILL), (
            f"Unexpected exit code {process.returncode}:\n{output}"
        )
    finally:
        display.stop()


def _run_synthetic_world_model_latency_smoke(map_path: Path) -> None:
    """Run the synthetic world model through the interactive-drive latency path."""
    display = Display(backend="xvfb", size=(1280, 720), visible=False)
    display.start()
    try:
        env = os.environ.copy()
        assert "DISPLAY" in env, (
            "pyvirtualdisplay did not publish DISPLAY after start()"
        )
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "omnidreams_game_engine",
                "--no-hud",
                "--map",
                str(map_path),
                "--backend",
                "omnidreams",
                "--manifest",
                "example_world_model_synthetic.yaml",
                "--synthetic-model",
                "--profile-world-model",
                "--stop-after-chunks",
                "2",
                "--camera",
                "camera_front_wide_120fov",
                "--variant",
                "default",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        assert process.stdout is not None

        output_lines: list[str] = []
        output_lock = threading.Lock()
        reader = threading.Thread(
            target=_pump_stream,
            args=(process.stdout, output_lines, output_lock),
            name="interactive_drive-world-model-smoke-reader",
            daemon=True,
        )
        reader.start()

        try:
            _wait_for_sentinel(
                process=process,
                output_lines=output_lines,
                output_lock=output_lock,
                sentinel=_WORLD_MODEL_LATENCY_SENTINEL,
                timeout_s=_WORLD_MODEL_TIMEOUT_S,
            )
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            reader.join(timeout=5.0)

        output = _joined(output_lines, output_lock)
        assert "Traceback (most recent call last)" not in output, (
            f"App logged a Python traceback:\n{output}"
        )
        assert process.returncode in (0, -signal.SIGTERM, -signal.SIGKILL), (
            f"Unexpected exit code {process.returncode}:\n{output}"
        )
    finally:
        display.stop()


@pytest.mark.gpu
@pytest.mark.xvfb
def test_interactive_drive_raster_ui_smoke() -> None:
    _run_raster_ui_smoke(_GAME_MAP)


# gpu + xvfb -> routed to ``manual`` by conftest (see pytest_collection_modifyitems):
# the public GPU CI runner image isn't guaranteed to have Xvfb, so don't pin an
# explicit ``ci_gpu`` tier here. The real CI latency coverage runs internally
# under ``xvfb-run`` in the benchmark job.
@pytest.mark.gpu
@pytest.mark.xvfb
def test_interactive_drive_synthetic_world_model_latency_smoke() -> None:
    _run_synthetic_world_model_latency_smoke(_GAME_MAP)
