# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving import output_targets as output_targets_module
from flashdreams.serving.output_targets import (
    OutputLaunchOptions,
    OutputTargetSpec,
    OutputTargetUnavailableError,
    available_output_modes,
    launch_output_target,
    resolve_output_target,
)

pytestmark = pytest.mark.ci_cpu


def _runner_config(
    *,
    runner_name: str,
    pipeline_name: str | None = None,
    num_views: int = 1,
    compile_network: bool = True,
) -> RunnerConfig:
    transformer = SimpleNamespace(
        num_views=num_views,
        compile_network=compile_network,
    )
    pipeline = SimpleNamespace(
        name=pipeline_name or runner_name,
        diffusion_model=SimpleNamespace(seed=42, transformer=transformer),
    )
    return cast(
        RunnerConfig,
        SimpleNamespace(
            runner_name=runner_name,
            pipeline=pipeline,
            device="cuda:1",
            pixel_height=480,
            pixel_width=832,
            fps=20,
            output_fps=24,
            example_idx=3,
            postprocess=SimpleNamespace(preset=""),
        ),
    )


def test_lingbot_webrtc_target_translates_runner_config() -> None:
    config = _runner_config(
        runner_name="lingbot-world-fast",
        compile_network=False,
    )

    spec = resolve_output_target(
        config,
        mode="webrtc",
        options=OutputLaunchOptions(
            host="127.0.0.1",
            port=9010,
            prefer_sw_encoder=True,
        ),
    )

    assert spec.module == "lingbot.demo.cli"
    assert spec.argv == (
        "webrtc",
        "--preset-id",
        "lingbot-world-fast",
        "--device",
        "cuda:1",
        "--fps",
        "20",
        "--video-height",
        "480",
        "--video-width",
        "832",
        "--no-compile",
        "--example-idx",
        "3",
        "--host",
        "127.0.0.1",
        "--port",
        "9010",
        "--prefer-sw-encoder",
    )


def test_omnidreams_webrtc_target_rejects_multi_view() -> None:
    config = _runner_config(
        runner_name="omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae",
        num_views=4,
    )

    assert available_output_modes(config) == ("cli",)
    with pytest.raises(OutputTargetUnavailableError, match="Supported modes: cli"):
        resolve_output_target(
            config,
            mode="webrtc",
        )


def test_omnidreams_local_window_target_uses_manifest_override() -> None:
    config = _runner_config(
        runner_name="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
    )
    config.postprocess.preset = "flashvsr-v1.1-sparse-2.0"

    spec = resolve_output_target(
        config,
        mode="local-window",
        options=OutputLaunchOptions(local_window_manifest=Path("custom.yaml")),
    )

    assert spec.module == "omnidreams.interactive_drive"
    assert spec.argv == (
        "--manifest",
        "custom.yaml",
        "--postprocess-preset",
        "flashvsr-v1.1-sparse-2.0",
    )
    assert spec.notes


def test_output_manifest_extends_local_window_availability() -> None:
    config = _runner_config(runner_name="omnidreams-sv-2steps-chunk3-loc6-vae-vae")
    options = OutputLaunchOptions(local_window_manifest=Path("custom.yaml"))

    assert available_output_modes(
        config,
        options,
    ) == ("cli", "webrtc", "local-window")


def test_launch_output_target_runs_module_with_translated_argv(monkeypatch) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    original_argv = list(sys.argv)

    def fake_run_module(module: str, *, run_name: str) -> None:
        calls.append((module, run_name, tuple(sys.argv)))

    monkeypatch.setattr(output_targets_module.runpy, "run_module", fake_run_module)

    launch_output_target(
        OutputTargetSpec(
            mode="webrtc",
            label="test",
            module="demo.server",
            argv=("--port", "9000"),
        )
    )

    assert calls == [("demo.server", "__main__", ("demo.server", "--port", "9000"))]
    assert sys.argv == original_argv
