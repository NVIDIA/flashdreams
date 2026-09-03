# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving import launch as launch_module
from flashdreams.serving.launch import (
    ResolvedLaunch,
    available_launch_modes,
    resolve_launch,
)

pytestmark = pytest.mark.ci_cpu


def _runner_config(
    *,
    runner_name: str,
) -> RunnerConfig:
    pipeline = SimpleNamespace(
        name=runner_name,
        diffusion_model=SimpleNamespace(
            seed=42,
            transformer=SimpleNamespace(num_views=1, compile_network=True),
        ),
    )
    return cast(
        RunnerConfig,
        SimpleNamespace(
            runner_name=runner_name,
            launch_capability=None,
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


def test_capabilities_extend_launch_without_shared_routing_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCapability:
        def supported_modes(self, config, options):
            del config, options
            return ("webrtc",)

        def resolve(self, config, *, mode, options):
            del config, options
            if mode != "webrtc":
                return None
            return ResolvedLaunch(
                mode="webrtc",
                label="plugin launch",
                launch=lambda: None,
            )

    config = _runner_config(runner_name="third-party-model")
    config.launch_capability = "plugin:capability"
    monkeypatch.setattr(
        launch_module,
        "_load_launch_capability",
        lambda path: _FakeCapability(),
    )

    assert available_launch_modes(config) == ("run", "webrtc")
    assert resolve_launch(config, mode="webrtc").label == "plugin launch"
