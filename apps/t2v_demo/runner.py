# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``flashdreams-run t2v`` configuration and default replay runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import tyro

from flashdreams.infra.runner import Runner, RunnerConfig

from .backends import backend_choices, resolve_backend


@dataclass(kw_only=True)
class T2VDemoRunnerConfig(RunnerConfig):
    """Configuration exposed by the ``flashdreams-run t2v`` slug."""

    _target: type["T2VDemoRunner"] = field(default_factory=lambda: T2VDemoRunner)
    launch_capability: Annotated[str | None, tyro.conf.Suppress] = (
        "t2v_demo.launch:LAUNCH_CAPABILITY"
    )
    pipeline: Annotated[Any, tyro.conf.Suppress] = field(
        default_factory=lambda: resolve_backend("causal-forcing")
        .resolve_runner()
        .pipeline
    )
    backend: str = "causal-forcing"
    """Backend key: one of ``causal-forcing``, ``cosmos-predict2``, or ``self-forcing``."""

    preset_id: str | None = None
    prompt: str | None = None
    total_blocks: int | None = None
    pixel_height: int | None = None
    pixel_width: int | None = None
    fps: int | None = None
    compile: bool | None = None
    output: Path = Path("outputs/t2v.mp4")

    def __post_init__(self) -> None:
        if self.backend not in backend_choices():
            raise ValueError(
                f"Unknown T2V backend {self.backend!r}; choose one of "
                f"{', '.join(backend_choices())}."
            )


class T2VDemoRunner(Runner[T2VDemoRunnerConfig, Any]):
    """Default ``run`` mode delegates to the same app replay entry point."""

    def __init__(self, config: T2VDemoRunnerConfig) -> None:
        # The demo runtime owns pipeline construction so that replay and
        # WebRTC share identical lifecycle handling. Avoid constructing a
        # second Runner pipeline here.
        self.config = config

    def run(self) -> None:
        from .app import launch_t2v

        launch_t2v(config=self.config, mode="mp4")


RUNNER_T2V = T2VDemoRunnerConfig(
    runner_name="t2v",
    description="Text-to-video runtime demo (replay or WebRTC).",
)

__all__ = ["RUNNER_T2V", "T2VDemoRunner", "T2VDemoRunnerConfig"]
