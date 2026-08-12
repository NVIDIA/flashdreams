# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved runner-launch arguments for replay and WebRTC demos."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import DemoSpec, OutputSpec

from .launch import LaunchOptions


@dataclass(frozen=True, slots=True)
class DemoLaunchArguments:
    """Resolve runner fields and manifest overrides for one demo launch."""

    config: RunnerConfig
    """Resolved runner configuration parsed by ``flashdreams-run``."""

    options: LaunchOptions
    """Host, port, scenario, and output settings parsed by ``flashdreams-run``."""

    def scenario(self, defaults: Mapping[str, object]) -> dict[str, object]:
        """Merge scenario overrides, runner fields, and model defaults.

        Args:
            defaults: Model-specific fallback values keyed by scenario field.

        Returns:
            Resolved scenario values, with manifest settings taking precedence.
        """
        resolved: dict[str, object] = {}
        for name, default in defaults.items():
            configured = getattr(self.config, name, None)
            resolved[name] = self.options.scenario.get(
                name,
                default if configured is None else configured,
            )
        return resolved

    def output(self, name: str, default: object) -> object:
        """Return an output override or its fallback value."""
        return self.options.output.get(name, default)

    def host(self, default: str) -> str:
        """Return the CLI host, output override, or supplied default."""
        return str(self.options.host or self.output("host", default))

    def port(self, default: int) -> int:
        """Return the CLI port, output override, or supplied default."""
        return int(self.options.port or self.output("port", default))

    def spec(
        self,
        *,
        model_id: str,
        preset_id: str | None,
        input_mode: Literal["replay", "webrtc"],
        scenario: object,
        output: OutputSpec,
        runtime_options: Mapping[str, object],
        compile: bool | None = None,
        device: str | None = None,
    ) -> DemoSpec:
        """Build a common demo spec from the resolved launch arguments.

        Args:
            model_id: Stable runtime adapter identity.
            preset_id: Selected model preset.
            input_mode: Shared demo input mode.
            scenario: Model-owned resolved scenario.
            output: Selected output specification.
            runtime_options: Model-owned runtime options.
            compile: Optional model compile setting.
            device: Runtime device; ``None`` uses ``config.device``.

        Returns:
            Demo specification with its matching inference configuration.
        """
        return DemoSpec(
            model_id=model_id,
            preset_id=preset_id,
            input_mode=input_mode,
            scenario=scenario,
            output=output,
            config=InferenceConfig(
                model_id=model_id,
                preset_id=preset_id,
                device=device or self.config.device,
                compile=compile,
                runtime_options=runtime_options,
            ),
        )


__all__ = ["DemoLaunchArguments"]
