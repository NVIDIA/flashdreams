# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output target selection for ``flashdreams-run``."""

from __future__ import annotations

import runpy
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from flashdreams.infra.runner import RunnerConfig

OutputMode: TypeAlias = Literal["cli", "webrtc", "local-window"]

_OMNIDREAMS_LOCAL_WINDOW_MANIFESTS = {
    "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae": ("example_world_model.yaml"),
    "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf": (
        "example_world_model_perf.yaml"
    ),
    "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-native-perf": (
        "example_world_model_perf.yaml"
    ),
}


class OutputTargetUnavailableError(ValueError):
    """Raised when a runner cannot be launched through a requested output."""


@dataclass(frozen=True, slots=True)
class OutputLaunchOptions:
    """Common launch options shared by non-CLI output targets."""

    host: str | None = None
    port: int | None = None
    prefer_sw_encoder: bool = False
    local_window_manifest: Path | None = None


@dataclass(frozen=True, slots=True)
class OutputTargetSpec:
    """A concrete output target module plus argv translated from a runner config."""

    mode: OutputMode
    label: str
    module: str
    argv: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def command(self) -> str:
        """Return a copy-pasteable module command for diagnostics."""
        return shlex.join(("python", "-m", self.module, *self.argv))


def available_output_modes(
    config: RunnerConfig,
    options: OutputLaunchOptions | None = None,
) -> tuple[OutputMode, ...]:
    """Return output modes known to support ``config``."""
    options = options or OutputLaunchOptions()
    modes: list[OutputMode] = ["cli"]
    if _webrtc_spec(config, options) is not None:
        modes.append("webrtc")
    if _local_window_spec(config, options) is not None:
        modes.append("local-window")
    return tuple(modes)


def resolve_output_target(
    config: RunnerConfig,
    *,
    mode: OutputMode,
    options: OutputLaunchOptions | None = None,
) -> OutputTargetSpec:
    """Resolve a non-CLI output target for a runner config."""
    if mode == "cli":
        raise ValueError("CLI mode is run directly by the selected Runner.")
    options = options or OutputLaunchOptions()
    spec = (
        _webrtc_spec(config, options)
        if mode == "webrtc"
        else _local_window_spec(config, options)
    )
    if spec is None:
        supported = ", ".join(available_output_modes(config, options))
        raise OutputTargetUnavailableError(
            f"Output mode {mode!r} is not available for runner "
            f"{config.runner_name!r}. Supported modes: {supported}."
        )
    return spec


def launch_output_target(spec: OutputTargetSpec) -> None:
    """Execute an output target module as if launched with ``python -m``."""
    original_argv = sys.argv
    sys.argv = [spec.module, *spec.argv]
    try:
        runpy.run_module(spec.module, run_name="__main__")
    finally:
        sys.argv = original_argv


def _webrtc_spec(
    config: RunnerConfig,
    options: OutputLaunchOptions,
) -> OutputTargetSpec | None:
    name = _runner_name(config)
    if _is_lingbot_runner(name):
        return _lingbot_webrtc_spec(config, options)
    if _is_omnidreams_runner(name) and _is_omnidreams_single_view(config):
        return _omnidreams_webrtc_spec(config, options)
    return None


def _local_window_spec(
    config: RunnerConfig,
    options: OutputLaunchOptions,
) -> OutputTargetSpec | None:
    name = _runner_name(config)
    if not _is_omnidreams_runner(name):
        return None
    manifest = options.local_window_manifest
    if manifest is None:
        manifest_name = _OMNIDREAMS_LOCAL_WINDOW_MANIFESTS.get(name)
        if manifest_name is None:
            return None
        manifest_arg = manifest_name
    else:
        manifest_arg = str(manifest)

    argv = ["--manifest", manifest_arg]
    _append_postprocess_preset(argv, config)
    return OutputTargetSpec(
        mode="local-window",
        label="Omnidreams local interactive window",
        module="omnidreams.interactive_drive",
        argv=tuple(argv),
        notes=(
            (
                "Local-window uses the Omnidreams interactive-drive manifest for "
                "scene, resolution, and runtime-specific controls."
            ),
        ),
    )


def _lingbot_webrtc_spec(
    config: RunnerConfig,
    options: OutputLaunchOptions,
) -> OutputTargetSpec:
    argv = [
        "webrtc",
        "--preset-id",
        _pipeline_name(config),
        "--device",
        _device(config),
        "--fps",
        str(getattr(config, "fps", 16)),
        "--video-height",
        str(getattr(config, "pixel_height", 464)),
        "--video-width",
        str(getattr(config, "pixel_width", 832)),
    ]
    if _compile_network(config) is False:
        argv.append("--no-compile")
    example_idx = getattr(config, "example_idx", None)
    if example_idx is not None:
        argv.extend(("--example-idx", str(example_idx)))
    if options.host:
        argv.extend(("--host", options.host))
    if options.port is not None:
        argv.extend(("--port", str(options.port)))
    if options.prefer_sw_encoder:
        argv.append("--prefer-sw-encoder")
    return OutputTargetSpec(
        mode="webrtc",
        label="LingBot shared demo WebRTC server",
        module="lingbot.demo.cli",
        argv=tuple(argv),
    )


def _omnidreams_webrtc_spec(
    config: RunnerConfig,
    options: OutputLaunchOptions,
) -> OutputTargetSpec:
    argv = [
        "--pipeline_config_name",
        _pipeline_name(config),
        "--device",
        _device(config),
        "--fps",
        str(getattr(config, "output_fps", 30)),
        "--video_height",
        str(getattr(config, "pixel_height", 704)),
        "--video_width",
        str(getattr(config, "pixel_width", 1280)),
    ]
    seed = _diffusion_seed(config)
    if seed is not None:
        argv.extend(("--seed", str(seed)))
    _append_postprocess_preset(argv, config)
    _append_webrtc_bind_args(argv, options)
    return OutputTargetSpec(
        mode="webrtc",
        label="Omnidreams WebRTC server",
        module="omnidreams.webrtc.server",
        argv=tuple(argv),
    )


def _append_webrtc_bind_args(
    argv: list[str],
    options: OutputLaunchOptions,
) -> None:
    if options.host:
        argv.extend(("--host", options.host))
    if options.port is not None:
        argv.extend(("--port", str(options.port)))
    if options.prefer_sw_encoder:
        argv.append("--prefer_sw_encoder")


def _append_postprocess_preset(argv: list[str], config: RunnerConfig) -> None:
    preset = getattr(getattr(config, "postprocess", None), "preset", "")
    if preset:
        argv.extend(("--postprocess-preset", str(preset)))


def _runner_name(config: RunnerConfig) -> str:
    return str(getattr(config, "runner_name", ""))


def _pipeline_name(config: RunnerConfig) -> str:
    pipeline = getattr(config, "pipeline", None)
    name = getattr(pipeline, "name", None)
    return str(name or config.runner_name)


def _device(config: RunnerConfig) -> str:
    return str(getattr(config, "device", "cuda"))


def _compile_network(config: RunnerConfig) -> bool | None:
    transformer = _transformer_config(config)
    value = getattr(transformer, "compile_network", None)
    return None if value is None else bool(value)


def _diffusion_seed(config: RunnerConfig) -> int | None:
    diffusion_model = getattr(
        getattr(config, "pipeline", None), "diffusion_model", None
    )
    seed = getattr(diffusion_model, "seed", None)
    return None if seed is None else int(seed)


def _transformer_config(config: RunnerConfig) -> Any:
    diffusion_model = getattr(
        getattr(config, "pipeline", None), "diffusion_model", None
    )
    return getattr(diffusion_model, "transformer", None)


def _is_lingbot_runner(name: str) -> bool:
    return name.startswith("lingbot-world")


def _is_omnidreams_runner(name: str) -> bool:
    return name.startswith("omnidreams-")


def _is_omnidreams_single_view(config: RunnerConfig) -> bool:
    num_views = getattr(_transformer_config(config), "num_views", 1)
    return int(num_views) == 1


__all__ = [
    "OutputLaunchOptions",
    "OutputMode",
    "OutputTargetSpec",
    "OutputTargetUnavailableError",
    "available_output_modes",
    "launch_output_target",
    "resolve_output_target",
]
