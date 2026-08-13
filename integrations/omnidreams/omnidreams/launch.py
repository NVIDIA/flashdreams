# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams launch capability for ``flashdreams-run``."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving.launch import (
    LaunchMode,
    LaunchOptions,
    ResolvedLaunch,
)

_LOCAL_WINDOW_MANIFESTS = {
    "omnidreams": "example_world_model.yaml",
    "omnidreams-perf": "example_world_model_perf.yaml",
}
_DEFAULT_MP4_OUTPUT_PATH = Path("outputs/omnidreams.mp4")
_REPLAY_OUTPUT_FIELDS = frozenset({"path", "output", "fps", "stats_path", "stats_dir"})
_REPLAY_SCENARIO_FIELDS = frozenset(
    {
        "conditioning_mode",
        "prompt",
        "hdmap_video_paths",
        "first_frame_paths",
        "camera_names",
        "keyboard_trace",
        "scene_path",
        "scene_dir",
        "scene_uuid",
        "scene_variant",
        "camera_name",
        "move_speed_per_s",
        "rotate_speed_rad_per_s",
        "ludus_backend",
        "example_data",
        "example_data_uuid",
        "total_blocks",
        "pixel_height",
        "pixel_width",
        "fps",
    }
)
_WEBRTC_SCENARIO_FIELDS = frozenset(
    {"scene_dir", "scene_uuid", "scene_variant", "camera_name"}
)
_WEBRTC_OUTPUT_FIELDS = frozenset(
    {
        "host",
        "port",
        "fps",
        "video_height",
        "video_width",
        "warmup_chunks",
        "warmup_timeout_s",
        "client_liveness_timeout_s",
        "debug_serve_hdmaps",
        "prefer_sw_encoder",
    }
)
_LOCAL_SCENARIO_FIELDS = frozenset(
    {
        "scene",
        "scene_dir",
        "camera",
        "variant",
        "prompt",
        "synthetic_scene",
        "synthetic_initial_rgb",
        "synthetic_prompt",
        "auto_start",
        "preload_scenes",
        "wheel_profile",
        "wheel_profiles_dir",
        "wheel_device",
        "wheel_steering_axis",
        "wheel_throttle_axis",
        "wheel_brake_axis",
        "wheel_pedals_inverted",
        "no_wheel",
        "control_assets_dir",
        "official_hdmap_dir",
    }
)
_LOCAL_OUTPUT_FIELDS = frozenset(
    {
        "world_model_manifest_path",
        "no_hud",
        "stream_mjpeg",
        "cuda_visible_devices",
        "compute_device",
        "ludus_backend",
        "sync_gpu_timing",
        "profile_world_model",
        "offload_text_encoder",
        "postprocess_preset",
        "hf_org",
        "stop_after_chunks",
        "synthetic_model",
        "bev",
        "bev_resolution",
        "bev_height_m",
        "bev_fov_deg",
        "bev_tilt_deg",
        "oob_warn_proximity",
        "oob_respawn_proximity",
        "oob_respawn_debounce_chunks",
        "oob_margin_m",
        "oob_warning_zone_m",
    }
)


class OmnidreamsLaunchCapability:
    """Construct OmniDreams replay, WebRTC, and local-window launches."""

    def supported_modes(
        self,
        config: RunnerConfig,
        options: LaunchOptions,
    ) -> tuple[LaunchMode, ...]:
        modes: list[LaunchMode] = ["mp4", "null"]
        if _is_single_view(config):
            modes.append("webrtc")
        if _world_model_manifest(config, options) is not None:
            modes.append("local-window")
        return tuple(modes)

    def resolve(
        self,
        config: RunnerConfig,
        *,
        mode: LaunchMode,
        options: LaunchOptions,
    ) -> ResolvedLaunch | None:
        if mode in {"mp4", "null"}:
            _validate_fields("scenario", options.scenario, _REPLAY_SCENARIO_FIELDS)
            _validate_fields("output", options.output, _REPLAY_OUTPUT_FIELDS)
            output_path = options.output.get("path") or options.output.get("output")
            if mode == "mp4" and output_path is None:
                output_path = _DEFAULT_MP4_OUTPUT_PATH
            return _demo_launch(config, mode, options, output_path=output_path)
        if mode == "webrtc" and _is_single_view(config):
            _validate_fields("scenario", options.scenario, _WEBRTC_SCENARIO_FIELDS)
            _validate_fields("output", options.output, _WEBRTC_OUTPUT_FIELDS)
            return _demo_launch(config, mode, options)
        if mode == "local-window":
            world_manifest = _world_model_manifest(config, options)
            if world_manifest is None:
                return None
            _validate_fields("scenario", options.scenario, _LOCAL_SCENARIO_FIELDS)
            _validate_fields("output", options.output, _LOCAL_OUTPUT_FIELDS)
            return _local_window_launch(config, options, world_manifest)
        return None


def _demo_launch(
    config: RunnerConfig,
    mode: LaunchMode,
    options: LaunchOptions,
    *,
    output_path: object | None = None,
) -> ResolvedLaunch:
    summary: dict[str, object] = {
        "runner": config.runner_name,
        "mode": mode,
        "device": config.device,
    }
    if output_path is not None:
        summary["output_path"] = output_path
    stats_path = options.output.get("stats_path")
    if stats_path is not None:
        summary["stats_path"] = stats_path
    stats_dir = options.output.get("stats_dir")
    if stats_dir is not None:
        summary["stats_dir"] = stats_dir
    if mode == "webrtc":
        summary["host"] = options.host or options.output.get("host", "0.0.0.0")
        summary["port"] = (
            options.port
            if options.port is not None
            else options.output.get("port", 8082)
        )
    return ResolvedLaunch(
        mode=mode,
        label=f"OmniDreams {mode} launch",
        summary=summary,
        launch=partial(
            _launch_demo,
            config=config,
            mode=mode,
            options=options,
            output_path=output_path,
        ),
    )


def _launch_demo(
    *,
    config: RunnerConfig,
    mode: LaunchMode,
    options: LaunchOptions,
    output_path: object | None,
) -> object:
    from omnidreams.demo.app import launch_from_runner

    if mode not in {"mp4", "null", "webrtc"}:
        raise ValueError(f"Unsupported OmniDreams launch mode: {mode!r}.")
    output = dict(options.output)
    if output_path is not None:
        output.setdefault("path", output_path)
    return launch_from_runner(
        config=config,
        mode=cast(Literal["mp4", "null", "webrtc"], mode),
        scenario=dict(options.scenario),
        output=output,
        host=options.host,
        port=options.port,
        prefer_sw_encoder=options.prefer_sw_encoder,
    )


def _local_window_launch(
    config: RunnerConfig,
    options: LaunchOptions,
    world_manifest: Path,
) -> ResolvedLaunch:
    return ResolvedLaunch(
        mode="local-window",
        label="OmniDreams local interactive window",
        summary={
            "runner": config.runner_name,
            "mode": "local-window",
            "world_model_manifest": world_manifest,
        },
        notes=(
            (
                "The compatibility world-model manifest supplies interactive "
                "runtime and native-acceleration settings."
            ),
        ),
        launch=partial(
            _launch_local_window,
            config=config,
            options=options,
            world_manifest=world_manifest,
        ),
    )


def _launch_local_window(
    *,
    config: RunnerConfig,
    options: LaunchOptions,
    world_manifest: Path,
) -> object:
    from omnidreams.interactive_drive.demo import (
        launch_from_runner,
    )

    return launch_from_runner(
        config=config,
        world_model_manifest=world_manifest,
        scenario=dict(options.scenario),
        output=dict(options.output),
    )


def _world_model_manifest(
    config: RunnerConfig,
    options: LaunchOptions,
) -> Path | None:
    configured = options.output.get("world_model_manifest_path")
    if configured is not None:
        return Path(cast(Any, configured))
    if options.legacy_world_manifest is not None:
        return options.legacy_world_manifest
    bundled = _LOCAL_WINDOW_MANIFESTS.get(config.runner_name)
    return None if bundled is None else Path(bundled)


def _validate_fields(
    section: str,
    values: Mapping[str, object],
    allowed: set[str] | frozenset[str],
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported OmniDreams {section} fields: {', '.join(unknown)}."
        )


def _is_single_view(config: RunnerConfig) -> bool:
    diffusion_model = getattr(config.pipeline, "diffusion_model", None)
    transformer: Any = getattr(diffusion_model, "transformer", None)
    return int(getattr(transformer, "num_views", 1)) == 1


LAUNCH_CAPABILITY = OmnidreamsLaunchCapability()

__all__ = ["LAUNCH_CAPABILITY", "OmnidreamsLaunchCapability"]
