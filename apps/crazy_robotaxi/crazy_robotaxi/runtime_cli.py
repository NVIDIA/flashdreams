# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from loguru import logger
from omnidreams.hf_org import DEFAULT_HF_ORG, apply_cli_to_env
from omnidreams.hf_org import ENV_VAR as _HF_ORG_ENV_VAR
from omnidreams_game_engine.app import InteractiveDriveApp
from omnidreams_game_engine.backends.base import RenderBackend
from omnidreams_game_engine.backends.raster import RasterRenderBackend
from omnidreams_game_engine.backends.world_model import WorldModelRenderBackend
from omnidreams_game_engine.cli_args import ExplicitArgTrackingArgumentParser
from omnidreams_game_engine.config import (
    AppConfig,
    WorldModelProfileConfig,
)
from omnidreams_game_engine.game_map import GAME_MAP_SUFFIX, load_game_map_header
from omnidreams_game_engine.log import configure_logging
from omnidreams_game_engine.renderer_settings import (
    RendererSettings,
    load_renderer_settings,
)
from omnidreams_game_engine.world_model.manifest import (
    load_world_model_manifest,
    resolve_world_model_manifest_path,
)

from flashdreams.infra.postprocess import VideoPostprocessChainConfig
from flashdreams.plugins.registry import discover_postprocess_presets
from flashdreams.serving.realtime.timing import TraceSink

# Package root (from this file's location) so packaged config paths resolve
# relative to the install, not the user's cwd.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_CONFIGS_ROOT = _PACKAGE_ROOT / "configs"


def resolve_manifest_path(path: str | Path) -> Path:
    """Resolve a CLI manifest value.

    Relative paths first mean "from the caller's cwd". Bare filenames and
    package-relative paths also fall back to the bundled interactive-drive
    config directory, so ``--manifest example_world_model_perf.yaml`` works
    from a workspace root.
    """
    return resolve_world_model_manifest_path(path)


def resolve_app_config_path(path: str | Path) -> Path:
    """Resolve an application config from the working or packaged directory."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    bundled = _CONFIGS_ROOT / candidate
    if bundled.is_file():
        return bundled.resolve()
    return candidate.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = ExplicitArgTrackingArgumentParser(
        description="Standalone Crazy Robotaxi game"
    )
    parser.add_argument(
        "--map",
        dest="scene",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a .robotaxi.yaml game map.",
    )
    parser.add_argument(
        "--force-map-recompile",
        action="store_true",
        help="Rebuild each selected map's compiled cache once in this process.",
    )
    # ``--backend`` exists primarily for the test suite, which exercises
    # the raster path (~30s warmup) instead of the full omnidreams pipeline
    # (~minutes of HF + compile). Suppress from --help so the user-facing
    # surface only ever shows the demo's actual production knobs; users
    # who really want raster can still pass it explicitly.
    parser.add_argument(
        "--backend",
        choices=("raster", "omnidreams"),
        default="raster",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--camera",
        default="camera_front_wide_120fov",
        help="Camera name, e.g. camera_front_wide_120fov or camera:front:wide:120fov",
    )
    parser.add_argument(
        "--variant",
        default="default",
        help="Visual variant defined by the game map.",
    )
    parser.add_argument("--prompt", default=None, help="Optional prompt override")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Omnidreams pipeline manifest (YAML). Accepts a path or a bundled "
            "config filename such as example_world_model_perf.yaml."
        ),
    )
    parser.add_argument(
        "--renderer-config",
        type=Path,
        default=_CONFIGS_ROOT / "default_renderer.yaml",
        help="Complete renderer YAML; defaults to the packaged game renderer.",
    )
    parser.add_argument(
        "--game-config",
        type=Path,
        default=_CONFIGS_ROOT / "default_game.yaml",
        help="Complete game-rules and taxi-physics YAML.",
    )
    parser.add_argument(
        "--synthetic-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the world-model manifest's synthetic_model setting. "
            "Synthetic mode uses local default-initialized model weights and "
            "synthetic embeddings while keeping the manifest's performance knobs."
        ),
    )
    parser.add_argument(
        "--official-hdmap-dir",
        type=Path,
        default=None,
        help="Optional directory containing official hdmap_00.png... frames used to override the first world-model chunk",
    )
    parser.add_argument(
        "--compute-device",
        choices=("cuda", "vulkan", "automatic"),
        default="cuda",
        help="SlangPy device used for raster compute; presenter still uses Vulkan for swapchain",
    )
    parser.add_argument(
        "--sync-gpu-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Submit each raster compute pass separately and wait for GPU idle to get per-pass timings",
    )
    parser.add_argument(
        "--profile-world-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable flashdreams pipeline CUDA-event profiling for the world-model runtime",
    )
    parser.add_argument(
        "--offload-text-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Precompute the flashdreams one-shot text/first-frame embeddings, "
            "free those encoders before the AR pipeline is built, and reuse "
            "the cached embeddings across world-model resets."
        ),
    )
    parser.add_argument(
        "--postprocess-preset",
        "--postprocess_preset",
        dest="postprocess_preset",
        default="",
        choices=sorted(discover_postprocess_presets()),
        help=(
            "Video post-process preset for generated frames. A configured "
            "preset starts enabled and can be toggled in the local HUD."
        ),
    )
    parser.add_argument(
        "--hf-org",
        default=None,
        metavar="ORG",
        help=(
            "Hugging Face org that hosts the omni-dreams model and sample"
            f" repos. Defaults to {DEFAULT_HF_ORG!r}."
            f" Equivalent to setting {_HF_ORG_ENV_VAR}; the flag wins when"
            " both are present. Stamped into the env var early in main()"
            " so every downstream HF lookup -- including URLs read from"
            " the world-model manifest yaml -- honours the chosen org."
        ),
    )
    parser.add_argument(
        "--stream-mjpeg",
        default=None,
        metavar="[HOST:]PORT",
        help=(
            "Instead of opening a Vulkan window, serve frames as an MJPEG "
            "HTTP stream on this bind address. Accepts ``HOST:PORT`` (e.g. "
            "``127.0.0.1:8080``), bare ``:PORT``, or a bare port number "
            "(``8080``); the bare forms bind on all interfaces. The user "
            "opens http://HOST:PORT/ in a browser to view the demo and "
            "send keyboard input. Useful on compute-only hosts (e.g. "
            "GB300-only DGX Station) where no Vulkan-capable GPU exists; "
            "for a richer browser viewer prefer the separate "
            "``omnidreams.webrtc.server`` entry point. Implies --no-hud "
            "when launched via the demo wrapper."
        ),
    )
    parser.add_argument(
        "--stream-token",
        default=None,
        metavar="TOKEN",
        help=(
            "Shared secret gating every MJPEG endpoint (/, /stream, /control,"
            " ...). Clients must send it as a ``?token=`` query parameter or"
            " an ``X-Stream-Token`` header; requests without a valid token"
            " get 403. The served page picks the token up from its own URL,"
            " so players just open http://HOST:PORT/?token=TOKEN. Omitted or"
            " empty keeps the historical open (unauthenticated) behavior."
        ),
    )
    parser.add_argument(
        "--stream-jpeg-quality",
        type=int,
        default=85,
        metavar="Q",
        help=(
            "JPEG quality for the MJPEG camera stream (1-100, default 85)."
            " Lower values cut per-frame bytes roughly linearly; useful on"
            " slow links (VPN / SSH tunnel) together with --stream-scale."
        ),
    )
    parser.add_argument(
        "--stream-scale",
        type=float,
        default=1.0,
        metavar="S",
        help=(
            "Downscale factor for the MJPEG camera stream in (0.1, 1.0];"
            " e.g. 0.5 sends 640x352 instead of 1280x704 (~4x fewer bytes)."
            " Applied before JPEG encode, off the render thread. Only the"
            " streamed pixels shrink; the model still renders full size."
        ),
    )
    parser.add_argument(
        "--stop-after-chunks",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Exit cleanly after consuming chunk index N from the present "
            "queue. Used by internal latency tracing; chunk 0 is warmup, so "
            "N yields N traced chunks (1..N)."
        ),
    )
    parser.add_argument(
        "--game-mode",
        choices=("taxi", "race"),
        default="taxi",
        help=("Select taxi fares or checkpoint racing (default: taxi)."),
    )
    parser.add_argument(
        "--disable-visual-flare",
        action="store_true",
        help=("Keep the collision visual flare disabled (the default)."),
    )
    parser.add_argument(
        "--bev",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Render a synthetic top-down BEV map alongside the main camera and"
            " publish it on /bev_stream. The default is a straight-down,"
            " orthographic-style view of the HD-map plane. Disable to skip the"
            " extra rasterizer dispatch when running without the GTC HUD."
        ),
    )
    parser.add_argument(
        "--bev-resolution",
        default=None,
        help=(
            "BEV render resolution as WIDTHxHEIGHT (default: 1024x1024). The"
            " HUD panel is roughly 470x400, so 1024 gives ~2x SSAA per axis"
            " and lets the LANCZOS panel resize cleanly bandlimit the"
            " result. Drop this if BEV encode + decode cost is hurting the"
            " main camera path; render quality scales with this number."
        ),
    )
    parser.add_argument(
        "--bev-height-m",
        type=float,
        default=None,
        help="BEV camera altitude in metres above the rig.",
    )
    parser.add_argument(
        "--bev-fov-deg",
        type=float,
        default=None,
        help="BEV camera vertical field-of-view in degrees.",
    )
    parser.add_argument(
        "--bev-tilt-deg",
        type=float,
        default=None,
        help=(
            "Advanced override for the BEV camera pitch in degrees. The"
            " default ``0`` keeps the mini-map straight down; positive values"
            " re-enable the older perspective navigation view and should stay"
            " below ``bev-fov-deg / 2``."
        ),
    )
    parser.add_argument(
        "--race-course",
        default=None,
        metavar="ID",
        help="Race course ID. Race mode uses the map's first course by default.",
    )
    parser.add_argument(
        "--race-times",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Race leaderboard CSV path. Defaults to "
            "$FLASHDREAMS_CACHE_DIR/interactive-drive/race_times.csv."
        ),
    )
    parser.add_argument(
        "--taxi-seed",
        type=int,
        default=None,
        help=(
            "Debug seed mixed with the scene ID to produce repeatable taxi fares. "
            "Omit it for a fresh random layout each game."
        ),
    )
    parser.add_argument(
        "--taxi-highscores",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Taxi leaderboard CSV path. Defaults to "
            "$FLASHDREAMS_CACHE_DIR/interactive-drive/highscores.csv."
        ),
    )
    parser.add_argument(
        "--taxi-alignment-diagnostics",
        type=Path,
        default=None,
        metavar="DIRECTORY",
        help=(
            "Capture frame-synchronized conditioning, generated RGB, BEV, "
            "PhysX geometry, and pose telemetry under a timestamped directory. "
            "Captures the standalone game session."
        ),
    )
    from crazy_robotaxi.live_edit.config import add_live_edit_args

    add_live_edit_args(parser)
    return parser


def _parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise SystemExit(f"--bev-resolution expected WIDTHxHEIGHT, got {value!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise SystemExit(
            f"--bev-resolution components must be integers: {value!r}"
        ) from exc
    if width <= 0 or height <= 0:
        raise SystemExit(f"--bev-resolution must be positive: {value!r}")
    return width, height


def renderer_settings_from_args(args: argparse.Namespace) -> RendererSettings:
    """Load renderer YAML and apply explicit visual CLI overrides."""
    cached = getattr(args, "_renderer_settings", None)
    if cached is not None:
        return cached
    path = resolve_app_config_path(
        getattr(args, "renderer_config", _CONFIGS_ROOT / "default_renderer.yaml")
    )
    settings = load_renderer_settings(path)
    bev = settings.bev
    if getattr(args, "bev", None) is not None:
        bev = replace(bev, enabled=bool(args.bev))
    if getattr(args, "bev_resolution", None) is not None:
        width, height = _parse_resolution(args.bev_resolution)
        bev = replace(bev, width=width, height=height)
    for arg_name, field_name in (
        ("bev_height_m", "height_m"),
        ("bev_fov_deg", "fov_deg"),
        ("bev_tilt_deg", "tilt_deg"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            bev = replace(bev, **{field_name: float(value)})
    settings = replace(settings, bev=bev)
    setattr(args, "_renderer_settings", settings)
    args.renderer_config = path
    # The HUD presenters still consume these resolved values from the argparse
    # namespace. Keep that compatibility surface concrete after YAML + CLI merge.
    args.bev = settings.bev.enabled
    args.bev_resolution = f"{settings.bev.width}x{settings.bev.height}"
    args.bev_height_m = settings.bev.height_m
    args.bev_fov_deg = settings.bev.fov_deg
    args.bev_tilt_deg = settings.bev.tilt_deg
    return settings


def main() -> None:
    """Stand-alone entry point for ``python -m omnidreams_game_engine.cli``.

    The console script and ``python -m omnidreams_game_engine`` go through
    :func:`omnidreams_game_engine.demo.main` (HUD wrapper behind
    ``--no-hud``); this stays so ``run`` can still be driven via the parser.
    """
    run(build_parser().parse_args())


def prepare_config_and_backend(
    args: argparse.Namespace,
) -> tuple[AppConfig, RenderBackend]:
    """Build the :class:`AppConfig` and :class:`RenderBackend` for ``args``.

    Split out of :func:`run` so the demo wrappers build the backend once and
    hand it to a long-lived :class:`InteractiveDriveApp` that switches scenes in
    place (keeping the warmed model resident).
    """
    # Stamp the resolved HF org before manifest and model artifact resolution.
    resolved_org = apply_cli_to_env(args.hf_org)
    if resolved_org != DEFAULT_HF_ORG:
        logger.info(
            f"[interactive-drive] using HF org '{resolved_org}' for omni-dreams repos",
        )

    scene_path = args.scene
    if scene_path is None:
        raise SystemExit("--map is required")
    if (
        args.game_mode == "race"
        and scene_path.exists()
        and scene_path.name.endswith(GAME_MAP_SUFFIX)
    ):
        header = load_game_map_header(scene_path)
        if not header.race_course_ids:
            raise SystemExit(f"Map {header.map_id!r} does not define any race_courses")
        if (
            args.race_course is not None
            and args.race_course not in header.race_course_ids
        ):
            available = ", ".join(header.race_course_ids)
            raise SystemExit(
                f"Unknown race course {args.race_course!r} for map {header.map_id!r}; "
                f"available: {available}"
            )

    renderer_settings = renderer_settings_from_args(args)
    bev_config = renderer_settings.bev
    manifest_path = (
        resolve_manifest_path(args.manifest) if args.manifest is not None else None
    )

    config = AppConfig(
        scene_path=scene_path,
        backend=args.backend,
        camera_name=args.camera,
        variant=args.variant,
        prompt_override=args.prompt,
        force_map_recompile=bool(args.force_map_recompile),
        manifest_path=manifest_path,
        raster=replace(
            renderer_settings.raster,
            compute_device=args.compute_device,
            sync_gpu_timing=args.sync_gpu_timing,
        ),
        world_model_profile=WorldModelProfileConfig(
            enabled=bool(args.profile_world_model),
        ),
        world_model_offload_text_encoder=bool(args.offload_text_encoder),
        postprocess=VideoPostprocessChainConfig(preset=args.postprocess_preset),
        bev=bev_config,
        game_mode=True,
        stream_mjpeg_bind=args.stream_mjpeg,
        stop_after_consumed_chunks=args.stop_after_chunks,
        visual_flare_enabled=(
            False
            if args.disable_visual_flare
            else renderer_settings.visual_flare_enabled
        ),
    )

    backend: RenderBackend
    if config.backend == "raster":
        backend = RasterRenderBackend(
            chunk=config.chunk,
            raster=config.raster,
            bev=config.bev,
            synchronize_bev_with_rgb=True,
        )
    else:
        if config.manifest_path is None:
            raise SystemExit("--manifest is required for the omnidreams backend")
        manifest = load_world_model_manifest(config.manifest_path)
        if args.synthetic_model is not None:
            manifest = replace(manifest, synthetic_model=bool(args.synthetic_model))
        if args.official_hdmap_dir is not None:
            manifest = replace(
                manifest, debug_condition_frame_dir=args.official_hdmap_dir.resolve()
            )
        if config.raster.resolution_wh != manifest.resolution_wh:
            config = replace(
                config,
                raster=replace(
                    config.raster,
                    width=manifest.resolution_wh[0],
                    height=manifest.resolution_wh[1],
                ),
            )
        backend = WorldModelRenderBackend(
            manifest=manifest,
            chunk=config.chunk,
            raster=config.raster,
            profile=config.world_model_profile,
            bev=config.bev,
            offload_text_encoder=config.world_model_offload_text_encoder,
            postprocess=config.postprocess,
            synchronize_bev_with_rgb=True,
            motion_conformance_diagnostics_enabled=(
                args.taxi_alignment_diagnostics is not None
            ),
        )
    return config, backend


def run(args: argparse.Namespace, trace_sink: TraceSink | None = None) -> None:
    """Execute the interactive-drive backend for ``args`` (single-scene ``--no-hud`` path).

    The HUD / streaming paths instead build one long-lived
    :class:`InteractiveDriveApp` and call ``load_scene`` / ``run_scene`` per scene.
    """
    configure_logging()
    config, backend = prepare_config_and_backend(args)
    from crazy_robotaxi.app import CrazyRobotaxiApp, taxi_config_from_args
    from crazy_robotaxi.live_edit.config import live_edit_config_from_args

    app = CrazyRobotaxiApp(
        config=config,
        taxi_config=taxi_config_from_args(args),
        backend=backend,
        game_mode=args.game_mode,
        race_course_id=args.race_course,
        race_times_path=(
            None if args.race_times is None else args.race_times.expanduser()
        ),
        alignment_diagnostics_root=args.taxi_alignment_diagnostics,
        trace_sink=trace_sink,
        live_edit_config=live_edit_config_from_args(args),
    )
    app.run()
