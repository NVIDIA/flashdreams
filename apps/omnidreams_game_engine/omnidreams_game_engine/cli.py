# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from loguru import logger
from omnidreams.hf_org import DEFAULT_HF_ORG, apply_cli_to_env
from omnidreams.hf_org import ENV_VAR as _HF_ORG_ENV_VAR

from flashdreams.infra.postprocess import VideoPostprocessChainConfig
from flashdreams.plugins.registry import discover_postprocess_presets
from flashdreams.serving.realtime.timing import TraceSink
from omnidreams_game_engine.app import InteractiveDriveApp
from omnidreams_game_engine.backends.base import RenderBackend
from omnidreams_game_engine.backends.raster import RasterRenderBackend
from omnidreams_game_engine.backends.world_model import WorldModelRenderBackend
from omnidreams_game_engine.cli_args import ExplicitArgTrackingArgumentParser
from omnidreams_game_engine.config import (
    AppConfig,
    BevConfig,
    RasterConfig,
    WorldModelProfileConfig,
)
from omnidreams_game_engine.log import configure_logging
from omnidreams_game_engine.world_model.manifest import (
    load_world_model_manifest,
    resolve_world_model_manifest_path,
)

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


def build_parser() -> argparse.ArgumentParser:
    parser = ExplicitArgTrackingArgumentParser(
        description="Single-process flashdreams driving demo"
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
            "for a richer browser viewer prefer the centralized "
            "``webrtc`` launch mode. Implies --no-hud "
            "when launched via the demo wrapper."
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
        action="store_true",
        help=(
            "Enable game-style actor and static-world collisions, along with "
            "the vehicle speed limit and collision visual flare. By default, "
            "collisions, the speed limit, and their visual effect are disabled."
        ),
    )
    parser.add_argument(
        "--disable-visual-flare",
        action="store_true",
        help=(
            "Disable the strong full-screen dark fade that signals a collision "
            "when --game-mode is enabled."
        ),
    )
    parser.add_argument(
        "--bev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render a synthetic top-down BEV map alongside the main camera and"
            " publish it on /bev_stream. The default is a straight-down,"
            " orthographic-style view of the HD-map plane. Disable to skip the"
            " extra rasterizer dispatch when running without the GTC HUD."
        ),
    )
    parser.add_argument(
        "--bev-resolution",
        default="1024x1024",
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
        default=BevConfig().height_m,
        help="BEV camera altitude in metres above the rig.",
    )
    parser.add_argument(
        "--bev-fov-deg",
        type=float,
        default=BevConfig().fov_deg,
        help="BEV camera vertical field-of-view in degrees.",
    )
    parser.add_argument(
        "--bev-tilt-deg",
        type=float,
        default=BevConfig().tilt_deg,
        help=(
            "Advanced override for the BEV camera pitch in degrees. The"
            " default ``0`` keeps the mini-map straight down; positive values"
            " re-enable the older perspective navigation view and should stay"
            " below ``bev-fov-deg / 2``."
        ),
    )
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

    bev_width, bev_height = _parse_resolution(args.bev_resolution)
    bev_config = BevConfig(
        enabled=bool(args.bev),
        width=bev_width,
        height=bev_height,
        height_m=float(args.bev_height_m),
        fov_deg=float(args.bev_fov_deg),
        tilt_deg=float(args.bev_tilt_deg),
    )
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
        raster=RasterConfig(
            compute_device=args.compute_device,
            sync_gpu_timing=args.sync_gpu_timing,
        ),
        world_model_profile=WorldModelProfileConfig(
            enabled=bool(args.profile_world_model),
        ),
        world_model_offload_text_encoder=bool(args.offload_text_encoder),
        postprocess=VideoPostprocessChainConfig(preset=args.postprocess_preset),
        bev=bev_config,
        game_mode=bool(args.game_mode),
        stream_mjpeg_bind=args.stream_mjpeg,
        stop_after_consumed_chunks=args.stop_after_chunks,
        visual_flare_enabled=False if args.disable_visual_flare else None,
    )

    backend: RenderBackend
    if config.backend == "raster":
        backend = RasterRenderBackend(
            chunk=config.chunk, raster=config.raster, bev=config.bev
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
        )
    return config, backend


def run(args: argparse.Namespace, trace_sink: TraceSink | None = None) -> None:
    """Execute the interactive-drive backend for ``args`` (single-scene ``--no-hud`` path).

    The HUD / streaming paths instead build one long-lived
    :class:`InteractiveDriveApp` and call ``load_scene`` / ``run_scene`` per scene.
    """
    configure_logging()
    config, backend = prepare_config_and_backend(args)
    app = InteractiveDriveApp(config=config, backend=backend, trace_sink=trace_sink)
    app.run()
