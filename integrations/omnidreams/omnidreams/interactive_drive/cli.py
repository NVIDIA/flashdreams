# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from omnidreams.hf_org import DEFAULT_HF_ORG, apply_cli_to_env
from omnidreams.hf_org import ENV_VAR as _HF_ORG_ENV_VAR
from omnidreams.interactive_drive.app import InteractiveDriveApp, PresenterFactory
from omnidreams.interactive_drive.backends.base import RenderBackend
from omnidreams.interactive_drive.backends.raster import RasterRenderBackend
from omnidreams.interactive_drive.backends.world_model import WorldModelRenderBackend
from omnidreams.interactive_drive.config import (
    AppConfig,
    BevConfig,
    RasterConfig,
    WorldModelProfileConfig,
)
from omnidreams.interactive_drive.synthetic_scene import build_synthetic_scene_to_temp
from omnidreams.interactive_drive.world_model.manifest import load_world_model_manifest
from omnidreams.scenes import local_scene_archive_path

# Package root, derived from this file's location so all packaged-asset
# defaults below resolve correctly regardless of the user's cwd. Layout:
#   ``omnidreams/interactive_drive/cli.py`` (this file)
#   ``omnidreams/interactive_drive/configs/{example_world_model.yaml,wheels/}``
# Users invoke ``uv run --package flashdreams-omnidreams interactive-drive``
# from the workspace root; the bundled configs are reached relative to
# the installed package, not the cwd. Scene USDZs no longer live under
# the package -- they're staged into ``$FLASHDREAMS_CACHE_DIR/
# omnidreams-scenes/`` (shared with the webrtc server).
_PACKAGE_ROOT = Path(__file__).resolve().parent

# UUID of the scene staged by ``omnidreams-prepare`` when no
# ``--scene-uuid`` is specified and used as the demo's ``--scene``
# default.
DEFAULT_SCENE_UUID = "01d503d4-449b-46fc-8d78-9085e70d3554"

# Default scene path: shared cache dir under ``$FLASHDREAMS_CACHE_DIR/
# omnidreams-scenes/clipgt-<uuid>.usdz``. The desktop demo and the
# webrtc server both cache under this root, so the HF download is
# shared via huggingface_hub's content-addressed cache and a
# pre-existing staged scene from one demo is visible to the other.
DEFAULT_SCENE = local_scene_archive_path(DEFAULT_SCENE_UUID)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-process flashdreams driving demo"
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help=(
            "Path to the input USDZ scene. Defaults to the scene staged by "
            f"prepare.py at {DEFAULT_SCENE}; any UUID from "
            "nvidia/omni-dreams-scenes/scenes/ works once staged."
        ),
    )
    parser.add_argument(
        "--synthetic-scene",
        action="store_true",
        help=(
            "Skip the USDZ download / staging and build a procedural,"
            " HD-map-data-free scene at startup instead. Useful for"
            " demos in territories where the real-world scenes can't be"
            " distributed. The generated scene is a wavy 2-lane road"
            " with a single intersection; pair with --synthetic-initial-rgb"
            " to supply a natural-looking starting camera frame."
        ),
    )
    parser.add_argument(
        "--synthetic-initial-rgb",
        type=Path,
        default=None,
        help=(
            "Path to a JPG / PNG used as the initial camera frame when"
            " --synthetic-scene is set. The world model is trained on"
            " natural driving frames, so a real photo (any forward-facing"
            " roadway) gives noticeably better generation than the"
            " default debug gradient. Resized to the raster resolution"
            " automatically."
        ),
    )
    parser.add_argument(
        "--synthetic-prompt",
        default=None,
        help=(
            "Optional text prompt embedded in the synthetic scene."
            " Mutually overridable by --prompt at run time. When omitted,"
            " the synthetic-scene builder uses a generic forward-driving"
            " caption."
        ),
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
        help="Prompt / first-image variant (default, 1, 2, 3)",
    )
    parser.add_argument("--prompt", default=None, help="Optional prompt override")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Omnidreams pipeline manifest (YAML) overriding the bundled default",
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
        "--hf-org",
        default=None,
        metavar="ORG",
        help=(
            "Hugging Face org that hosts the omni-dreams repos (models /"
            f" samples / scenes). Defaults to {DEFAULT_HF_ORG!r}."
            f" Equivalent to setting {_HF_ORG_ENV_VAR}; the flag wins when"
            " both are present. Stamped into the env var early in main()"
            " so every downstream HF lookup -- including URLs read from"
            " the world-model manifest yaml -- honours the chosen org."
        ),
    )
    parser.add_argument(
        "--bev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render a synthetic top-down BEV map alongside the main camera and"
            " publish it on /bev_stream. Mirrors AlpaSim's BEV camera (a"
            " pinhole projection looking straight down). Disable to skip the"
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
            "Forward pitch of the BEV camera in degrees. ``0`` is pure"
            " top-down; positive values tilt forward for a Google-Maps"
            " navigation-mode look. Should stay below ``bev-fov-deg / 2``"
            " so the bottom of the image doesn't cross the horizon."
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
    """Stand-alone entry point for ``python -m omnidreams.interactive_drive.cli``.

    The packaged ``interactive-drive`` console script and the
    ``python -m omnidreams.interactive_drive`` invocation both go through
    :func:`omnidreams.interactive_drive.demo.main` so the HUD wrapper can wrap
    this same backend behind ``--no-hud``. This function stays in
    place so callers that want to import :func:`run` can still
    exercise the parser via ``main()`` directly.
    """
    run(build_parser().parse_args())


def prepare_config_and_backend(
    args: argparse.Namespace,
) -> tuple[AppConfig, RenderBackend]:
    """Build the :class:`AppConfig` and :class:`RenderBackend` for ``args``.

    Split out of :func:`run` so the slangpy HUD path in
    :mod:`omnidreams.interactive_drive.demo` can call this in a loop -- once per
    scene change -- while keeping the same window / presenter alive
    across runs. The HUD's outer loop tears down the old backend with
    ``backend.close()``, calls this to build a fresh one for the
    newly-selected scene, then constructs a new
    :class:`InteractiveDriveApp` over the same presenter.
    """
    # Stamp the resolved HF org into the env var BEFORE we touch anything
    # that fetches (manifest loader, scene staging, world-model build).
    # All downstream omni-dreams URL composition reads this env var
    # lazily, so this single call is the only place the CLI flag plumbs
    # through to runtime fetches.
    resolved_org = apply_cli_to_env(args.hf_org)
    if resolved_org != DEFAULT_HF_ORG:
        print(
            f"[interactive-drive] using HF org '{resolved_org}' for omni-dreams repos",
            flush=True,
        )

    scene_path = args.scene
    if args.synthetic_scene:
        # Materialise a procedural USDZ to a temp dir for this process.
        # The scene loader treats it like any other USDZ; downstream code
        # paths (rasterizer, world model, presenter) need no changes.
        scene_path = build_synthetic_scene_to_temp(
            initial_rgb_path=args.synthetic_initial_rgb,
            prompt=args.synthetic_prompt,
        )
        print(
            f"[interactive-drive] synthetic scene materialised at {scene_path}",
            flush=True,
        )
    elif args.synthetic_initial_rgb is not None or args.synthetic_prompt is not None:
        raise SystemExit(
            "--synthetic-initial-rgb / --synthetic-prompt require --synthetic-scene"
        )

    bev_width, bev_height = _parse_resolution(args.bev_resolution)
    bev_config = BevConfig(
        enabled=bool(args.bev),
        width=bev_width,
        height=bev_height,
        height_m=float(args.bev_height_m),
        fov_deg=float(args.bev_fov_deg),
        tilt_deg=float(args.bev_tilt_deg),
    )

    config = AppConfig(
        scene_path=scene_path,
        backend=args.backend,
        camera_name=args.camera,
        variant=args.variant,
        prompt_override=args.prompt,
        manifest_path=args.manifest,
        raster=RasterConfig(
            compute_device=args.compute_device,
            sync_gpu_timing=args.sync_gpu_timing,
        ),
        world_model_profile=WorldModelProfileConfig(
            enabled=bool(args.profile_world_model),
        ),
        world_model_offload_text_encoder=bool(args.offload_text_encoder),
        bev=bev_config,
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
        if args.official_hdmap_dir is not None:
            manifest = replace(
                manifest, debug_condition_frame_dir=args.official_hdmap_dir.resolve()
            )
        backend = WorldModelRenderBackend(
            manifest=manifest,
            chunk=config.chunk,
            raster=config.raster,
            profile=config.world_model_profile,
            bev=config.bev,
            offload_text_encoder=config.world_model_offload_text_encoder,
        )
    return config, backend


def run(
    args: argparse.Namespace, *, presenter_factory: PresenterFactory | None = None
) -> None:
    """Execute the interactive-drive backend with the given parsed args.

    Convenience wrapper used by the ``--no-hud`` path that doesn't need
    to switch scenes mid-run. The slangpy HUD path drives
    :func:`prepare_config_and_backend` directly so it can rebuild the
    backend per scene click without recreating the presenter.
    """
    config, backend = prepare_config_and_backend(args)
    app = InteractiveDriveApp(
        config=config, backend=backend, presenter_factory=presenter_factory
    )
    app.run()
