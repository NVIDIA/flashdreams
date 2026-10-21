# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The multiview checkpoint's binding to the reusable interactive-drive app.

Two hooks rather than the one OmniDreams uses. The backend is ours because the
model is, and the scene loader reads prepared JSON bundles rather than a USDZ; see
:mod:`cosmos3_nano_sim_auto.apps.interactive_drive.scene` for why that cannot be
otherwise.

The frame size is the checkpoint's rather than the app's usual 1280x704. It
decides both the conditioning the model is fed and, through the app's session,
the size a camera is presented at, so the two cannot be set apart.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import fields, replace
from functools import partial
from pathlib import Path

from interactive_drive import (
    InteractiveDriveApplication,
    InteractiveDriveApplicationDefaults,
    InteractiveDriveSceneOption,
)
from interactive_drive.config import AppConfig, PresentationLayout
from interactive_drive.core import hud_canvas
from interactive_drive.multiview import AuxiliaryView

from cosmos3_nano_sim_auto.apps.interactive_drive.backend import (
    MultiViewRenderBackend,
    MultiViewSettings,
)
from cosmos3_nano_sim_auto.apps.interactive_drive.scene import load_scene_bundle
from cosmos3_nano_sim_auto.apps.interactive_drive.ui import MultiviewDriveSession
from cosmos3_nano_sim_auto.config import DEFAULT_CHECKPOINT, DEFAULT_SCENE
from cosmos3_nano_sim_auto.impl.conditioning.camera_rig import CAMERAS
from cosmos3_nano_sim_auto.impl.distributed.parallel import init_parallel
from cosmos3_nano_sim_auto.inputs import VIEW_SIZES
from cosmos3_nano_sim_auto.scenes import SCENE_NAMES, prepare_scene
from flashdreams.core.visualization.lidar import LidarView
from flashdreams.infra.postprocess import VideoSpec
from flashdreams.runtime_v2.session_desc import SessionDesc

MODEL_WIDTH, MODEL_HEIGHT = VIEW_SIZES["480"]

DEFAULT_SAMPLE = DEFAULT_SCENE
"""The matching prompt and seven opening images in the prepared scene bundle."""

_VIEW_PRESETS = {
    1: CAMERAS[:1],
    4: tuple(CAMERAS[index] for index in (0, 1, 3, 5)),
    7: CAMERAS[:7],
}
"""Prepared-scene camera subsets; four views cover the front, right, rear and left."""


def build_backend(
    config: AppConfig, *, settings: MultiViewSettings
) -> MultiViewRenderBackend:
    """One backend for the scene the app has settled on."""
    return MultiViewRenderBackend(
        config.chunk,
        config.raster,
        settings=replace(
            settings,
            device=config.world_model_device,
            seed=settings.seed
            if config.world_model_seed is None
            else config.world_model_seed,
        ),
    )


def defaults(
    settings: MultiViewSettings | None = None,
) -> InteractiveDriveApplicationDefaults:
    """App defaults pointing at this checkpoint."""
    settings = settings or MultiViewSettings()
    return InteractiveDriveApplicationDefaults(
        title="Interactive Drive (cosmos3-nano-sim-auto)",
        slug="interactive-drive-cosmos3-nano-sim-auto",
        total_blocks=0,
        fps=30,
        width=MODEL_WIDTH,
        height=MODEL_HEIGHT,
        scene=DEFAULT_SCENE,
        sample=DEFAULT_SAMPLE,
        # Our map is built straight into ludus pools on the GPU and never comes
        # back as world-space layers, so there is nothing for the host to build
        # a collider from. The scene bundle says as much with empty layers.
        physics=False,
        reverse=True,
        # The overhead view is a camera the app's rasterizer appends to the rig
        # it renders, and this checkpoint's rig is ours. Nothing generates one,
        # so the panel would read the second camera of the drive instead.
        bev=False,
        # A view is 832x480, so a rig of one would draw the HUD at 0.65 and
        # leave whatever is displaying the stream to magnify the text back up.
        # Four views already tile past the HUD's own size and pay nothing here.
        full_size_hud=True,
        # The one thing the app needs to know about a mesh: which rank owns the
        # window, and which ranks it has to keep in step. The axes themselves
        # stay in the backend, where the checkpoint that defines them is.
        parallel_context=settings.parallel,
        backend_factory=partial(build_backend, settings=settings),
    )


class _LidarAppConfig(AppConfig):
    """Drive config reserving a sensor cell after the camera rig."""

    @property
    def presentation(self) -> PresentationLayout:
        if self.present_camera is not None:
            return super().presentation
        return PresentationLayout.grid(len(self.camera_names) + 1)


class CosmosInteractiveDriveApplication(InteractiveDriveApplication):
    """Add joint RGB/LiDAR options to the shared steering application."""

    session_type = MultiviewDriveSession

    def __init__(self, *, settings: MultiViewSettings) -> None:
        self._settings = settings
        super().__init__(defaults=defaults(settings), scene_loader=load_scene_bundle)

    def init(self, commandline_args: Sequence[str]) -> None:
        parser = argparse.ArgumentParser(
            prog="Interactive Drive model options", add_help=False
        )
        parser.add_argument("--no-ui", action="store_true", help=argparse.SUPPRESS)
        # Keep the shared RGB/HDMAP flag from being parsed as short for --views.
        parser.add_argument("--view", help=argparse.SUPPRESS)
        cameras = parser.add_mutually_exclusive_group()
        cameras.add_argument(
            "--views",
            type=int,
            choices=tuple(_VIEW_PRESETS),
            help="Camera preset: 1 front-wide (default), 4 front/right/rear/left, "
            "or all 7 prepared-scene cameras. Use instead of --camera.",
        )
        cameras.add_argument("--camera", help=argparse.SUPPRESS)
        parser.add_argument(
            "--scene",
            type=Path,
            default=DEFAULT_SCENE,
            help=f"Prepared scene directory or built-in name: {', '.join(SCENE_NAMES)}. "
            f"Default: {DEFAULT_SCENE}.",
        )
        parser.add_argument("--checkpoint", help="Joint or RGB model export")
        parser.add_argument(
            "--fps", type=float, help="Prepared scene frame rate (default: 30)"
        )
        parser.add_argument(
            "--lidar",
            action="store_true",
            help="Enable RGB+LiDAR using the model's matching HF V1.2 codec",
        )
        parser.add_argument(
            "--lidar-checkpoint",
            type=Path,
            help="Enable RGB+LiDAR with an explicit local codec instead of the HF default",
        )
        parser.add_argument(
            "--lidar-tokenizer-version",
            choices=("v1p1", "v1p2"),
            default=self._settings.lidar_tokenizer_version,
        )
        parser.add_argument(
            "--lidar-contiguous-linears",
            action=argparse.BooleanOptionalAction,
            default=self._settings.lidar_contiguous_linears,
            help="Opt in to contiguous FP32 codec projections (benchmark first)",
        )
        parser.add_argument(
            "--lidar-view",
            choices=[view.value for view in LidarView],
            default=self._settings.lidar_view.value,
        )
        parser.add_argument(
            "--optimized-lidar-renderer",
            action=argparse.BooleanOptionalAction,
            default=self._settings.optimized_lidar_renderer,
            help="Opt in to faster prepared-scene LiDAR controls (benchmark first)",
        )
        parser.add_argument(
            "--culled-lidar-renderer",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Range-cull LiDAR scans (default with LiDAR; --no-culled-lidar-renderer restores baseline)",
        )
        parser.add_argument(
            "--lidar-codec-tf32",
            action=argparse.BooleanOptionalAction,
            default=self._settings.lidar_codec_tf32,
            help="TF32 matmuls in the LiDAR codec only (changes outputs slightly)",
        )
        from cosmos3_nano_sim_auto.impl.rollout import COMPILE_DENOISE_MODES

        # The batch command's pipelining flags, meaning the same.
        parser.add_argument(
            "--parallel-decode",
            action="store_true",
            help="Each rank decodes the cameras it owns; rank 0 gathers them",
        )
        parser.add_argument(
            "--decoder-compile",
            action="store_true",
            help="Run the TAEHV decoder under torch.compile",
        )
        parser.add_argument(
            "--compile-denoise",
            choices=("off", *COMPILE_DENOISE_MODES),
            default=self._settings.compile_denoise or "off",
            help="torch.compile the noisy pass; RGB-only, and needs "
            "COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL=1 under torchrun",
        )
        parser.add_argument(
            "--compile-control-encode",
            action="store_true",
            help="Run the map's control encoder under torch.compile",
        )
        parser.add_argument(
            "--prefetch-controls",
            action="store_true",
            help="Draw and encode the map in a worker process. RGB-only, the "
            "next chunk is drawn while this one denoises, at one chunk (0.4 s) "
            "of extra steering latency",
        )
        parser.add_argument(
            "--prefetch-controls-gpu-offset",
            type=int,
            default=self._settings.prefetch_controls_gpu_offset,
            help="Put the worker on this rank's GPU plus N, modulo the node's; "
            "at least the per-node rank count keeps it off model GPUs",
        )
        args, shared = parser.parse_known_args(list(commandline_args))
        if args.views is not None:
            shared.extend(("--camera", ",".join(_VIEW_PRESETS[args.views])))
        elif args.camera is not None:
            shared.extend(("--camera", args.camera))
        if args.view is not None:
            shared.extend(("--view", args.view))
        if args.fps is not None:
            if not math.isfinite(args.fps) or args.fps <= 0:
                parser.error("--fps must be finite and positive")
            # The shared parser's default accepts a numeric value, while its
            # CLI converter accepts only integers. Recorded scenes may be 32.7 Hz.
            self._default_fps = args.fps
        if args.no_ui:
            shared.append("--no-ui")
        if "--help" in shared or "-h" in shared:
            parser.print_help()
        lidar_enabled = (
            args.lidar
            or args.lidar_checkpoint is not None
            or self._settings.lidar_checkpoint is not None
        )
        settings = replace(
            self._settings,
            checkpoint=args.checkpoint or self._settings.checkpoint,
            lidar_checkpoint=args.lidar_checkpoint or self._settings.lidar_checkpoint,
            lidar_tokenizer_version=args.lidar_tokenizer_version,
            lidar_contiguous_linears=args.lidar_contiguous_linears,
            lidar_view=LidarView(args.lidar_view),
            optimized_lidar_renderer=args.optimized_lidar_renderer,
            culled_lidar_renderer=(
                self._settings.culled_lidar_renderer and lidar_enabled
                if args.culled_lidar_renderer is None
                else args.culled_lidar_renderer
            ),
            lidar_codec_tf32=args.lidar_codec_tf32,
            compile_denoise=None
            if args.compile_denoise == "off"
            else args.compile_denoise,
            parallel_decode=args.parallel_decode or self._settings.parallel_decode,
            decoder_compile=args.decoder_compile or self._settings.decoder_compile,
            compile_control_encode=args.compile_control_encode
            or self._settings.compile_control_encode,
            prefetch_controls=args.prefetch_controls
            or self._settings.prefetch_controls,
            prefetch_controls_gpu_offset=args.prefetch_controls_gpu_offset,
        )
        if (
            settings.optimized_lidar_renderer
            or settings.culled_lidar_renderer
            or settings.lidar_codec_tf32
        ) and not lidar_enabled:
            parser.error("LiDAR renderer options require --lidar or --lidar-checkpoint")
        if settings.lidar_contiguous_linears and not lidar_enabled:
            parser.error(
                "--lidar-contiguous-linears requires --lidar or --lidar-checkpoint"
            )
        if settings.prefetch_controls_gpu_offset and not settings.prefetch_controls:
            parser.error("--prefetch-controls-gpu-offset requires --prefetch-controls")
        if lidar_enabled and "--help" not in shared and "-h" not in shared:
            from cosmos_interactive.checkpoint_load import resolve_checkpoint_source

            from cosmos3_nano_sim_auto.checkpoint import resolve_lidar_checkpoint
            from cosmos3_nano_sim_auto.impl.conditioning.world_scenario.prepared_lidar import (
                require_implementation,
            )

            require_implementation()
            source = resolve_checkpoint_source(
                settings.checkpoint, default=DEFAULT_CHECKPOINT
            )
            settings = replace(
                settings,
                checkpoint=source,
                lidar_checkpoint=resolve_lidar_checkpoint(
                    source,
                    checkpoint=settings.lidar_checkpoint,
                    version=settings.lidar_tokenizer_version,
                ),
            )
        self._settings = settings
        self._backend_factory = partial(build_backend, settings=settings)
        if "--help" not in shared and "-h" not in shared:
            if not args.scene.is_dir() and str(args.scene) in SCENE_NAMES:
                self._interactive_scene_options = tuple(
                    InteractiveDriveSceneOption(
                        path=prepare_scene(Path(name)),
                        label=name,
                        variants=("default",),
                    )
                    for name in SCENE_NAMES
                )
                scene = self._interactive_scene_options[
                    SCENE_NAMES.index(str(args.scene))
                ].path
            else:
                scene = args.scene
            self._default_sample = scene
            shared.extend(("--scene", str(scene)))
        super().init(shared)
        if settings.lidar_checkpoint is not None:
            assert self._config is not None
            app = _LidarAppConfig(
                **{
                    item.name: getattr(self._config.app, item.name)
                    for item in fields(AppConfig)
                    if item.init
                }
            )
            self._config = replace(self._config, app=app)
            output = app.postprocess.output_spec(
                VideoSpec(
                    width=app.raster.width, height=app.raster.height, fps=app.chunk.fps
                )
            )
            layout = app.presentation
            width, height = output.width * layout.columns, output.height * layout.rows
            # A worker's automatic UI suppression must not change mesh geometry.
            if self._full_size_hud and not args.no_ui:
                width, height = hud_canvas(width, height)
            self._desc = replace(self._desc, video_width=width, video_height=height)

    def create_session(self, session_desc: SessionDesc) -> MultiviewDriveSession:
        """Declare the sensors in the live backend's output order."""
        session = super().create_session(session_desc)
        assert isinstance(session, MultiviewDriveSession)
        if self._settings.lidar_checkpoint is not None:
            session.auxiliary_views = (AuxiliaryView("lidar", "LiDAR"),)
        return session


def create_app() -> CosmosInteractiveDriveApplication:
    """Create Interactive Drive driving the multiview checkpoint.

    The mesh is built here rather than in the backend because the app needs it
    before any backend exists: the runtime reads the session's mesh to decide
    which ranks it synchronizes and which one owns the window, and it asks
    before initialization. Off a launcher this is a mesh of one and nothing
    below behaves differently.
    """
    mesh = init_parallel()
    return CosmosInteractiveDriveApplication(settings=MultiViewSettings(parallel=mesh))


__all__ = [
    "build_backend",
    "create_app",
    "defaults",
]
