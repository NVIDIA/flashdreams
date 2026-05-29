# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections.abc import Callable

from omnidreams.interactive_drive.backends.base import RenderBackend
from omnidreams.interactive_drive.config import AppConfig
from omnidreams.interactive_drive.input.keyboard import (
    KeyboardInputBackend,
    KeyboardState,
)
from omnidreams.interactive_drive.loading_overlay import render_loading_overlay
from omnidreams.interactive_drive.presenter import SlangPyPresenter
from omnidreams.interactive_drive.runtime.loop import (
    LoopConfig,
    PresenterBackend,
    run_main_loop,
)
from omnidreams.interactive_drive.scene_loader import load_scene_bundle
from omnidreams.interactive_drive.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
    build_ground_snapper,
    build_map_bounds,
    state_from_initial_pose,
)
from omnidreams.interactive_drive.streaming_presenter import (
    MJPEGStreamingPresenter,
    parse_bind,
)
from omnidreams.interactive_drive.types import PresentedFrame
from omnidreams.interactive_drive.video_model.chunk_pipeline import ChunkPipeline
from omnidreams.interactive_drive.video_model.local import LocalVideoModelAdapter

PresenterFactory = Callable[[AppConfig, KeyboardState], PresenterBackend]


class InteractiveDriveApp:
    def __init__(
        self,
        config: AppConfig,
        backend: RenderBackend,
        presenter_factory: PresenterFactory | None = None,
        *,
        close_presenter_on_exit: bool = True,
    ) -> None:
        """Construct the engine.

        ``presenter_factory`` lets the demo wrapper inject a HUD-aware
        presenter (e.g. :class:`SlangPyHudPresenter`) that needs
        constructor arguments outside :class:`AppConfig`'s vocabulary
        (scene-selector options, wheel device, control assets). When
        ``None``, :func:`_build_presenter` returns either the default
        :class:`SlangPyPresenter` (a local Vulkan window) or, when
        ``config.stream_mjpeg_bind`` is set, an
        :class:`MJPEGStreamingPresenter` that serves frames over HTTP
        with no GPU-graphics dependency. Browser viewers with a richer
        frontend are served by ``omnidreams.webrtc.server`` instead.
        """
        self._config = config
        self._backend = backend
        self._scene = load_scene_bundle(
            scene_path=config.scene_path,
            camera_name=config.camera_name,
            variant=config.variant,
            prompt_override=config.prompt_override,
            raster=config.raster,
        )
        self._keyboard = KeyboardState()
        if config.backend == "omnidreams":
            self._keyboard.set_view_mode("model_rgb")
        factory = presenter_factory or _build_presenter
        self._presenter = factory(config, self._keyboard)
        # When ``False`` the caller (the slangpy HUD's outer scene-change
        # loop) owns the presenter's lifecycle: it constructs one
        # presenter at startup, reuses it across many ``app.run()``
        # calls, and only closes it when the user actually closes the
        # window. Default ``True`` matches the bare ``--no-hud`` path
        # where each ``app.run()`` owns one presenter end-to-end.
        self._close_presenter_on_exit = bool(close_presenter_on_exit)

    def run(self) -> None:
        # Pre-rendered "Loading..." overlay. Used as the loop's initial
        # ``last_presented_frame`` so the user sees something while the
        # pipeline worker thread does ``backend.warmup`` (slow for the
        # world-model backend) and again briefly between rollouts during
        # ``pipeline.reset`` while the next chunk is being rendered.
        loading_frame = PresentedFrame(
            timestamp_us=0,
            rgb_host_uint8=render_loading_overlay(self._scene.initial_rgb),
            depth_host_f32=None,
        )
        local_backend = LocalVideoModelAdapter(self._backend)
        pipeline = ChunkPipeline(local_backend, self._scene)
        # Build the OOB map bounds once at scene load -- the AABB is a
        # property of the scene's geometry and is invariant across the
        # rollout-restart loop below.
        map_bounds = build_map_bounds(self._scene)
        try:
            while not self._presenter.should_close:
                simulation = EgoVehicleKinematics(
                    initial_state=state_from_initial_pose(
                        initial_rig_to_world=self._scene.initial_rig_to_world,
                        initial_yaw_rad=self._scene.initial_yaw_rad,
                        initial_speed_mps=(
                            0.0
                            if self._keyboard.command().manual_control
                            else self._scene.initial_speed_mps
                        ),
                    ),
                    vehicle_config=self._config.vehicle,
                    ground_snapper=build_ground_snapper(self._scene),
                    initial_timestamp_us=self._scene.initial_timestamp_us,
                    map_bounds=map_bounds,
                    oob_margin_m=self._config.oob_margin_m,
                    oob_warning_zone_m=self._config.oob_warning_zone_m,
                )
                input_backend = KeyboardInputBackend(self._keyboard)
                reset_requested = run_main_loop(
                    presenter=self._presenter,
                    runtime_controls=self._keyboard,
                    initial_presented_frame=loading_frame,
                    input_backend=input_backend,
                    simulation=simulation,
                    pipeline=pipeline,
                    config=LoopConfig(
                        initial_chunk_size=self._config.chunk.initial_chunk_frames,
                        chunk_size=self._config.chunk.chunk_frames,
                        frame_interval_s=self._config.chunk.frame_interval_s,
                        oob_warn_proximity=self._config.oob_warn_proximity,
                        oob_respawn_proximity=self._config.oob_respawn_proximity,
                        oob_respawn_debounce_chunks=(
                            self._config.oob_respawn_debounce_chunks
                        ),
                    ),
                )
                if not reset_requested:
                    break
                pipeline.reset()
        finally:
            pipeline.shutdown()
            self._backend.close()
            if self._close_presenter_on_exit:
                self._presenter.close()


def _build_presenter(config: AppConfig, keyboard: KeyboardState) -> PresenterBackend:
    """Default presenter factory.

    Returns an :class:`MJPEGStreamingPresenter` when
    ``config.stream_mjpeg_bind`` is set (a HOST:PORT bind address) --
    that path renders no window and has no graphics-GPU dependency, so
    it works on compute-only SKUs (e.g. GB300) where SlangPy can't
    create a Vulkan swapchain. Otherwise returns the default
    :class:`SlangPyPresenter` -- a local Vulkan window.

    For browser viewers with a richer frontend, ``omnidreams.webrtc.server``
    (a separate entry point) is the preferred path; this MJPEG fallback
    is the in-process, dependency-free alternative for headless boxes.
    """
    if config.stream_mjpeg_bind is not None:
        host, port = parse_bind(config.stream_mjpeg_bind)
        return MJPEGStreamingPresenter(
            raster=config.raster,
            keyboard=keyboard,
            bind_host=host,
            bind_port=port,
        )
    return SlangPyPresenter(raster=config.raster, keyboard=keyboard)
