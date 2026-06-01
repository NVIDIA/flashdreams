# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from omnidreams.interactive_drive.backends.base import RenderBackend
from omnidreams.interactive_drive.config import AppConfig
from omnidreams.interactive_drive.input.keyboard import (
    KeyboardInputBackend,
    KeyboardState,
)
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
from omnidreams.interactive_drive.simulation.map_bounds import MapBounds
from omnidreams.interactive_drive.streaming_presenter import (
    MJPEGStreamingPresenter,
    parse_bind,
)
from omnidreams.interactive_drive.types import PresentedFrame, SceneBundle
from omnidreams.interactive_drive.video_model.chunk_pipeline import ChunkPipeline
from omnidreams.interactive_drive.video_model.local import LocalVideoModelAdapter


class InteractiveDriveApp:
    """Long-lived interactive-drive engine.

    The backend, video-model adapter and :class:`ChunkPipeline` are built
    once in :meth:`__init__`; the pipeline worker starts warming the
    scene-independent model immediately, overlapping the model load with
    any scene-selection wait. Each scene the user picks is bound via
    :meth:`load_scene` and driven by :meth:`run_scene`. The warmed model
    stays resident across scene changes, so switching scenes never re-pays
    the warmup/compile cost (only the per-scene geometry upload).
    """

    def __init__(
        self,
        config: AppConfig,
        backend: RenderBackend,
        presenter: PresenterBackend | None = None,
        *,
        close_presenter_on_exit: bool = True,
    ) -> None:
        """Construct the engine and begin model warmup.

        ``presenter`` lets the demo wrapper inject a HUD-aware presenter
        (e.g. :class:`SlangPyHudPresenter`) that needs constructor
        arguments outside :class:`AppConfig`'s vocabulary (scene-selector
        options, wheel device, control assets); the app rebinds it to its
        own long-lived keyboard. When ``None``, :func:`_build_presenter`
        returns either the default :class:`SlangPyPresenter` (a local
        Vulkan window) or, when ``config.stream_mjpeg_bind`` is set, an
        :class:`MJPEGStreamingPresenter` that serves frames over HTTP with
        no GPU-graphics dependency. Browser viewers with a richer frontend
        are served by ``omnidreams.webrtc.server`` instead.
        """
        self._config = config
        self._backend = backend
        self._keyboard = KeyboardState()
        if config.backend == "omnidreams":
            self._keyboard.set_view_mode("model_rgb")
        if presenter is None:
            self._presenter = _build_presenter(config, self._keyboard)
        else:
            self._presenter = presenter
            # Injected presenters (HUD / streaming) are constructed by the
            # demo with a placeholder keyboard; rebind to ours so input
            # lands on the engine's actual state object.
            bind_keyboard = getattr(self._presenter, "bind_keyboard", None)
            if callable(bind_keyboard):
                bind_keyboard(self._keyboard)
        # When ``False`` the caller (the demo's outer scene-change loop)
        # owns the presenter's lifecycle: it constructs one presenter at
        # startup, reuses it across many scenes, and only closes it when
        # the user actually closes the window. Default ``True`` matches the
        # bare ``--no-hud`` path where the app owns the presenter
        # end-to-end.
        self._close_presenter_on_exit = bool(close_presenter_on_exit)
        # Build the video-model pipeline once and start warming the
        # scene-independent model now, on the pipeline worker thread.
        # Scenes are bound later via load_scene; the model is never rebuilt
        # on a scene change.
        self._adapter = LocalVideoModelAdapter(backend)
        self._pipeline = ChunkPipeline(self._adapter)
        self._scene: SceneBundle | None = None
        self._map_bounds: MapBounds | None = None

    @property
    def presenter(self) -> PresenterBackend:
        return self._presenter

    @property
    def keyboard(self) -> KeyboardState:
        return self._keyboard

    @property
    def can_prewarm(self) -> bool:
        """Whether model warmup runs without a scene (drives the HUD text)."""
        return self._backend.can_prewarm

    def model_ready(self) -> bool:
        """``True`` once the scene-independent model warmup has completed."""
        return self._pipeline.model_ready.is_set()

    def load_scene(
        self, scene_path: object, variant: str, prompt_override: str | None
    ) -> None:
        """Load a scene bundle and bind it on the pipeline worker.

        Cheap relative to warmup: reads the USDZ and enqueues a
        ``request_scene`` that runs ``backend.load_scene`` (geometry
        upload + rollout restart) on the worker, FIFO behind any pending
        model warmup. The resident model is reused as-is.
        """
        self._scene = load_scene_bundle(
            scene_path=scene_path,
            camera_name=self._config.camera_name,
            variant=variant,
            prompt_override=prompt_override,
            raster=self._config.raster,
        )
        # Build the OOB map bounds at scene load -- the AABB is a property
        # of the scene's geometry and is invariant across the rollout
        # restarts inside run_scene.
        self._map_bounds = build_map_bounds(self._scene)
        self._pipeline.request_scene(self._scene)

    def run_scene(self) -> None:
        """Drive the current scene until the presenter closes or switches.

        Must be called after :meth:`load_scene`. Returns when
        ``run_main_loop`` reports the presenter wants to close -- which the
        slangpy HUD also uses to signal a scene change -- so the caller
        inspects ``presenter.pending_scene_change`` to tell the two apart.
        A manual reset / OOB respawn keeps the loop going with a fresh
        simulation and ``pipeline.reset`` (the warmed model is kept).
        """
        if self._scene is None or self._map_bounds is None:
            raise RuntimeError("load_scene() must be called before run_scene()")
        # Seed the loop's initial ``last_presented_frame`` with the scene's
        # first frame. The loop overlays a live loading status over it (see
        # ``_loading_status_message``) until the first generated chunk
        # arrives, and again briefly between rollouts during
        # ``pipeline.reset``.
        loading_frame = PresentedFrame(
            timestamp_us=0,
            rgb_host_uint8=self._scene.initial_rgb,
            depth_host_f32=None,
        )
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
                map_bounds=self._map_bounds,
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
                pipeline=self._pipeline,
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
                loading_status=self._loading_status_message,
            )
            if not reset_requested:
                break
            self._pipeline.reset()

    def _loading_status_message(self) -> str:
        """Phase text shown over the loading frame until the first chunk.

        World-model warmup takes priority; once the model is resident a
        scene (re)load only uploads geometry and renders the first chunk,
        so the lighter "Loading scene..." message is shown instead.
        """
        if not self.model_ready():
            return "Loading world model..."
        return "Loading scene..."

    def run(self) -> None:
        """Single-scene convenience: load the configured scene, run, tear down.

        Used by the bare ``--no-hud`` path, which never switches scenes.
        The scene-switching demo loops call ``load_scene`` / ``run_scene``
        / ``shutdown`` directly so the pipeline survives across scenes.
        """
        self.load_scene(
            self._config.scene_path,
            self._config.variant,
            self._config.prompt_override,
        )
        try:
            self.run_scene()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._pipeline.shutdown()
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
