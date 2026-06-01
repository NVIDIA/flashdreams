# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np
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
from omnidreams.interactive_drive.simulation.ground_snap import GroundSnapper
from omnidreams.interactive_drive.simulation.map_bounds import MapBounds
from omnidreams.interactive_drive.streaming_presenter import (
    MJPEGStreamingPresenter,
    parse_bind,
)
from omnidreams.interactive_drive.types import PresentedFrame, SceneBundle
from omnidreams.interactive_drive.video_model.chunk_pipeline import ChunkPipeline
from omnidreams.interactive_drive.video_model.local import LocalVideoModelAdapter

# Cadence for the event-pump loop that keeps the presenter alive while a
# scene parses on a background thread. ~60 Hz keeps input latency low and
# the loading indicator smooth without burning a core.
_SCENE_LOAD_PUMP_INTERVAL_S = 1.0 / 60.0


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
        # Ground snapper for the current scene. Built once per scene (its
        # spatial grid is invariant across rollouts) and reused, so a reset
        # doesn't rebuild it -- that pure-Python grid build can take seconds
        # on a dense mesh and would otherwise freeze the UI on every reset.
        self._ground_snapper: GroundSnapper | None = None
        # Lazily-built black frame at the render resolution, used as the
        # backdrop for the loading overlay while a scene parses (before the
        # scene's own initial frame is available).
        self._loading_base_rgb: np.ndarray | None = None
        # Optional parsed-scene cache, keyed by (path, variant, prompt).
        # Disabled unless --preload-scenes opts in via preload_scenes(); when
        # enabled, load_scene reuses parsed bundles so switching to a
        # preloaded (or previously-visited) scene skips the USDZ parse.
        self._cache_scenes = False
        self._scene_cache: dict[
            tuple[str, str, str | None],
            tuple[SceneBundle, MapBounds | None, GroundSnapper | None],
        ] = {}
        self._scene_cache_lock = threading.Lock()
        # Set while --preload-scenes is still parsing scenes in the
        # background; the presenter locks scene selection until it clears so
        # the user always gets the instant (cache-hit) switch.
        self._preload_started = False
        self._preload_done = threading.Event()

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
    ) -> bool:
        """Load a scene bundle off the UI thread and bind it on the worker.

        ``load_scene_bundle`` parses the USDZ (camera rig, HD-map parquet,
        ground mesh), which can take a second or more. Running it on a
        background thread keeps the presenter responsive -- the window
        manager won't flag the HUD as "not responding" -- and lets the
        loading indicator stay on screen throughout. Once parsed, the scene
        is bound on the pipeline worker (geometry upload + rollout restart,
        FIFO behind any pending model warmup); the resident model is reused
        as-is.

        Returns ``False`` if the presenter closed before the scene finished
        loading, in which case the caller must not call :meth:`run_scene`.
        """
        # Fast path: a preloaded / previously-parsed bundle is reused with no
        # parse (and no background pump). The worker still uploads geometry
        # and renders the first chunk, so the loop's "Loading scene..."
        # indicator still covers that part.
        cached = self._cached_scene(scene_path, variant, prompt_override)
        if cached is not None:
            self._scene, self._map_bounds, self._ground_snapper = cached
            self._pipeline.request_scene(self._scene)
            return True

        scene_ready = threading.Event()
        loaded: list[object] = []
        error: list[BaseException] = []

        def _parse() -> None:
            try:
                scene = load_scene_bundle(
                    scene_path=scene_path,
                    camera_name=self._config.camera_name,
                    variant=variant,
                    prompt_override=prompt_override,
                    raster=self._config.raster,
                )
                # The OOB AABB and the ground snapper's spatial grid are
                # properties of the scene geometry and invariant across the
                # rollout restarts inside run_scene -- build both here (on the
                # background thread) so a reset never rebuilds the snapper.
                loaded.append(scene)
                loaded.append(build_map_bounds(scene))
                loaded.append(build_ground_snapper(scene))
            except BaseException as exc:  # re-raised on the UI thread below
                error.append(exc)
            finally:
                scene_ready.set()

        threading.Thread(
            target=_parse, name="interactive_drive-scene-loader", daemon=True
        ).start()
        self._pump_presenter_until(scene_ready)
        if not scene_ready.is_set():
            return False  # presenter closed mid-load
        if error:
            raise error[0]
        self._scene, self._map_bounds, self._ground_snapper = (  # type: ignore[assignment]
            loaded[0],
            loaded[1],
            loaded[2],
        )
        self._store_scene(
            scene_path,
            variant,
            prompt_override,
            self._scene,
            self._map_bounds,
            self._ground_snapper,
        )
        self._pipeline.request_scene(self._scene)
        return True

    def preload_scenes(self, specs: Iterable[tuple[object, str, str | None]]) -> None:
        """Parse scene bundles in the background so later switches are instant.

        ``specs`` is an iterable of ``(scene_path, variant, prompt_override)``.
        Opt-in (the demo's ``--preload-scenes``); parsing runs sequentially
        on one daemon thread to bound peak CPU / memory, populating the cache
        that :meth:`load_scene` consults. Already-cached entries are skipped,
        and failures are logged and skipped so one bad scene doesn't abort the
        rest. This only skips the USDZ parse on switch -- the per-scene
        geometry upload and first-chunk generation still happen on the worker.
        While it runs, :meth:`preload_in_progress` returns True so the
        presenter can lock scene selection until every scene is ready.
        """
        self._cache_scenes = True
        self._preload_started = True
        self._preload_done.clear()
        pending = list(specs)

        def _worker() -> None:
            try:
                self._preload_worker(pending)
            finally:
                self._preload_done.set()

        threading.Thread(
            target=_worker, name="interactive_drive-scene-preloader", daemon=True
        ).start()

    def preload_in_progress(self) -> bool:
        """True while --preload-scenes is still parsing scenes in the background."""
        return self._preload_started and not self._preload_done.is_set()

    def _preload_worker(self, pending: list[tuple[object, str, str | None]]) -> None:
        for scene_path, variant, prompt_override in pending:
            key = self._scene_cache_key(scene_path, variant, prompt_override)
            with self._scene_cache_lock:
                if key in self._scene_cache:
                    continue
            try:
                scene = load_scene_bundle(
                    scene_path=scene_path,
                    camera_name=self._config.camera_name,
                    variant=variant,
                    prompt_override=prompt_override,
                    raster=self._config.raster,
                )
                bounds = build_map_bounds(scene)
                snapper = build_ground_snapper(scene)
            except BaseException as exc:  # noqa: BLE001 - log & skip one scene
                print(
                    f"[interactive-drive] scene preload failed for "
                    f"{Path(str(scene_path)).name} variant={variant!r}: {exc}",
                    flush=True,
                )
                continue
            with self._scene_cache_lock:
                self._scene_cache[key] = (scene, bounds, snapper)
            print(
                f"[interactive-drive] preloaded scene "
                f"{Path(str(scene_path)).name} variant={variant!r}",
                flush=True,
            )

    @staticmethod
    def _scene_cache_key(
        scene_path: object, variant: str, prompt_override: str | None
    ) -> tuple[str, str, str | None]:
        return (str(scene_path), variant, prompt_override)

    def _cached_scene(
        self, scene_path: object, variant: str, prompt_override: str | None
    ) -> tuple[SceneBundle, MapBounds | None, GroundSnapper | None] | None:
        if not self._cache_scenes:
            return None
        with self._scene_cache_lock:
            return self._scene_cache.get(
                self._scene_cache_key(scene_path, variant, prompt_override)
            )

    def _store_scene(
        self,
        scene_path: object,
        variant: str,
        prompt_override: str | None,
        scene: SceneBundle,
        map_bounds: MapBounds | None,
        ground_snapper: GroundSnapper | None,
    ) -> None:
        if not self._cache_scenes:
            return
        with self._scene_cache_lock:
            self._scene_cache[
                self._scene_cache_key(scene_path, variant, prompt_override)
            ] = (scene, map_bounds, ground_snapper)

    def _pump_presenter_until(self, done: threading.Event) -> None:
        """Pump events + a loading overlay until ``done`` is set or we close.

        Keeps the window responsive and the loading indicator live while a
        scene parses on the background thread, mirroring the loop's own
        loading-phase rendering. The overlay text follows
        :meth:`_loading_status_message` (model warmup takes priority).
        """
        frame = PresentedFrame(
            timestamp_us=0,
            rgb_host_uint8=self._loading_base_frame(),
            depth_host_f32=None,
        )
        view_mode = self._keyboard.view_mode
        while not done.is_set() and not self._presenter.should_close:
            self._presenter.process_events()
            self._presenter.present_frame(
                replace(frame, status_message=self._loading_status_message()),
                view_mode=view_mode,
            )
            time.sleep(_SCENE_LOAD_PUMP_INTERVAL_S)

    def _loading_base_frame(self) -> np.ndarray:
        if self._loading_base_rgb is None:
            width, height = self._config.raster.resolution_wh
            self._loading_base_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        return self._loading_base_rgb

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
        # First rollout is the scene load ("Loading scene..." / "Loading
        # world model..."); subsequent rollouts come from a manual reset or
        # OOB respawn, so switch the indicator to "Resetting..." for those.
        loading_status = self._loading_status_message
        while not self._presenter.should_close:
            simulation = EgoVehicleKinematics(
                initial_state=state_from_initial_pose(
                    initial_rig_to_world=self._scene.initial_rig_to_world,
                    initial_yaw_rad=self._scene.initial_yaw_rad,
                    # Always start at rest so the initial load matches a
                    # reset, rather than launching at the clip's recorded speed.
                    initial_speed_mps=0.0,
                ),
                vehicle_config=self._config.vehicle,
                ground_snapper=self._ground_snapper,
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
                loading_status=loading_status,
            )
            if not reset_requested:
                break
            self._pipeline.reset()
            loading_status = self._resetting_status_message
            # Paint the reset indicator at once, before the next rollout's
            # setup, so a reset shows on screen the instant it's requested
            # rather than after the rebuild completes.
            self._present_loading_once(loading_status)

    def _present_loading_once(self, loading_status: Callable[[], str]) -> None:
        """Render a single loading-overlay frame immediately (used on reset)."""
        if self._scene is None:
            return
        self._presenter.process_events()
        frame = PresentedFrame(
            timestamp_us=0,
            rgb_host_uint8=self._scene.initial_rgb,
            depth_host_f32=None,
        )
        self._presenter.present_frame(
            replace(frame, status_message=loading_status()),
            view_mode=self._keyboard.view_mode,
        )

    def _loading_status_message(self) -> str:
        """Phase text shown over the loading frame until the first chunk.

        World-model warmup takes priority; once the model is resident a
        scene (re)load only uploads geometry and renders the first chunk,
        so the lighter "Loading scene..." message is shown instead.
        """
        if not self.model_ready():
            return "Loading world model..."
        return "Loading scene..."

    def _resetting_status_message(self) -> str:
        """Phase text shown while a reset / respawn re-primes the rollout."""
        return "Resetting..."

    def run(self) -> None:
        """Single-scene convenience: load the configured scene, run, tear down.

        Used by the bare ``--no-hud`` path, which never switches scenes.
        The scene-switching demo loops call ``load_scene`` / ``run_scene``
        / ``shutdown`` directly so the pipeline survives across scenes.
        """
        try:
            if self.load_scene(
                self._config.scene_path,
                self._config.variant,
                self._config.prompt_override,
            ):
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
