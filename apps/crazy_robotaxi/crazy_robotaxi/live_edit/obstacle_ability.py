# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Obstacle events: a real scene vehicle track cloned into the lane ahead.

Ports the template-clone machinery of
``integrations/omnidreams/omnidreams/webrtc/actors.py`` /
``scripts/probe_moving_clone.py`` onto the game engine's
``DynamicActorTrajectory`` seam (``CrazyRobotaxiRuntime.advance_frames``
appends actors exactly like the pickup passengers; the rasterizer rebuilds
the obstacle cube pool from them through the same
``build_hdmap_object_pool`` path scene traffic uses, so a cloned track is
bit-compatible with scene-native tracks: same colors, flags, and per-frame
statistics).

Why clones: user-synthesized boxes render correctly in the conditioning but
never materialize (mask-verified 2026-08-10); clones of real perception
tracks do, and MOVING clones materialize solidly (probe_moving_clone.py,
2026-08-13) — a parked clone fights the history that shows its spot empty
and renders at ghost strength (~40%). The spawn key therefore clones a
moving vehicle track, retimes it to "now", and rigidly shifts it (XY only,
plus a ground-height correction) so it starts ahead of the ego and replays
its recorded motion there. With ``count`` > 1 one key press spawns a
"traffic" burst: distinct crossing/oncoming templates staggered in distance
(``spacing_m``) and time (``stagger_chunks``), each despawning after its
own pass.

Optional box-axis guidance (``guide_scale`` > 0, probe operating point 3.0)
extrapolates the flow along the with-box/without-box conditioning direction:
:class:`ObstacleGuidance` keeps a shadow encoder cache fed with obstacle-free
conditioning every chunk (identical frames while no event is active, an
extra no-box raster during events) and doubles the ``predict_flow`` call
while an event is on screen.

CUDA-graph compatibility (2026-08-21): the guidance is graph-safe. The
transformer's graph wrapper stages every top-level tensor kwarg — including
the ``hdmap_condition`` input — into static buffers on each call, so the
two forwards of a guided step are two REPLAYS of the same captured cond
graph with different conditioning staged in (the exact mechanism the
two-prompt text-edit guidance already rides). The ``predict_flow`` dispatch
itself runs eagerly outside any graph. The one graph seam that must be
bypassed is the ENCODER: the Wan VAE encoder's graph wrapper passes its
streaming cache dict through verbatim, so captured kernels are bound to one
cache's buffer addresses — replaying it against the shadow cache would
silently read/write the real cache. Shadow encodes therefore run eagerly
(:func:`_eager_vae_scope`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger
from omnidreams_game_engine.types import (
    DynamicActorTrajectory,
    SceneBundle,
    TrajectoryChunk,
    VehicleState,
)

from crazy_robotaxi.live_edit.config import LiveEditObstacleConfig

OBSTACLE_ENTITY_PREFIX = "live-edit-obstacle"
"""Entity-id prefix identifying obstacle clones (guidance strips by it)."""

_VEHICLE_TYPES = frozenset({"Vehicle", "Car", "Truck", "Bus"})
"""Actor object types eligible as obstacle templates."""


@dataclass(frozen=True)
class ObstacleTemplate:
    """A moving vehicle track extracted from the scene for cloning."""

    object_type: str
    timestamps_us: npt.NDArray[np.int64]
    centers_world: npt.NDArray[np.float32]
    orientations_xyzw: npt.NDArray[np.float32]
    dimensions_lwh: npt.NDArray[np.float32]
    drift_m: float
    """Ground-plane distance between the first and last sample."""

    duration_s: float
    """Track coverage in seconds."""


@dataclass(frozen=True)
class ObstacleEvent:
    """One active clone: a retimed, shifted copy of a template track."""

    entity_id: str
    object_type: str
    timestamps_us: npt.NDArray[np.int64]
    translations_world: npt.NDArray[np.float32]
    orientations_xyzw: npt.NDArray[np.float32]
    dimensions_lwh: npt.NDArray[np.float32]

    def actor(self) -> DynamicActorTrajectory:
        """Return the renderer track for this event (full retimed span)."""
        return DynamicActorTrajectory(
            entity_id=self.entity_id,
            object_type=self.object_type,
            timestamps_us=self.timestamps_us,
            translations_world=self.translations_world,
            orientations_xyzw=self.orientations_xyzw,
            dimensions_lwh=self.dimensions_lwh,
            is_simulated=True,
        )

    def center_at(self, timestamp_us: int) -> npt.NDArray[np.float32] | None:
        """Interpolated world center, or ``None`` outside the track span."""
        ts = self.timestamps_us
        if timestamp_us < int(ts[0]) or timestamp_us > int(ts[-1]):
            return None
        t = float(timestamp_us)
        return np.array(
            [np.interp(t, ts, self.translations_world[:, axis]) for axis in range(3)],
            dtype=np.float32,
        )


def extract_moving_templates(
    tracks: tuple[Any, ...],
    config: LiveEditObstacleConfig,
) -> tuple[ObstacleTemplate, ...]:
    """Extract car-sized moving tracks from ``SceneBundle.vehicle_bbox_tracks``.

    Filters mirror ``probe_moving_clone._extract_moving_templates``: enough
    samples/coverage, car-sized bbox length, ground-plane drift at least
    ``config.min_drift_m``. Slow movers (2-8 m/s average) sort first,
    longest coverage first within each group: the taxi catches a slow
    lead clone within a couple of seconds, bringing it into the model's
    materialization range (~15-25 m), while faster clones hold a 30+ m
    gap and never render.
    """
    templates: list[ObstacleTemplate] = []
    lo, hi = config.length_range_m
    for track in tracks:
        if getattr(track, "object_type", None) not in _VEHICLE_TYPES:
            continue
        timestamps = np.asarray(track.timestamps_us, dtype=np.int64)
        if len(timestamps) < 8:
            continue
        duration_s = float(timestamps[-1] - timestamps[0]) * 1e-6
        if duration_s < config.min_coverage_s:
            continue
        dimensions = np.asarray(track.dimensions_lwh, dtype=np.float32)
        if dimensions.ndim == 2:
            # Perception tracks carry per-sample dims; renderer tracks
            # carry one box per track (the game uses the first sample too).
            dimensions = dimensions[0]
        length = float(dimensions.max())
        if not lo <= length <= hi:
            continue
        centers = np.asarray(track.centers_world, dtype=np.float32)
        drift = float(np.linalg.norm(centers[-1, :2] - centers[0, :2]))
        if drift < config.min_drift_m:
            continue
        templates.append(
            ObstacleTemplate(
                object_type=str(track.object_type),
                timestamps_us=timestamps,
                centers_world=centers,
                orientations_xyzw=np.asarray(track.orientations_xyzw, dtype=np.float32),
                dimensions_lwh=dimensions,
                drift_m=drift,
                duration_s=duration_s,
            )
        )

    def order_key(template: ObstacleTemplate) -> tuple[int, float]:
        speed = template.drift_m / template.duration_s
        return (0 if 2.0 <= speed <= 8.0 else 1, -template.duration_s)

    templates.sort(key=order_key)
    return tuple(templates)


def local_ground_z(
    vertices: npt.NDArray[np.floating] | None,
    xy: npt.NDArray[np.floating],
    radius_m: float = 3.0,
) -> float | None:
    """Median z of ground-mesh vertices within ``radius_m`` of ``xy``.

    The validated bring-up recipe: rig-derived heights float boxes ~1.6 m
    above the road, the ground mesh grounds them.
    """
    if vertices is None:
        return None
    near = np.linalg.norm(vertices[:, :2] - np.asarray(xy)[None, :], axis=1) < radius_m
    if not near.any():
        return None
    return float(np.median(vertices[near, 2]))


def build_obstacle_event(
    template: ObstacleTemplate,
    *,
    ego_state: VehicleState,
    spawn_timestamp_us: int,
    config: LiveEditObstacleConfig,
    ground_vertices: npt.NDArray[np.floating] | None = None,
    entity_id: str = f"{OBSTACLE_ENTITY_PREFIX}-0",
    ahead_m: float | None = None,
    lateral_m: float | None = None,
) -> ObstacleEvent:
    """Retime a template to ``spawn_timestamp_us`` and shift it ahead of ego.

    Rigid XY translation only (orientation and per-frame jitter preserved —
    the statistics the model keys on); z is corrected by the ground-height
    delta between the source and target locations when the ground mesh
    covers both. ``ahead_m``/``lateral_m`` override the config placement
    (per-clone slots of a traffic burst).
    """
    if ahead_m is None:
        ahead_m = config.spawn_ahead_m
    if lateral_m is None:
        lateral_m = config.lateral_m
    forward = np.array(
        [np.cos(ego_state.yaw_rad), np.sin(ego_state.yaw_rad)], dtype=np.float64
    )
    left = np.array([-forward[1], forward[0]])
    ego_xy = np.array([ego_state.x_m, ego_state.y_m], dtype=np.float64)
    target_xy = ego_xy + ahead_m * forward + lateral_m * left

    centers = template.centers_world.copy()
    source_xy = centers[0, :2].astype(np.float64)
    shift_xy = target_xy - source_xy
    centers[:, 0] += np.float32(shift_xy[0])
    centers[:, 1] += np.float32(shift_xy[1])
    source_z = local_ground_z(ground_vertices, source_xy)
    target_z = local_ground_z(ground_vertices, target_xy)
    if source_z is not None and target_z is not None:
        centers[:, 2] += np.float32(target_z - source_z)

    timestamps = (
        template.timestamps_us
        - template.timestamps_us[0]
        + np.int64(spawn_timestamp_us)
    )
    return ObstacleEvent(
        entity_id=entity_id,
        object_type=template.object_type,
        timestamps_us=timestamps,
        translations_world=centers,
        orientations_xyzw=template.orientations_xyzw.copy(),
        dimensions_lwh=template.dimensions_lwh.copy(),
    )


@dataclass
class _ActiveClone:
    """Per-clone lifecycle state (each clone despawns independently)."""

    event: ObstacleEvent
    template_index: int
    chunks: int = 0
    hit_logged: bool = False


class ObstacleAbility:
    """Spawn, advance, and despawn obstacle events for one rollout.

    One spawn request queues ``config.count`` clones (a "traffic" burst for
    count > 1): distinct crossing/oncoming templates, staggered ahead of the
    ego by ``spacing_m`` per slot and ``stagger_chunks`` chunks apart in
    time. Each clone despawns on its own after its pass.
    """

    def __init__(
        self,
        templates: tuple[ObstacleTemplate, ...],
        config: LiveEditObstacleConfig,
        *,
        ground_vertices: npt.NDArray[np.floating] | None = None,
    ) -> None:
        self._templates = templates
        self._config = config
        self._ground_vertices = ground_vertices
        self._pending: list[tuple[int, int]] = []  # (slot, due chunk index)
        self._clones: list[_ActiveClone] = []
        self._chunk_index = 0
        self._event_count = 0
        self._burst_count = 0
        self._hit_count = 0

    @classmethod
    def from_scene(
        cls, scene: SceneBundle, config: LiveEditObstacleConfig
    ) -> ObstacleAbility:
        """Extract templates from the scene bundle's perception tracks."""
        templates = extract_moving_templates(scene.vehicle_bbox_tracks, config)
        logger.info(
            f"[live-edit] obstacle templates: {len(templates)} moving "
            f"vehicle tracks (of {len(scene.vehicle_bbox_tracks)})"
        )
        return cls(templates, config, ground_vertices=scene.ground_mesh_vertices)

    @property
    def active(self) -> bool:
        """Return whether any obstacle clone is currently on screen."""
        return bool(self._clones)

    @property
    def event(self) -> ObstacleEvent | None:
        """Return the oldest active event (single-clone compatibility)."""
        return self._clones[0].event if self._clones else None

    @property
    def events(self) -> tuple[ObstacleEvent, ...]:
        """Return all active events (presenter annotation hook)."""
        return tuple(clone.event for clone in self._clones)

    @property
    def hit_count(self) -> int:
        """Return how many events the ego has collided with."""
        return self._hit_count

    def request_spawn(self) -> None:
        """Queue a burst of ``config.count`` clones for the next chunks."""
        if not self._templates:
            logger.warning("[live-edit] obstacle spawn requested but no templates")
            return
        if self._pending or self._clones:
            return  # one burst at a time (matches the single-clone behavior)
        base = self._chunk_index
        self._pending = [
            (slot, base + slot * self._config.stagger_chunks)
            for slot in range(self._config.count)
        ]
        self._burst_count += 1

    def reset(self) -> None:
        """Clear pending and active clones (rollout restart)."""
        self._pending = []
        self._clones = []
        self._chunk_index = 0

    def advance_frames(
        self, trajectory: TrajectoryChunk
    ) -> tuple[DynamicActorTrajectory, ...]:
        """Advance clone lifecycles for one chunk; return actors to append."""
        first_ts = int(trajectory.timestamps_us[0])
        due = [entry for entry in self._pending if entry[1] <= self._chunk_index]
        self._pending = [
            entry for entry in self._pending if entry[1] > self._chunk_index
        ]
        self._chunk_index += 1
        for slot, _ in due:
            self._spawn_clone(slot, trajectory, first_ts)

        if not self._clones:
            return ()

        last_ts = int(trajectory.timestamps_us[-1])
        survivors: list[_ActiveClone] = []
        actors: list[DynamicActorTrajectory] = []
        for clone in self._clones:
            clone.chunks += 1
            self._check_collision(clone, trajectory)
            actors.append(clone.event.actor())
            track_exhausted = last_ts >= int(clone.event.timestamps_us[-1])
            if clone.chunks >= self._config.active_chunks or track_exhausted:
                reason = "track exhausted" if track_exhausted else "duration reached"
                logger.info(
                    f"[live-edit] obstacle despawned {clone.event.entity_id} "
                    f"after {clone.chunks} chunks ({reason})"
                )
            else:
                survivors.append(clone)
        self._clones = survivors
        return tuple(actors)

    def _spawn_clone(
        self, slot: int, trajectory: TrajectoryChunk, spawn_timestamp_us: int
    ) -> None:
        """Materialize one burst slot: distinct template, staggered ahead."""
        template_index = self._select_template_index(trajectory.vehicle_states[0])
        event = build_obstacle_event(
            self._templates[template_index],
            ego_state=trajectory.vehicle_states[0],
            spawn_timestamp_us=spawn_timestamp_us,
            config=self._config,
            ground_vertices=self._ground_vertices,
            entity_id=f"{OBSTACLE_ENTITY_PREFIX}-{self._event_count}",
            ahead_m=self._config.spawn_ahead_m + slot * self._config.spacing_m,
        )
        self._event_count += 1
        self._clones.append(_ActiveClone(event=event, template_index=template_index))
        start = event.translations_world[0]
        template = self._templates[template_index]
        logger.info(
            f"[live-edit] obstacle spawned {event.entity_id} slot={slot} "
            f"type={event.object_type} drift={template.drift_m:.1f}m "
            f"at ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})"
        )

    def _select_template_index(self, ego_state: VehicleState) -> int:
        """Pick a template whose recorded motion crosses the ego heading.

        Materialization tracks relative screen motion: a clone that holds a
        near-constant bearing (a lead car pacing the ego) sits against the
        same contradicting history pixels for many chunks and renders at
        ghost strength, while crossing/oncoming clones sweep the frame and
        materialize (probe_moving_clone; in-game bring-up 2026-08-20). Rank
        by |heading alignment| ascending (perpendicular first) and, within a
        burst, take the most-crossing template not already on screen so one
        traffic event shows DIFFERENT vehicles; rotate the pool start per
        burst so repeated events vary.
        """
        forward = np.array([np.cos(ego_state.yaw_rad), np.sin(ego_state.yaw_rad)])

        def crossness(index: int) -> float:
            template = self._templates[index]
            motion = (
                template.centers_world[-1, :2] - template.centers_world[0, :2]
            ).astype(np.float64)
            return abs(float(motion @ forward / (np.linalg.norm(motion) or 1.0)))

        ranked = sorted(range(len(self._templates)), key=crossness)
        pool_size = max(1, min(max(4, self._config.count), len(ranked)))
        top = ranked[:pool_size]
        offset = (self._burst_count - 1) % len(top) if top else 0
        rotated = top[offset:] + top[:offset]
        in_use = {clone.template_index for clone in self._clones}
        unused = [index for index in rotated if index not in in_use]
        return (unused or rotated)[0]

    def _check_collision(
        self, clone: _ActiveClone, trajectory: TrajectoryChunk
    ) -> None:
        """Log a hit when the ego passes within the collision radius."""
        if clone.hit_logged:
            return
        for timestamp_us, state in zip(
            trajectory.timestamps_us, trajectory.vehicle_states, strict=True
        ):
            center = clone.event.center_at(int(timestamp_us))
            if center is None:
                continue
            distance = float(np.hypot(center[0] - state.x_m, center[1] - state.y_m))
            if distance <= self._config.collision_radius_m:
                clone.hit_logged = True
                self._hit_count += 1
                logger.info(
                    f"[live-edit] obstacle HIT {clone.event.entity_id} "
                    f"distance={distance:.1f}m"
                )
                return


## Box-axis guidance (model side, GPU only)


class ObstacleGuidance:
    """Guide the flow along the with-box/without-box conditioning axis.

    Probe recipe (``probe_moving_clone.py`` with ``GUIDE_SCALE``): render the
    conditioning twice (with and without the clone), encode the no-box branch
    through a shadow encoder cache whose temporal state tracks the real one
    from chunk 0, and combine ``flow_nobox + s * (flow_box - flow_nobox)``
    per denoising step while a clone is on screen. Costs one extra lightVAE
    encode per chunk always, plus one raster and one extra network forward
    per step during events.
    """

    def __init__(self, scale: float) -> None:
        if scale <= 0.0:
            raise ValueError("ObstacleGuidance requires a positive scale")
        self._scale = float(scale)
        self._alt_frames: list[Any] | None = None
        self._alt_input: Any | None = None
        self._shadow_cache: Any | None = None
        self._ar_index = 0

    def install(self, backend: Any) -> None:
        """Hook the warmed backend's raster and session seams."""
        session = backend._session
        self._guard_transformer(session)
        rasterizer = backend._rasterizer

        original_first = backend.render_first_chunk
        original_next = backend.render_next_chunk

        def render_first_chunk(trajectory: Any) -> Any:
            self._stash_alt_frames(rasterizer, trajectory)
            return original_first(trajectory)

        def render_next_chunk(trajectory: Any) -> Any:
            self._stash_alt_frames(rasterizer, trajectory)
            return original_next(trajectory)

        backend.render_first_chunk = render_first_chunk
        backend.render_next_chunk = render_next_chunk

        original_start = session.start
        original_continue = session.continue_generation

        def start(initial_rgb: Any, condition_frames: Any, prompt: str) -> Any:
            self._reset_shadow(session)
            self._encode_shadow(session, condition_frames)
            return original_start(initial_rgb, condition_frames, prompt)

        def continue_generation(condition_frames: Any) -> Any:
            self._encode_shadow(session, condition_frames)
            return original_continue(condition_frames)

        session.start = start
        session.continue_generation = continue_generation
        self._wrap_predict_flow(session)
        logger.info(f"[live-edit] obstacle box-axis guidance armed s={self._scale}")

    def _stash_alt_frames(self, rasterizer: Any, trajectory: Any) -> None:
        """Render the obstacle-free conditioning when a clone is present."""
        actors = trajectory.dynamic_actors
        others = tuple(
            actor
            for actor in actors
            if not actor.entity_id.startswith(OBSTACLE_ENTITY_PREFIX)
        )
        if len(others) == len(actors):
            self._alt_frames = None
            return
        chunk = rasterizer.render_chunk(
            rig_poses_world=trajectory.rig_poses_world,
            timestamps_us=trajectory.timestamps_us,
            dynamic_actors=others,
        )
        self._alt_frames = [frame.rgb_host_uint8 for frame in chunk.frames]

    def _reset_shadow(self, session: Any) -> None:
        self._shadow_cache = session.pipeline.encoder.initialize_autoregressive_cache()
        self._ar_index = 0
        self._alt_frames = None
        self._alt_input = None

    def _encode_shadow(self, session: Any, condition_frames: Any) -> None:
        """Advance the shadow encoder; publish the patchified no-box input.

        Runs every chunk (with the identical conditioning when no clone is
        active) so the shadow cache's temporal state matches the real
        encoder's — an event can then start mid-run without a history
        mismatch between the two branches.

        The encode runs EAGERLY (:func:`_eager_vae_scope`): the encoder's
        CUDA-graph wrapper captures against one streaming cache's buffer
        addresses, so a captured replay fed the shadow cache would silently
        operate on the real cache's state. The eager shadow encode also
        keeps the wrapper's warmup/capture stream fed by the real cache
        only, so the real branch captures correctly.
        """
        import torch
        from flashdreams.core.distributed.context_parallel import split_inputs_cp

        pipeline = session.pipeline
        if self._shadow_cache is None:
            self._reset_shadow(session)
        frames = self._alt_frames if self._alt_frames is not None else condition_frames
        with torch.no_grad():
            hdmap = session._condition_tensor(frames)
            hdmap = split_inputs_cp(hdmap, seq_dim=1, cp_group=pipeline.V_group)
            with _eager_vae_scope(pipeline.encoder):
                encoded = pipeline.encoder(
                    input=hdmap,
                    autoregressive_index=self._ar_index,
                    cache=self._shadow_cache,
                )
            transformer = pipeline.diffusion_model.transformer
            self._alt_input = (
                transformer.patchify_and_maybe_split_cp(encoded)
                if self._alt_frames is not None
                else None
            )
        self._ar_index += 1

    def _wrap_predict_flow(self, session: Any) -> None:
        transformer = session.pipeline.diffusion_model.transformer
        original_predict_flow = transformer.predict_flow

        def guided_predict_flow(
            noisy_latent: Any, timestep: Any, cache: Any, input: Any = None
        ) -> Any:
            alt = self._alt_input
            if alt is None or transformer._finalizing_kv_cache:
                return original_predict_flow(noisy_latent, timestep, cache, input=input)
            flow_box = original_predict_flow(noisy_latent, timestep, cache, input=input)
            flow_nobox = original_predict_flow(noisy_latent, timestep, cache, input=alt)
            return flow_nobox + self._scale * (flow_box - flow_nobox)

        transformer.predict_flow = guided_predict_flow

    @staticmethod
    def _guard_transformer(session: Any) -> None:
        """Reject executors the predict_flow dispatch cannot intercept.

        CUDA graphs and ``compile_network`` are fine: the dispatch wraps the
        transformer's eager ``predict_flow`` (outside any capture), and the
        graph wrapper stages the ``hdmap_condition`` kwarg into its static
        buffers per call — the two forwards of a guided step are two replays
        of the same captured graph with different conditioning staged in.
        The native optimized-DiT executor is the one seam that bypasses the
        Python conditioning path.
        """
        transformer = session.pipeline.diffusion_model.transformer
        if getattr(transformer, "_optimized_dit_executor", None) is not None:
            raise RuntimeError(
                "obstacle guidance is not wired for the native optimized-DiT "
                "executor; set native_dit_acceleration: disabled in the "
                "world-model manifest."
            )


class _eager_vae_scope:
    """Route a graph-wrapped Wan VAE's calls through its eager encoder.

    The VAE's ``CUDAGraphWrapper`` passes the streaming cache dict through
    verbatim, binding captured kernels to ONE cache's buffer addresses; a
    replay fed a different cache would silently read/write the capture-time
    cache. Flipping ``_use_cuda_graph`` off for the duration makes the
    encode dispatch to the (possibly compiled) eager module with the cache
    that was actually passed. No-op for encoders without the knob (pixel
    shuffle, fakes).
    """

    def __init__(self, encoder: Any) -> None:
        self._vae = getattr(encoder, "vae", None)
        if self._vae is not None and not hasattr(self._vae, "_use_cuda_graph"):
            self._vae = None
        self._saved: bool | None = None

    def __enter__(self) -> None:
        if self._vae is not None:
            self._saved = self._vae._use_cuda_graph
            self._vae._use_cuda_graph = False

    def __exit__(self, *exc: object) -> None:
        if self._vae is not None and self._saved is not None:
            self._vae._use_cuda_graph = self._saved


def install_obstacle_guidance_on_backend(
    backend: Any, config: LiveEditObstacleConfig
) -> None:
    """Arm box-axis guidance before model warmup starts.

    Mirrors ``install_style_ability_on_backend``'s deferred attach: the hook
    install waits until ``warmup_model`` has built the pipeline. The session
    keeps its accelerated pipeline — the guidance is CUDA-graph safe (see
    :class:`ObstacleGuidance`), so no graph-free rebuild is needed.
    """
    if config.guide_scale <= 0.0:
        return
    session = getattr(backend, "_session", None)
    if session is None:
        raise ValueError(
            "--live-edit-obstacle guidance requires the omnidreams world-model backend."
        )
    guidance = ObstacleGuidance(config.guide_scale)
    original_warmup = session.warmup_model

    def warmup_and_install() -> None:
        original_warmup()
        guidance.install(backend)

    session.warmup_model = warmup_and_install
