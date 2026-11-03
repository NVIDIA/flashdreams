# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side session flow for one external engine, run from the model loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

import torch
from torch import Tensor

from omnidreams_game_engine.api.v0.assembly import ExternalChunk
from omnidreams_game_engine.api.v0.external import (
    CONNECTED,
    END,
    ERROR,
    HOLD,
    INVALID,
    LOST,
    RELEASE,
    RESTART,
    SEED,
    START,
    VIEWS,
    EngineSource,
    ExternalEngineSession,
    SeedFrame,
    SourceEvent,
)
from omnidreams_game_engine.api.v0.messages import (
    PROTOCOL,
    Cancel,
    Config,
    Error,
    Frame,
    FrameReq,
    GenFrame,
    Hello,
    Input,
    Message,
    Restart,
    SeedReq,
    Start,
    ViewFrame,
    Views,
)
from omnidreams_game_engine.api.v0.recorder import now_us
from omnidreams_game_engine.api.v0.timeline import ChunkTimeline
from omnidreams_game_engine.contracts import ConditionRenderer
from omnidreams_game_engine.model import (
    VideoPostprocessor,
    WorldModelRollout,
    WorldModelStep,
)
from omnidreams_game_engine.types import DriverCommand, SceneDefinition

_LOGGER = logging.getLogger(__name__)

MAP_VIEW = "map"
"""Host view of the conditioning frames (``view_frame{view: "map"}``)."""

HOST_VIEWS = frozenset({MAP_VIEW})
"""Host views this driver serves."""

DriverState = Literal["disconnected", "idle", "seeding", "running"]
"""Session state of :class:`ExternalSessionDriver`."""

DISCONNECTED: Final = "disconnected"
"""No engine connected."""
IDLE: Final = "idle"
"""An engine is connected and has no driving session (title screen)."""
SEEDING: Final = "seeding"
"""A seed was requested for the current epoch."""
RUNNING: Final = "running"
"""Chunks of the current epoch are generated as the engine completes them."""


class SceneProvider(Protocol):
    """Build the scene an epoch starts from: the map plus the seed."""

    def scene_for_seed(self, hello: Hello, seed: SeedFrame) -> SceneDefinition: ...


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    """One encoded image and its size."""

    data: bytes
    width: int
    height: int


class FrameEncoder(Protocol):
    """Encode generated video for ``gen_frame`` payloads."""

    @property
    def encoding(self) -> str: ...

    @property
    def quality(self) -> int | None: ...

    def encode(self, video_bvtchw: Tensor) -> list[EncodedFrame]:
        """Encode each frame of ``video_bvtchw``, in order."""
        ...


class RawRgbEncoder:
    """Uncompressed RGB24 frames (``raw``): for tests and same-machine engines."""

    @property
    def encoding(self) -> str:
        return "raw"

    @property
    def quality(self) -> int | None:
        return None

    def encode(self, video_bvtchw: Tensor) -> list[EncodedFrame]:
        frames = video_to_uint8_thwc(video_bvtchw)
        height, width = int(frames.shape[1]), int(frames.shape[2])
        return [
            EncodedFrame(frame.contiguous().numpy().tobytes(), width, height)
            for frame in frames
        ]


def video_to_uint8_thwc(video_bvtchw: Tensor) -> Tensor:
    """Convert ``[1, 1, T, C, H, W]`` video in ``[-1, 1]`` to uint8 ``[T, H, W, C]``."""
    if video_bvtchw.ndim != 6 or tuple(video_bvtchw.shape[:2]) != (1, 1):
        raise ValueError(
            f"Expected [1, 1, T, C, H, W]; got {tuple(video_bvtchw.shape)}"
        )
    frames = video_bvtchw[0, 0].detach().float().clamp(-1.0, 1.0)
    frames = ((frames + 1.0) * 127.5).round().to(torch.uint8)
    return frames.permute(0, 2, 3, 1).cpu()


@dataclass(frozen=True, slots=True)
class DriverConfig:
    """Settings of :class:`ExternalSessionDriver`; most are sent in ``config``."""

    model_fps: int = 30
    """Frames the model generates per second; ``sim_hz`` must be a multiple."""
    lead_chunks: int = 2
    """Chunks granted beyond the one being generated; at least 1."""
    display_delay_ticks: int = 28
    """Presentation delay of a generated frame after its tick (0.47 s at 60 Hz)."""
    render: str = "off"
    """Engine world rendering between seeds: ``off`` or ``always``."""
    seed_width: int | None = None
    """Seed image width asked for in ``seed_req``; ``None`` leaves it to the engine."""
    seed_height: int | None = None
    """Seed image height asked for in ``seed_req``; ``None`` leaves it to the engine."""
    seed_fov_deg: float | None = None
    """Seed camera field of view; ``None`` leaves it to the engine."""
    return_width: int | None = None
    """Width of the frames the encoder produces, announced in ``config``."""
    return_height: int | None = None
    """Height of the frames the encoder produces, announced in ``config``."""
    preset: str | None = None
    """Name of the host's model preset, for the engine's logs."""
    host_build: str | None = None
    """Build of the host, for the engine's logs."""
    config_extra: Mapping[str, Any] = field(default_factory=dict)
    """Further ``config`` fields, sent as given."""

    def __post_init__(self) -> None:
        if self.lead_chunks < 1:
            raise ValueError("lead_chunks must be at least 1")


@dataclass(frozen=True, slots=True)
class DriverStep:
    """One generated chunk."""

    epoch: int
    ar_index: int
    chunk: ExternalChunk
    step: WorldModelStep
    encode_ms: float
    views_ms: float = 0.0
    """Encoding and sending the host views' frames (``view_frame``)."""


@dataclass(slots=True)
class DriverStats:
    """Counts of the driver's session events."""

    connects: int = 0
    epochs: int = 0
    seeds: int = 0
    chunks: int = 0
    view_frames_sent: int = 0
    ends: int = 0
    lost_reasons: list[str] = field(default_factory=list)
    """Reason of every lost connection."""


class ExternalSessionDriver:
    """Run one external engine's sessions on the model thread."""

    def __init__(
        self,
        *,
        source: EngineSource,
        pipeline: Any,
        scene_provider: SceneProvider,
        condition_renderer: ConditionRenderer,
        encoder: FrameEncoder | None = None,
        config: DriverConfig = DriverConfig(),
        postprocess: VideoPostprocessor | None = None,
        trace_chunk_lifecycle: bool = False,
    ) -> None:
        self.source = source
        self.pipeline = pipeline
        self.scene_provider = scene_provider
        self.condition_renderer = condition_renderer
        self.encoder: FrameEncoder = encoder if encoder is not None else RawRgbEncoder()
        self.config = config
        self.postprocess = postprocess
        self.trace_chunk_lifecycle = trace_chunk_lifecycle
        self.state: DriverState = DISCONNECTED
        self.hello: Hello | None = None
        self.timeline: ChunkTimeline | None = None
        self.rollout: WorldModelRollout | None = None
        self.epoch = 0
        self.ar_index = 0
        self.granted_g = 0
        self.t0_tick: int | None = None
        self.last_tick: int | None = None
        self.engine_held = False
        self.views: frozenset[str] = frozenset()
        """Host views the engine shows; their frames follow each chunk."""
        self.stats = DriverStats()
        self._scene: SceneDefinition | None = None
        self._seed: SeedFrame | None = None
        self._session: ExternalEngineSession | None = None
        self._loaded_scene_key: tuple[str, str] | None = None

    ## API

    def poll(self, timeout_s: float = 0.1) -> DriverStep | None:
        """Handle the engine's events; generate the next chunk once it is complete."""
        wait = 0.0 if self.state == RUNNING else timeout_s
        for event in self.source.poll_events(wait):
            self._handle(event)
        if self.state != RUNNING:
            return None
        chunk = self.source.wait_chunk(self.epoch, self.ar_index, timeout_s)
        if chunk is None:
            return None
        try:
            return self._generate(chunk)
        except Exception:
            # The chunk is consumed: a new seed restarts the session.
            self._reanchor("error")
            raise

    def request_refresh(self, reason: str = "refresh") -> bool:
        """Re-anchor on a new seed at the engine's current pose (drift refresh)."""
        if self.state != RUNNING:
            return False
        self.source.send(Restart(reason=reason, epoch=self.epoch + 1))
        self._reanchor(reason)
        return True

    def forward_input(self, action: str, phase: str) -> None:
        """Send a key edge from a host-side view to the engine."""
        self.source.send(Input(action=action, phase=phase, t_host_us=now_us()))

    def status(self) -> dict[str, Any]:
        """Return the fields of the host's periodic ``status`` message."""
        return {
            "state": self.state,
            "epoch": self.epoch,
            "chunk": self.ar_index,
            "chunks": self.stats.chunks,
            "engine_held": self.engine_held,
        }

    def close(self) -> None:
        """Close the rollout; the pipeline, renderer and source stay open."""
        self._end_session()

    ## Handling

    def _handle(self, event: SourceEvent) -> None:
        kind = event.kind
        if kind == CONNECTED:
            assert isinstance(event.message, Hello)
            self._on_connected(event.message)
        elif kind == START:
            message = event.message
            if self.state == DISCONNECTED or not isinstance(message, Start):
                return
            self._begin_epoch(message.tick, message.reason)
        elif kind == SEED:
            assert event.seed is not None
            self._on_seed(event.seed)
        elif kind == RESTART:
            if self.state not in (RUNNING, SEEDING):
                return
            message = event.message
            tick = getattr(message, "tick", None)
            if tick is None:
                tick = self.last_tick if self.last_tick is not None else 0
            self._begin_epoch(int(tick), event.reason or "restart")
        elif kind == HOLD:
            self.engine_held = True
        elif kind == RELEASE:
            self.engine_held = False
        elif kind == END:
            self.stats.ends += 1
            self._end_session()
            if self.state != DISCONNECTED:
                self.state = IDLE
        elif kind == LOST:
            self.stats.lost_reasons.append(event.reason or "")
            self._end_session()
            self.state = DISCONNECTED
            self.engine_held = False
            self.views = frozenset()
        elif kind == VIEWS:
            message = event.message
            if isinstance(message, Views):
                self.views = frozenset(message.views) & HOST_VIEWS
        elif kind == ERROR:
            _LOGGER.warning("Engine reported an error: %s", event.message)
        elif kind == INVALID:
            frame = event.message
            stale = isinstance(frame, Frame) and frame.epoch != self.epoch
            if self.state in (RUNNING, SEEDING) and not stale:
                self._reanchor("invalid")

    def _on_connected(self, hello: Hello) -> None:
        self._end_session()
        self.stats.connects += 1
        self.hello = hello
        self.engine_held = False
        self.views = frozenset()
        try:
            self.timeline = ChunkTimeline.from_frame_count(
                self.pipeline.get_num_output_frames,
                sim_hz=hello.sim_hz,
                model_fps=self.config.model_fps,
            )
        except ValueError as error:
            self.source.send(Error(code="timing", message=str(error)))
            self.state = DISCONNECTED
            return
        self.source.send(
            Config(
                session=hello.session,
                api=PROTOCOL,
                model_fps=self.config.model_fps,
                frame_ticks=self.timeline.frame_ticks,
                chunk_plan=self.timeline.plan_list,
                lead_chunks=self.config.lead_chunks,
                display_delay_ticks=self.config.display_delay_ticks,
                return_encoding=self.encoder.encoding,
                return_width=self.config.return_width,
                return_height=self.config.return_height,
                return_quality=self.encoder.quality,
                render=self.config.render,
                preset=self.config.preset,
                host_build=self.config.host_build,
                extra=dict(self.config.config_extra),
            )
        )
        # A (re)connected engine starts or resumes its session with ``start``.
        self.state = IDLE

    def _reanchor(self, reason: str) -> None:
        assert self.timeline is not None
        self._begin_epoch((self.last_tick or 0) + self.timeline.frame_ticks, reason)

    def _begin_epoch(self, tick: int, reason: str) -> None:
        if self.timeline is None:
            return
        if self.state in (RUNNING, SEEDING):
            self.source.send(Cancel(epoch=self.epoch))
        self.epoch += 1
        self.stats.epochs += 1
        self.ar_index = 0
        self.granted_g = 0
        self.t0_tick = None
        self._seed = None
        self.source.begin_epoch(self.epoch, self.timeline)
        self.source.send(
            SeedReq(
                epoch=self.epoch,
                tick=int(tick),
                reason=reason,
                width=self.config.seed_width,
                height=self.config.seed_height,
                fov_deg=self.config.seed_fov_deg,
            )
        )
        self.state = SEEDING

    def _on_seed(self, seed: SeedFrame) -> None:
        if self.state != SEEDING or seed.epoch != self.epoch or self.hello is None:
            return
        self.stats.seeds += 1
        try:
            scene = self.scene_provider.scene_for_seed(self.hello, seed)
            key = (str(scene.scene_path), scene.scene_id)
            if key != self._loaded_scene_key:
                self.condition_renderer.load_scene(scene)
                self._loaded_scene_key = key
        except Exception as error:
            _LOGGER.exception("Seed of epoch %d is unusable", seed.epoch)
            self.source.send(Error(code="protocol", message=f"Unusable seed: {error}"))
            self._reanchor("invalid")
            return
        self._scene = scene
        self._seed = seed
        self.t0_tick = seed.tick
        self.last_tick = seed.tick
        self._end_session()
        self.rollout = WorldModelRollout(
            pipeline=self.pipeline,
            scene=scene,
            engine_factory=self._new_engine,
            postprocess=self.postprocess,
            trace_chunk_lifecycle=self.trace_chunk_lifecycle,
        )
        self.state = RUNNING
        self._grant_through(self.config.lead_chunks)

    def _new_engine(self) -> ExternalEngineSession:
        if self._scene is None or self.timeline is None:
            raise RuntimeError("An external epoch starts from a seed")
        self._session = ExternalEngineSession(
            source=self.source,
            epoch=self.epoch,
            condition_renderer=self.condition_renderer,
            timeline=self.timeline,
            base_timestamp_us=self._scene.initial_timestamp_us,
            initial_frame=None if self._seed is None else self._seed.state,
        )
        return self._session

    def _grant_through(self, chunk: int) -> None:
        assert self.timeline is not None and self.t0_tick is not None
        timeline = self.timeline
        for g in timeline.grant_frames(chunk, after_g=self.granted_g):
            k = timeline.chunk_of(g)
            self.source.send(
                FrameReq(
                    epoch=self.epoch,
                    chunk=k,
                    index=g - timeline.first_frame(k),
                    g=g,
                    tick=timeline.tick_of(g, self.t0_tick),
                )
            )
            self.granted_g = g

    def _generate(self, chunk: ExternalChunk) -> DriverStep:
        rollout = self.rollout
        session = self._session
        assert rollout is not None and session is not None
        self._grant_through(self.ar_index + self.config.lead_chunks)
        session.supply(chunk)
        count = rollout.frame_count(self.ar_index)
        step = rollout.step(
            autoregressive_index=self.ar_index,
            commands=tuple(DriverCommand() for _ in range(count)),
        )
        encode_started = time.perf_counter()
        encoded = self.encoder.encode(step.video_bvtchw)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        if len(encoded) != len(chunk.frames):
            raise ValueError("The encoder must return one frame per generated frame")
        sent_us = now_us()
        frames: list[tuple[Message, bytes]] = [
            (
                GenFrame(
                    epoch=self.epoch,
                    chunk=self.ar_index,
                    g=state.g,
                    tick=state.tick,
                    encoding=self.encoder.encoding,
                    width=item.width,
                    height=item.height,
                    t_host_us=sent_us,
                ),
                item.data,
            )
            for state, item in zip(chunk.frames, encoded, strict=True)
        ]
        self.source.send_frames(frames)
        views_ms = self._send_views(chunk, step) if self.views else 0.0
        result = DriverStep(
            epoch=self.epoch,
            ar_index=self.ar_index,
            chunk=chunk,
            step=step,
            encode_ms=encode_ms,
            views_ms=views_ms,
        )
        self.ar_index += 1
        self.last_tick = chunk.frames[-1].tick
        self.stats.chunks += 1
        return result

    def _send_views(self, chunk: ExternalChunk, step: WorldModelStep) -> float:
        """Send the map view's frames; return the time it took in ms."""
        started = time.perf_counter()
        encoded = self.encoder.encode(step.engine.condition.hdmap_bvtchw)
        sent_us = now_us()
        for state, item in zip(chunk.frames, encoded, strict=True):
            self.source.send(
                ViewFrame(
                    view=MAP_VIEW,
                    epoch=self.epoch,
                    chunk=self.ar_index,
                    g=state.g,
                    tick=state.tick,
                    encoding=self.encoder.encoding,
                    width=item.width,
                    height=item.height,
                    t_host_us=sent_us,
                ),
                item.data,
            )
        self.stats.view_frames_sent += len(encoded)
        return (time.perf_counter() - started) * 1000.0

    def _end_session(self) -> None:
        if self.rollout is not None:
            self.rollout.close()
            self.rollout = None
        self._session = None


__all__ = [
    "DISCONNECTED",
    "HOST_VIEWS",
    "IDLE",
    "MAP_VIEW",
    "RUNNING",
    "SEEDING",
    "DriverConfig",
    "DriverState",
    "DriverStats",
    "DriverStep",
    "EncodedFrame",
    "ExternalSessionDriver",
    "FrameEncoder",
    "RawRgbEncoder",
    "SceneProvider",
    "video_to_uint8_thwc",
]
