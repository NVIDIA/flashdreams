# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side session flow for one external engine.

:class:`ExternalSessionDriver` runs the protocol's session flow on the model
thread: it answers ``hello`` with ``config``, asks for a seed at ``start`` and
at every restart, builds or resets the
:class:`~omnidreams_game_engine.model.WorldModelRollout` from the seed, grants
frames ``lead_chunks`` chunks ahead, generates each chunk once the engine has
completed it, and sends the frames back. A model host calls :meth:`poll`
from its model loop; nothing here depends on the FlashDreams runtime.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor

from omnidreams_game_engine.api.v0.assembly import ExternalChunk
from omnidreams_game_engine.api.v0.external import (
    CONNECTED,
    END,
    ERROR,
    HOLD,
    LOST,
    RELEASE,
    RESTART,
    SEED,
    START,
    EngineSource,
    ExternalEngineSession,
    SeedFrame,
    SourceEvent,
)
from omnidreams_game_engine.api.v0.messages import (
    Cancel,
    Config,
    Error,
    FrameReq,
    GenFrame,
    Hello,
    Input,
    Message,
    Restart,
    SeedReq,
    Start,
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

DISCONNECTED = "disconnected"
"""No engine connected."""
IDLE = "idle"
"""An engine is connected and has no driving session (title screen)."""
SEEDING = "seeding"
"""A seed was requested for the current epoch."""
RUNNING = "running"
"""Chunks of the current epoch are generated as the engine completes them."""


class SceneProvider(Protocol):
    """Build the scene an epoch starts from: the map plus the seed."""

    def scene_for_seed(self, hello: Hello, seed: SeedFrame) -> SceneDefinition: ...


@dataclass(frozen=True, slots=True)
class EncodedFrame:
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
        """One encoded frame per generated frame, in order."""
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
    """``[1, 1, T, C, H, W]`` video in ``[-1, 1]`` to uint8 ``[T, H, W, C]`` on the CPU."""
    if video_bvtchw.ndim != 6 or tuple(video_bvtchw.shape[:2]) != (1, 1):
        raise ValueError(
            f"Expected [1, 1, T, C, H, W]; got {tuple(video_bvtchw.shape)}"
        )
    frames = video_bvtchw[0, 0].detach().float().clamp(-1.0, 1.0)
    frames = ((frames + 1.0) * 127.5).round().to(torch.uint8)
    return frames.permute(0, 2, 3, 1).cpu()


@dataclass(frozen=True, slots=True)
class DriverConfig:
    model_fps: int = 30
    lead_chunks: int = 2
    """Chunks granted beyond the one being generated."""
    display_delay_ticks: int = 28
    render: str | None = "off"
    chunk_timeout_s: float = 5.0
    """Wait of :class:`ExternalEngineSession` when used without a driver."""
    seed_width: int | None = None
    seed_height: int | None = None
    seed_fov_deg: float | None = None
    seed_warmup_renders: int | None = None
    return_width: int | None = None
    return_height: int | None = None
    checkpoint: str | None = None
    profile: str | None = None
    preset: str | None = None
    host_build: str | None = None
    config_extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lead_chunks < 0:
            raise ValueError("lead_chunks must not be negative")


@dataclass(frozen=True, slots=True)
class DriverStep:
    """One generated chunk."""

    epoch: int
    ar_index: int
    chunk: ExternalChunk
    step: WorldModelStep
    encode_ms: float


@dataclass(slots=True)
class DriverStats:
    epochs: int = 0
    chunks: int = 0
    frames_sent: int = 0
    seeds: int = 0
    stale_seeds: int = 0
    connects: int = 0
    lost: int = 0
    lost_reasons: list[str] = field(default_factory=list)
    ends: int = 0
    engine_errors: list[str] = field(default_factory=list)


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
        self.state = DISCONNECTED
        self.hello: Hello | None = None
        self.timeline: ChunkTimeline | None = None
        self.rollout: WorldModelRollout | None = None
        self.epoch = 0
        self.ar_index = 0
        self.granted_g = 0
        self.t0_tick: int | None = None
        self.last_tick: int | None = None
        self.engine_held = False
        self.stats = DriverStats()
        self._scene: SceneDefinition | None = None
        self._seed: SeedFrame | None = None
        self._session: ExternalEngineSession | None = None
        self._loaded_scene_key: tuple[str, str] | None = None

    # ------------------------------------------------------------------ API

    def poll(self, timeout_s: float = 0.1) -> DriverStep | None:
        """Handle the engine's events, then generate the next chunk if it is complete."""
        wait = 0.0 if self.state == RUNNING else timeout_s
        for event in self.source.poll_events(wait):
            self._handle(event)
        if self.state != RUNNING:
            return None
        chunk = self.source.wait_chunk(self.epoch, self.ar_index, timeout_s)
        if chunk is None:
            return None
        return self._generate(chunk)

    def request_refresh(self, reason: str = "refresh") -> bool:
        """Re-anchor on a new seed at the engine's current pose (drift refresh)."""
        if self.state != RUNNING or self.timeline is None:
            return False
        self.source.send(Restart(reason=reason, epoch=self.epoch + 1))
        tick = (self.last_tick or 0) + self.timeline.frame_ticks
        self._begin_epoch(tick, reason)
        return True

    def forward_input(self, action: str, phase: str) -> None:
        """Send a key edge from a host-side view to the engine."""
        self.source.send(Input(action=action, phase=phase, t_host_us=now_us()))

    def status(self) -> dict[str, Any]:
        """Fields for the host's periodic ``status`` message."""
        return {
            "state": self.state,
            "epoch": self.epoch,
            "chunk": self.ar_index,
            "chunks": self.stats.chunks,
            "engine_held": self.engine_held,
        }

    def close(self) -> None:
        """Close the rollout; the pipeline, renderer and source stay with their owner."""
        if self.rollout is not None:
            self.rollout.close()
            self.rollout = None
        self._session = None

    # ------------------------------------------------------------- handling

    def _handle(self, event: SourceEvent) -> None:
        kind = event.kind
        if kind == CONNECTED:
            assert event.hello is not None
            self._on_connected(event.hello)
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
            self.stats.lost += 1
            self.stats.lost_reasons.append(event.reason or "")
            self.state = DISCONNECTED
            self.engine_held = False
        elif kind == ERROR:
            self.stats.engine_errors.append(event.reason or "")
            _LOGGER.warning("Engine reported an error: %s", event.message)

    def _on_connected(self, hello: Hello) -> None:
        self.stats.connects += 1
        self.hello = hello
        self.engine_held = False
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
                checkpoint=self.config.checkpoint,
                profile=self.config.profile,
                preset=self.config.preset,
                host_build=self.config.host_build,
                extra=dict(self.config.config_extra),
            )
        )
        # A (re)connected engine starts or resumes its session with ``start``.
        self.state = IDLE

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
                warmup_renders=self.config.seed_warmup_renders,
            )
        )
        self.state = SEEDING

    def _on_seed(self, seed: SeedFrame) -> None:
        if self.state != SEEDING or seed.epoch != self.epoch or self.hello is None:
            self.stats.stale_seeds += 1
            return
        self.stats.seeds += 1
        scene = self.scene_provider.scene_for_seed(self.hello, seed)
        key = (str(scene.scene_path), scene.scene_id)
        if key != self._loaded_scene_key:
            self.condition_renderer.load_scene(scene)
            self._loaded_scene_key = key
        self._scene = scene
        self._seed = seed
        self.t0_tick = seed.tick
        self.last_tick = seed.tick
        if self.rollout is None:
            self.rollout = WorldModelRollout(
                pipeline=self.pipeline,
                scene=scene,
                engine_factory=self._new_engine,
                postprocess=self.postprocess,
                trace_chunk_lifecycle=self.trace_chunk_lifecycle,
            )
        else:
            self.rollout.reset(scene=scene)
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
            chunk_timeout_s=self.config.chunk_timeout_s,
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
        result = DriverStep(
            epoch=self.epoch,
            ar_index=self.ar_index,
            chunk=chunk,
            step=step,
            encode_ms=encode_ms,
        )
        self.ar_index += 1
        self.last_tick = chunk.frames[-1].tick
        self.stats.chunks += 1
        self.stats.frames_sent += len(frames)
        return result

    def _end_session(self) -> None:
        if self.rollout is not None:
            self.rollout.close()
            self.rollout = None
        self._session = None


__all__ = [
    "DISCONNECTED",
    "IDLE",
    "RUNNING",
    "SEEDING",
    "DriverConfig",
    "DriverStats",
    "DriverStep",
    "EncodedFrame",
    "ExternalSessionDriver",
    "FrameEncoder",
    "RawRgbEncoder",
    "SceneProvider",
    "video_to_uint8_thwc",
]
