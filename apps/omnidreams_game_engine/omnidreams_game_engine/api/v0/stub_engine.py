# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Python external engine for protocol tests and host bring-up.

The stub connects to a host, sends ``hello`` and ``start``, answers
``seed_req`` with a small synthetic image and ``frame_req`` with the state of
the granted tick, never simulating past the newest grant. Its states come
from a scripted drive in the ``flu/0`` vocabulary, or from a recorded
``state.jsonl`` in an engine's own vocabulary (sent verbatim with
``--state-format``, for a host with a matching decoder).
A script can pause, restart, drop the connection or go silent at given
ticks, which exercises every session path of the protocol.

    python -m omnidreams_game_engine.api.v0.stub_engine --port 8095 --frames 600
"""

from __future__ import annotations

import argparse
import json
import math
import select
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from omnidreams_game_engine.api.v0.framing import MessageDecoder, encode_message
from omnidreams_game_engine.api.v0.messages import (
    PROTOCOL,
    Cancel,
    Config,
    End,
    Error,
    Frame,
    FrameReq,
    GenFrame,
    Hello,
    Hold,
    Message,
    Ping,
    Pong,
    ProtocolError,
    Release,
    Restart,
    SeedReq,
    Start,
    from_header,
)
from omnidreams_game_engine.api.v0.recorder import now_us
from omnidreams_game_engine.api.v0.state import (
    FLU_STATE_FORMAT,
    ActorState,
    FluStateCodec,
    FrameState,
)
from omnidreams_game_engine.api.v0.timeline import ChunkTimeline
from omnidreams_game_engine.types import DriverCommand, VehicleState

StateFn = Callable[[int], Mapping[str, Any]]
"""Engine state of one tick, in the wire vocabulary."""


@dataclass(frozen=True, slots=True)
class ScriptedDrive:
    """A taxi on a gentle left curve with one car ahead and one pedestrian.

    The pedestrian is visible on alternate half seconds, so chunks contain
    actors that appear and leave.
    """

    sim_hz: int = 60
    speed_mps: float = 8.0
    yaw_rate_radps: float = 0.05

    def frame_state(self, tick: int) -> FrameState:
        t = tick / self.sim_hz
        yaw = self.yaw_rate_radps * t
        if abs(self.yaw_rate_radps) > 1e-9:
            radius = self.speed_mps / self.yaw_rate_radps
            x = radius * math.sin(yaw)
            y = radius * (1.0 - math.cos(yaw))
        else:
            x, y = self.speed_mps * t, 0.0
        pose = np.eye(4, dtype=np.float32)
        pose[0, 0] = pose[1, 1] = math.cos(yaw)
        pose[0, 1] = -math.sin(yaw)
        pose[1, 0] = math.sin(yaw)
        pose[:3, 3] = (x, y, 1.5)
        vehicle = VehicleState(
            x_m=x,
            y_m=y,
            z_m=0.0,
            yaw_rad=yaw,
            speed_mps=self.speed_mps,
            steer_rad=0.02,
            yaw_rate_radps=self.yaw_rate_radps,
        )
        half = np.float32(0.5) * np.float32(yaw)
        quat = np.asarray([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float32)
        actors = [
            ActorState(
                entity_id="car-1",
                object_type="Car",
                position_world=np.asarray(
                    (x + 20.0 * math.cos(yaw), y + 20.0 * math.sin(yaw), 0.8),
                    dtype=np.float32,
                ),
                orientation_xyzw=quat,
                dimensions_lwh=np.asarray((4.5, 1.9, 1.6), dtype=np.float32),
            )
        ]
        if (tick // (self.sim_hz // 2)) % 2 == 0:
            actors.append(
                ActorState(
                    entity_id="ped-7",
                    object_type="Pedestrian",
                    position_world=np.asarray(
                        (x + 12.0, y + 4.0, 0.9), dtype=np.float32
                    ),
                    orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
                    dimensions_lwh=np.asarray((0.5, 0.5, 1.8), dtype=np.float32),
                )
            )
        return FrameState(
            g=-1,
            tick=tick,
            rig_to_world=pose,
            vehicle=vehicle,
            command=DriverCommand(throttle=0.3, steer=0.1),
            actors=tuple(actors),
        )

    def state(self, tick: int) -> dict[str, Any]:
        return FluStateCodec().encode(self.frame_state(tick))


def jsonl_states(path: str | Path) -> StateFn:
    """States of a recorded ``state.jsonl``, by tick (``tick``, else ``frame``)."""
    by_tick: dict[int, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("tick", record.get("frame"))
            if key is not None:
                by_tick[int(key)] = record
    if not by_tick:
        raise ValueError(f"{path} holds no states")
    ticks = sorted(by_tick)

    def state(tick: int) -> Mapping[str, Any]:
        index = min(range(len(ticks)), key=lambda i: abs(ticks[i] - tick))
        return by_tick[ticks[index]]

    return state


@dataclass(frozen=True, slots=True)
class StubScript:
    """What the stub does besides answering grants."""

    end_after_frames: int | None = None
    """Send ``end`` after answering this many granted frames in total."""
    realtime: bool = False
    """Pace ticks to wall time at ``sim_hz`` (else run as fast as granted)."""
    restart_at_ticks: tuple[int, ...] = ()
    hold_at_tick: int | None = None
    hold_s: float = 0.0
    drop_at_tick: int | None = None
    """Close the connection abruptly once at this tick, then reconnect."""
    silent_at_tick: int | None = None
    silent_s: float = 0.0
    """Stop reading and answering for ``silent_s`` once at this tick."""


@dataclass(slots=True)
class StubSummary:
    connects: int = 0
    configs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    seeds: list[tuple[int, int]] = field(default_factory=list)
    """``(epoch, tick)`` of every seed sent."""
    frames_sent: int = 0
    grants: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    """``(epoch, chunk, g, tick, chunks fully received in that epoch)`` per grant."""
    gen_frames: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    """``(epoch, chunk, g, tick, payload bytes)`` per generated frame received."""
    cancels: list[int | None] = field(default_factory=list)
    holds: list[int] = field(default_factory=list)
    restarts: list[int] = field(default_factory=list)
    host_messages: dict[str, int] = field(default_factory=dict)
    ended: bool = False
    final_tick: int = 0


class _Reconnect(Exception):
    pass


class StubEngine:
    """Run with :meth:`run` (blocking) or :meth:`start` (a daemon thread)."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        session: str = "stub-engine",
        state_fn: StateFn | None = None,
        state_format: str = FLU_STATE_FORMAT,
        sim_hz: int = 60,
        script: StubScript = StubScript(),
        seed_size: tuple[int, int] = (8, 4),
        start_tick: int = 0,
        reconnect_window_s: float = 30.0,
    ) -> None:
        self.address = (host, int(port))
        self.session = session
        self.sim_hz = int(sim_hz)
        self.state_fn: StateFn = state_fn or ScriptedDrive(sim_hz=self.sim_hz).state
        self.state_format = state_format
        self.script = script
        self.seed_size = seed_size
        self.reconnect_window_s = float(reconnect_window_s)
        self.summary = StubSummary(final_tick=start_tick)
        self.tick = int(start_tick)
        self._sock: socket.socket | None = None
        self._decoder = MessageDecoder()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._timeline: ChunkTimeline | None = None
        self._epoch: int | None = None
        self._pending_seed: SeedReq | None = None
        self._requests: dict[int, FrameReq] = {}
        self._received: dict[tuple[int, int], int] = {}
        self._complete: dict[int, int] = {}
        self._script_done: set[str] = set()
        self._restart_ticks = set(script.restart_at_ticks)
        self._held_until: float | None = None
        self._host_held = False
        self._silent_until: float | None = None
        self._wall_t0 = 0.0
        self._tick_t0 = 0
        self.failure: BaseException | None = None

    # ---------------------------------------------------------------- control

    def start(self) -> StubEngine:
        self._thread = threading.Thread(
            target=self._run_safely, name="stub-engine", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout_s: float) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()

    def run(self) -> StubSummary:
        self._connect("start")
        while not self._stop.is_set():
            try:
                self._loop_once()
            except _Reconnect:
                if not self._reconnect():
                    break
        self._close(graceful=self.summary.ended)
        self.summary.final_tick = self.tick
        return self.summary

    def _run_safely(self) -> None:
        try:
            self.run()
        except BaseException as error:  # noqa: BLE001 - reported through ``failure``
            self.failure = error
            self._close()

    # ------------------------------------------------------------- connection

    def _connect(self, reason: str) -> None:
        sock = socket.create_connection(self.address, timeout=5.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(None)
        self._sock = sock
        self._decoder = MessageDecoder()
        self.summary.connects += 1
        self._send(
            Hello(
                session=self.session,
                api=PROTOCOL,
                state_format=self.state_format,
                driver="autopilot",
                sim_hz=self.sim_hz,
                tick=self.tick,
                engine_build="stub",
                t_engine_us=now_us(),
            )
        )
        deadline = time.monotonic() + 10.0
        got_config = False
        while not got_config:
            if time.monotonic() > deadline:
                raise TimeoutError("The host sent no config")
            got_config = self._read(0.05)
            if self._stop.is_set():
                return
        self._epoch = None
        self._pending_seed = None
        self._requests.clear()
        self._send(Start(tick=self.tick, reason=reason))

    def _reconnect(self) -> bool:
        self._close()
        self._timeline = None
        deadline = time.monotonic() + self.reconnect_window_s
        while not self._stop.is_set() and time.monotonic() < deadline:
            try:
                self._connect("reconnect")
                return True
            except (OSError, TimeoutError, _Reconnect):
                self._close()
                time.sleep(0.05)
        return False

    def _close(self, *, graceful: bool = False) -> None:
        sock, self._sock = self._sock, None
        if sock is None:
            return
        if graceful:
            # Half-close and drain, so the last messages are not lost to a reset
            # while the host is still sending frames.
            try:
                sock.shutdown(socket.SHUT_WR)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    ready, _, _ = select.select([sock], [], [], 0.05)
                    if ready and not sock.recv(1 << 16):
                        break
            except OSError:
                pass
        try:
            sock.close()
        except OSError:
            pass

    def _send(self, message: Message, payload: bytes = b"") -> None:
        if self._sock is None:
            raise _Reconnect()
        try:
            self._sock.sendall(encode_message(message.to_header(), payload))
        except OSError as error:
            raise _Reconnect() from error

    def _read(self, timeout_s: float) -> bool:
        """Handle every available message; return whether a config arrived."""
        sock = self._sock
        if sock is None:
            raise _Reconnect()
        try:
            ready, _, _ = select.select([sock], [], [], timeout_s)
        except (OSError, ValueError) as error:
            raise _Reconnect() from error
        if not ready:
            return False
        try:
            data = sock.recv(1 << 16)
        except OSError as error:
            raise _Reconnect() from error
        if not data:
            raise _Reconnect()
        got_config = False
        for header, payload in self._decoder.feed(data):
            got_config |= self._handle(header, payload)
        return got_config

    # -------------------------------------------------------------- messages

    def _handle(self, header: dict[str, Any], payload: bytes) -> bool:
        try:
            message = from_header(header)
        except ProtocolError:
            return False
        kind = message.TYPE or str(header.get("type"))
        self.summary.host_messages[kind] = self.summary.host_messages.get(kind, 0) + 1
        if isinstance(message, Config):
            self.summary.configs.append(message.to_header())
            self._timeline = ChunkTimeline(
                plan=tuple(message.chunk_plan),
                sim_hz=self.sim_hz,
                model_fps=message.model_fps,
            )
            return True
        if isinstance(message, Ping):
            self._send(Pong(t_host_us=message.t_host_us, t_engine_us=now_us()))
        elif isinstance(message, SeedReq):
            self._pending_seed = message
            self._requests.clear()
        elif isinstance(message, FrameReq):
            if message.epoch == self._epoch or (
                self._pending_seed is not None
                and message.epoch == self._pending_seed.epoch
            ):
                self._requests[message.g] = message
                self.summary.grants.append(
                    (
                        message.epoch,
                        message.chunk,
                        message.g,
                        message.tick,
                        self._complete.get(message.epoch, 0),
                    )
                )
        elif isinstance(message, GenFrame):
            self.summary.gen_frames.append(
                (message.epoch, message.chunk, message.g, message.tick, len(payload))
            )
            self._count_generated(message)
        elif isinstance(message, Cancel):
            self.summary.cancels.append(message.epoch)
            self._requests.clear()
        elif isinstance(message, Hold):
            self._host_held = True
        elif isinstance(message, Release):
            self._host_held = False
        elif isinstance(message, Error):
            self.summary.errors.append(message.to_header())
            if message.code in ("version", "busy", "state_format", "timing"):
                self._stop.set()
        return False

    def _count_generated(self, message: GenFrame) -> None:
        if self._timeline is None:
            return
        key = (message.epoch, message.chunk)
        self._received[key] = self._received.get(key, 0) + 1
        complete = self._complete.get(message.epoch, 0)
        while self._received.get(
            (message.epoch, complete), 0
        ) >= self._timeline.frames_in_chunk(complete):
            complete += 1
        self._complete[message.epoch] = complete

    # ------------------------------------------------------------ simulation

    def _loop_once(self) -> None:
        now = time.monotonic()
        if self._silent_until is not None:
            if now < self._silent_until:
                time.sleep(0.01)
                return
            self._silent_until = None
        busy = self._can_advance()
        self._read(0.0 if busy else 0.02)
        if self._stop.is_set():
            return
        if self._held_until is not None and now >= self._held_until:
            self._held_until = None
            self._wall_t0 += self.script.hold_s
            self._send(Release(reason="player", tick=self.tick))
        self._advance()

    def _can_advance(self) -> bool:
        if self._pending_seed is not None:
            return True
        if self._epoch is None or self._held_until is not None or self._host_held:
            return False
        return any(request.tick > self.tick for request in self._requests.values())

    def _advance(self) -> None:
        if self._pending_seed is not None:
            self._send_seed(self._pending_seed)
            return
        if self._epoch is None or self._held_until is not None or self._host_held:
            return
        self._answer_due()
        target = max(
            (request.tick for request in self._requests.values()), default=None
        )
        if target is None or target <= self.tick:
            return
        if self.script.realtime:
            due = self._wall_t0 + (self.tick + 1 - self._tick_t0) / self.sim_hz
            if time.monotonic() < due:
                return
        self.tick += 1
        if self._run_script():
            return
        self._answer_due()

    def _send_seed(self, request: SeedReq) -> None:
        self._pending_seed = None
        self.tick = max(self.tick, request.tick)
        self._epoch = request.epoch
        self._tick_t0 = self.tick
        self._wall_t0 = time.monotonic()
        width, height = self.seed_size
        image = _seed_image(width, height, request.epoch)
        self._send(
            Frame(
                epoch=request.epoch,
                g=0,
                tick=self.tick,
                seed=True,
                state=dict(self.state_fn(self.tick)),
                encoding="raw",
                width=width,
                height=height,
                t_engine_us=now_us(),
            ),
            image,
        )
        self.summary.seeds.append((request.epoch, self.tick))

    def _answer_due(self) -> None:
        for g in sorted(g for g, r in self._requests.items() if r.tick <= self.tick):
            request = self._requests.pop(g)
            self._send(
                Frame(
                    epoch=request.epoch,
                    g=request.g,
                    tick=request.tick,
                    chunk=request.chunk,
                    index=request.index,
                    req_tick=request.tick,
                    state=dict(self.state_fn(request.tick)),
                    t_engine_us=now_us(),
                )
            )
            self.summary.frames_sent += 1
            limit = self.script.end_after_frames
            if limit is not None and self.summary.frames_sent >= limit:
                self._send(End(reason="done", tick=self.tick))
                self.summary.ended = True
                self._stop.set()
                return

    def _run_script(self) -> bool:
        """Apply the script's action at this tick; return True when it pauses the tick."""
        script = self.script
        if self.tick in self._restart_ticks:
            self._restart_ticks.discard(self.tick)
            self._epoch = None
            self._requests.clear()
            self.summary.restarts.append(self.tick)
            self._send(Restart(reason="reset", tick=self.tick))
            return True
        if script.hold_at_tick == self.tick and "hold" not in self._script_done:
            self._script_done.add("hold")
            self._held_until = time.monotonic() + script.hold_s
            self.summary.holds.append(self.tick)
            self._send(Hold(reason="player", tick=self.tick))
            return True
        if script.drop_at_tick == self.tick and "drop" not in self._script_done:
            self._script_done.add("drop")
            self._close()
            raise _Reconnect()
        if script.silent_at_tick == self.tick and "silent" not in self._script_done:
            self._script_done.add("silent")
            self._silent_until = time.monotonic() + script.silent_s
            return True
        return False


def _seed_image(width: int, height: int, epoch: int) -> bytes:
    xs = np.linspace(0, 255, width, dtype=np.float32)[None, :, None]
    ys = np.linspace(0, 255, height, dtype=np.float32)[:, None, None]
    image = np.concatenate(
        [
            np.broadcast_to(xs, (height, width, 1)),
            np.broadcast_to(ys, (height, width, 1)),
            np.full((height, width, 1), (37 * epoch) % 256, dtype=np.float32),
        ],
        axis=2,
    )
    return image.astype(np.uint8).tobytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the stub external engine against a model host."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--session", default="stub-engine")
    parser.add_argument("--frames", type=int, default=None, help="end after N frames")
    parser.add_argument(
        "--realtime", action="store_true", help="pace ticks to wall time"
    )
    parser.add_argument("--state-jsonl", type=Path, default=None)
    parser.add_argument("--state-format", default=None)
    parser.add_argument("--restart-at-tick", type=int, action="append", default=[])
    parser.add_argument("--hold-at-tick", type=int, default=None)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.state_jsonl and not args.state_format:
        parser.error("--state-jsonl needs --state-format, the states' vocabulary")
    state_fn = jsonl_states(args.state_jsonl) if args.state_jsonl else None
    state_format = args.state_format or FLU_STATE_FORMAT
    stub = StubEngine(
        args.host,
        args.port,
        session=args.session,
        state_fn=state_fn,
        state_format=state_format,
        script=StubScript(
            end_after_frames=args.frames,
            realtime=args.realtime,
            restart_at_ticks=tuple(args.restart_at_tick),
            hold_at_tick=args.hold_at_tick,
            hold_s=args.hold_s,
        ),
    )
    summary = stub.run()
    text = json.dumps(asdict(summary), indent=1)
    if args.summary_json:
        args.summary_json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if not summary.errors else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ScriptedDrive",
    "StubEngine",
    "StubScript",
    "StubSummary",
    "jsonl_states",
    "main",
]
