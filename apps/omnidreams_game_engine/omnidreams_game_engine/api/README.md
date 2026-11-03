# External-engine API

`omnidreams_game_engine.api.v0` connects a game engine that runs outside
FlashDreams (such as an Unreal build) to a world model the same way
an in-process engine such as Crazy Robotaxi's is connected: through
`WorldModelRollout` and its `RolloutEngine` seam. Its names are stable within
version 0 ([STABILITY.md](STABILITY.md)); the wire protocol is
[v0/PROTOCOL.md](v0/PROTOCOL.md).

## Pieces

| Module | Provides |
|---|---|
| `v0` (package) | The stable names, including the existing contract re-exported unchanged: `WorldModelRollout`, `RolloutEngine`, `GameEngine`, `EngineStep`, `GameRules`, `GameUpdate`, `SimulationWorld`, `ConditionRenderer`, `SceneDefinition`, `TrajectoryChunk`, `DynamicActorTrajectory`, `VehicleState`, `DriverCommand`, `ConditionBatch`, `CameraCalibration` |
| `messages`, `framing` | Wire message types, parsing with unknown fields kept, version check, framing; `schema.json` |
| `timeline` | `ChunkTimeline`: ticks, model frames, chunks (plan from the pipeline), grants, timestamps |
| `state`, `units` | `FrameState` (decoded per-frame state in FLU), `StateDecoder`, the `flu/0` vocabulary; Unreal-to-FLU conversions for engine-specific decoders |
| `assembly` | `ChunkAssembler` and `ExternalChunk`: per-frame states into chunks and into a `TrajectoryChunk`, actor tracks split at gaps |
| `external` | `ExternalEngineSession`, the `RolloutEngine` of an external engine; `EngineSource`, the interface of every state source |
| `driver` | `ExternalSessionDriver`: the host's session flow (config, seed, grants, generation, frame return, restart, hold, end, reconnect) |
| `link_tcp` | `TcpEngineLink`: the live engine over TCP, with pings, silence detection and reconnects |
| `recorder`, `replay` | `PacketRecorder`, `PacketLogReader`, `PacketLogSource`: record a session and replay it without the engine |
| `stub_engine` | A Python external engine for tests and host bring-up |
| `testing` | CPU stand-ins (pipeline, condition renderer, scene provider); not part of the stable API |

Threading: a transport's threads only move bytes and queue messages. Parsing,
state decoding, chunk assembly, events, the rollout and the condition renderer
run on the thread that calls `ExternalSessionDriver.poll`, the model thread.

## Host

```python
from omnidreams_game_engine.api import v0

link = v0.TcpEngineLink(decoder, port=8095,  # listens on 127.0.0.1
                        recorder=v0.PacketRecorder(session_dir))
driver = v0.ExternalSessionDriver(
    source=link,
    pipeline=pipeline,                  # an OmniDreams pipeline, loaded once
    scene_provider=scenes,              # map + seed -> SceneDefinition
    condition_renderer=renderer,        # e.g. LudusConditionRenderer, shared across epochs
    encoder=jpeg_encoder,               # gen_frame payloads; RawRgbEncoder by default
    config=v0.DriverConfig(lead_chunks=2, return_width=1024, return_height=560),
)
while running:                          # on the model thread
    step = driver.poll(timeout_s=0.1)   # None while waiting; a DriverStep per chunk
```

`TcpEngineLink` listens on `127.0.0.1` unless `host=` names another address.
The protocol has no authentication: to accept an engine on another machine,
pass `host=` (for example `"0.0.0.0"`), and do so only on a trusted network.

`decoder` reads the engine's `frame.state` vocabulary (`v0.FluStateCodec()`
for `flu/0`). A model host supplies the decoder, the scene provider and the
encoder for its engine; everything else is here. `driver.request_refresh()`
re-anchors on a new seed; `driver.forward_input(action, phase)` forwards key
edges from a host-side view.

Replay a recorded session without the engine:

```python
source = v0.PacketLogSource(session_dir, decoder)
driver = v0.ExternalSessionDriver(source=source, ...)
while not source.exhausted:
    driver.poll(0.0)
```

## Engine

Implement [v0/PROTOCOL.md](v0/PROTOCOL.md); `stub_engine.py` is a complete
reference in Python. To try a host without a game:

```bash
python -m omnidreams_game_engine.api.v0.stub_engine --host HOST --port 8095 --frames 600
```

`--realtime` paces the stub at 60 ticks per second, `--restart-at-tick` and
`--hold-at-tick` exercise restarts and pauses, `--state-jsonl` with
`--state-format` sends recorded states in an engine's own vocabulary (for a
host with a matching decoder).

## In-process engines

An in-process engine keeps implementing `SimulationWorld`, `GameRules` and
`ConditionRenderer` and runs in `GameEngine`, imported from `api.v0`. Crazy
Robotaxi does so. `GameEngine` and `ExternalEngineSession` are both
`RolloutEngine`s that produce a `TrajectoryChunk` for the same condition
renderer and rollout; given the same states they produce the same model input
(`tests/test_api_external_session.py`).

## Needs of an external engine

| Need | Version 0 |
|---|---|
| Session setup: protocol version, engine identity, map identity, start pose and seed image | `hello` (`api`, `session`, `state_format`, `layout_crc32`, `level`, `front_mount`, `wide_seed`), `config`, `start`, `seed_req` and the seed `frame` |
| Per-chunk state: per-frame pose, vehicle state, applied controls, actors, events | `frame.state` decoded by a `StateDecoder` to `FrameState`; `ChunkAssembler` builds the chunk and its `TrajectoryChunk` |
| Tick pacing: a fixed simulation rate, the engine never ahead of what the host can generate | `frame_req` grants, `lead_chunks` ahead of generation; `ChunkTimeline` maps ticks, frames and chunks |
| Frame return: the generated frames, each tied to the tick that produced it | `gen_frame` with `epoch`, `chunk`, `g`, `tick`; `display_delay_ticks` in `config` |
| Pause and resume without losing the model state | `hold`, `release` in both directions |
| Restart: reset, respawn, teleport, drift refresh | `restart`, `cancel`, a new epoch and seed; `WorldModelRollout.reset(scene=...)` keeps the pipeline |
| Reconnect after a network failure | the same `session` reconnects, `start{reason: "reconnect"}`, a new epoch; silence detection on the host |
| Recording: packet log of both directions with stamps | `PacketRecorder`, `PacketLogReader`; replay without the engine with `PacketLogSource` |
| Recording: player input and per-tick state hash | not in version 0; to be added as optional `frame` fields within version 0 |
| Key input from a host-side view | `input` |
