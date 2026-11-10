<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# External game engines

`omnidreams_game_engine.api.v0` connects a game engine that runs outside
FlashDreams, such as an Unreal build, to a world model the way an in-process
engine is connected: through `WorldModelRollout` and its `RolloutEngine`
interface. The engine sends its state for every model frame over the
[external-engine protocol](EXTERNAL_ENGINE_PROTOCOL.md); the
host draws the condition from it, generates the video and sends the frames
back. An in-process engine keeps implementing `SimulationWorld`, `GameRules`
and `ConditionRenderer` and runs in `GameEngine`; given the same states, both
engines produce the same model input.

## Host

```python
from omnidreams_game_engine.api import v0

link = v0.TcpEngineLink(decoder, port=8095, recorder=v0.PacketRecorder(session_dir))
driver = v0.ExternalSessionDriver(
    source=link,
    pipeline=pipeline,            # loaded once, kept across sessions
    scene_provider=scenes,        # the map plus the seed -> SceneDefinition
    condition_renderer=renderer,  # shared across epochs
    encoder=jpeg_encoder,         # gen_frame payloads; RawRgbEncoder by default
    config=v0.DriverConfig(return_width=1024, return_height=560),
)
while running:  # on the model thread
    step = driver.poll(timeout_s=0.1)  # a DriverStep per generated chunk, else None
```

The host supplies the pipeline, the condition renderer, the state decoder for
the engine's `frame.state` vocabulary (`v0.FluStateCodec()` for `flu/0`) and
the scene provider, and optionally an encoder. The link's threads only move
bytes; parsing, chunk assembly, the rollout and the condition renderer run on
the thread that calls `poll`.

`TcpEngineLink` listens on `127.0.0.1`. The protocol has no authentication,
so pass `host=` to accept an engine on another machine only on a trusted
network.

When the model fails to start an epoch or to generate a chunk, `poll` sends
`restart`, asks the engine for a new seed and raises, so a host can log the
error and keep polling.
`driver.request_refresh()` re-anchors on a new seed at the engine's current
pose, and `driver.forward_input(action, phase)` forwards a key edge from a
host-side view.

## Replay

`PacketRecorder` writes every message of a session. `PacketLogSource` replays
the recording into a host with the live session's `DriverConfig`, without the
engine, and the host generates the same chunks from the same states:

```python
source = v0.PacketLogSource(session_dir, decoder)
driver = v0.ExternalSessionDriver(source=source, ...)
while (step := driver.poll(0.0)) is not None or not source.exhausted:
    ...
```

A recording with a restart the host started (`request_refresh()` or a model
failure) is refused, because a replay cannot reproduce it.

## Engine bring-up

Implement the [protocol](EXTERNAL_ENGINE_PROTOCOL.md). The
stub engine in `omnidreams_game_engine.api.v0.stub_engine` is a complete
reference in Python and drives a host without a game:

```bash
uv run python -m omnidreams_game_engine.api.v0.stub_engine --host HOST --port 8095 --frames 600
```

`--realtime` paces the stub at 60 ticks per second, `--restart-at-tick` and
`--hold-at-tick` exercise restarts and pauses, and `--state-jsonl` with
`--state-format` sends recorded states in an engine's own vocabulary.

## Stability

Within version 0, the names in `api.v0.__all__` (for its modules `messages`,
`framing` and `units`, the names in their `__all__`) and the messages and
fields of the protocol keep their meaning, signatures and behavior. Additions
are allowed: optional fields, message types, optional keyword arguments and
names. Receivers ignore fields and message types they do not know, so an
older peer keeps working.

A breaking change adds `api.v1` with protocol `omnidreams-engine/1` beside
`api.v0`, which stays for at least one tagged release and is announced in the
release notes. A host refuses a `hello` whose `api` it does not serve.

Everything else is implementation: `api.v0.testing`, `api.v0.stub_engine`,
the package's modules outside `api`, log messages and metric names. In-tree
applications may use them and change with them.
