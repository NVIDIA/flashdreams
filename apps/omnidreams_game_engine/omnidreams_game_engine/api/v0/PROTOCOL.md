# External-engine protocol `omnidreams-engine/0`

The wire protocol between a game engine (such as an Unreal build)
and a model host that renders the engine's state with a world model. The
message classes are in [`messages.py`](messages.py); [`schema.json`](schema.json)
is a JSON Schema of every header, generated from them.

## Framing and connection

- Every message is a 4-byte big-endian header length, a UTF-8 JSON object
  with a string `type` and an integer `bytes`, then `bytes` bytes of binary
  payload (0 when absent).
- Limits: header at most 1 MiB, payload at most 16 MiB (`framing.py`).
- One TCP connection, opened by the engine. The host listens and serves one
  engine at a time; a different engine session is refused with
  `error{code: "busy"}`. The same session name connecting again replaces its
  old connection (a reconnect).
- The protocol has no authentication and no encryption. `TcpEngineLink`
  listens on `127.0.0.1` by default; a host that accepts an engine on another
  machine passes `host=` explicitly and does so only on a trusted network.
- Times on the wire are microseconds since the Unix epoch on the sender's
  clock (`t_host_us`, `t_engine_us`). The host estimates the offset from
  `ping`/`pong`.
- Receivers ignore header fields and message types they do not know.

## Messages

Engine to host:

| Type | Required fields | Optional fields | Meaning |
|---|---|---|---|
| `hello` | `session` | `api` (`omnidreams-engine/0`), `state_format`, `level`, `layout_crc32`, `driver` (`human`, `autopilot`), `sim_hz` (60), `tick`, `front_mount`, `wide_seed`, `engine_build`, `t_engine_us` | First message of every connection |
| `start` | `tick` | `reason` (`start`, `reconnect`) | A driving session begins, or resumes after a reconnect |
| `frame` | `epoch`, `g`, `tick` | `state`, `chunk`, `index`, `req_tick`, `seed`, `encoding`, `width`, `height`, `t_engine_us` | The state of one granted model frame; with `seed: true` the payload is the seed image |
| `pong` | `t_host_us` | `t_engine_us` | Answer to `ping` |
| `views` | | `views` (list), `t_engine_us` | The host views the engine shows now (API 0.2); the host sends their frames while they are listed |

Host to engine:

| Type | Required fields | Optional fields | Meaning |
|---|---|---|---|
| `config` | `session`, `api`, `model_fps`, `frame_ticks`, `chunk_plan`, `lead_chunks`, `display_delay_ticks`, `return_encoding` | `return_width`, `return_height`, `return_quality`, `render` (`off`), `checkpoint`, `profile`, `preset`, `host_build` | Answer to `hello` |
| `seed_req` | `epoch`, `tick` | `reason`, `width`, `height`, `fov_deg`, `warmup_renders` | Ask for the seed image and state that start an epoch |
| `frame_req` | `epoch`, `chunk`, `index`, `g`, `tick` | `condition`, `keys` | Grant: the engine may simulate up to `tick` and answers with `frame` |
| `gen_frame` | `epoch`, `chunk`, `g`, `tick`, `encoding`, `width`, `height` | `t_host_us`, `camera` | One generated frame; the payload is the image |
| `view_frame` | `view`, `epoch`, `chunk`, `g`, `tick`, `encoding`, `width`, `height` | `t_host_us` | One frame of a host view the engine listed in `views` (API 0.2); the payload is the image |
| `input` | `action`, `phase` | `t_host_us` | A key edge from a host-side view |
| `cancel` | | `epoch` | Drop the outstanding grants |
| `ping` | `t_host_us` | | Liveness and clock offset; every 0.5 s by default |

Both directions:

| Type | Required fields | Optional fields | Meaning |
|---|---|---|---|
| `hold` | | `reason`, `tick` | Engine: the player paused. Host: the engine must stop advancing |
| `release` | | `reason`, `tick` | Ends a `hold`; the same epoch continues |
| `restart` | | `reason`, `tick`, `epoch` | A discontinuity; the host starts a new epoch with a new seed |
| `end` | | `reason`, `tick` | The driving session ends; the connection and the host process stay |
| `status` | | `tick`, anything else | Periodic status, free-form |
| `error` | `code` | `message` | Refusal or failure: `version`, `state_format`, `busy`, `timing`, `protocol` |

## Ticks, frames and chunks

- The engine simulates at a fixed `sim_hz` (60). The model generates
  `model_fps` (30) frames per second, so a model frame is
  `frame_ticks = sim_hz / model_fps` (2) engine ticks.
- An epoch starts at its seed tick `t0`. Model frame `g` belongs to tick
  `t0 + frame_ticks * g`; frame 0 is the seed and carries the seed's state.
- Chunk `k` holds the frames of autoregressive step `k`. `chunk_plan` lists
  the frames per chunk, the last entry repeating: `[5, 8]` for OmniDreams
  (chunk 0 includes the seed). The host takes the plan from its pipeline.
- Model timestamps: frame `g` is at `base + g * round(1e6 / model_fps)` µs,
  the convention of the in-process engine.

Worked example, `t0` = seed tick, plan `[5, 8]`, `lead_chunks = 2`:

| Chunk k | Frames g | Ticks | Sealed at tick | Granted |
|---|---|---|---|---|
| 0 | 0-4 | t0 .. t0+8 | t0+8 | after the seed, with chunks 1 and 2 |
| 1 | 5-12 | t0+10 .. t0+24 | t0+24 | after the seed |
| 2 | 13-20 | t0+26 .. t0+40 | t0+40 | after the seed |
| 3 | 21-28 | t0+42 .. t0+56 | t0+56 | when the host starts chunk 1 |
| k >= 1 | 5+8(k-1) .. 12+8(k-1) | t0+10+16(k-1) .. t0+24+16(k-1) | t0+24+16(k-1) | when the host starts chunk k-2 |

## Session flow

1. **Connect.** The engine sends `hello`. The host checks `api` (major
   version 0) and `state_format` (its state decoder must read it) and answers
   with `config`, or with `error` and closes the connection. A `hello`
   without `api` is refused with `version`.
2. **Start.** The engine sends `start{tick}`. The host opens epoch `e`
   (epochs count up from 1 for the life of the host) and sends
   `seed_req{epoch: e, tick}`.
3. **Seed.** The engine answers `frame{epoch: e, g: 0, seed: true}` with its
   state and the seed image (`encoding`, `width`, `height`). An engine that has
   already passed the requested tick answers at its current tick; the reply's
   `tick` is `t0`. The host builds the scene from the seed, initializes the
   model cache, and grants chunks 0 to `lead_chunks`.
4. **Steady state.** For every `frame_req` the engine answers one `frame`
   with the state of exactly that tick, sent at the end of the tick, and never
   simulates past the newest granted tick. When all frames of chunk `k` are
   in, the host grants up to the end of chunk `k + lead_chunks`, generates
   chunk `k` and sends its frames as `gen_frame`, in order, each naming the
   `g` and `tick` whose state produced it. The engine ignores `frame_req`,
   `cancel` and `gen_frame` of an epoch other than its current one.
5. **Pause.** The engine sends `hold` and stops ticking; `release` resumes the
   same epoch. The host keeps its model cache.
6. **Restart.** The engine sends `restart{tick}` (reset, respawn, teleport,
   player restart), or the host sends `restart` itself (drift refresh). The
   host sends `cancel{epoch: e}` and then `seed_req{epoch: e + 1}`; the flow
   continues at step 3.
7. **Reconnect.** After a lost connection the engine connects again with the
   same `session`, sends `hello` and `start{reason: "reconnect"}`; a new epoch
   with a new seed follows. Resuming the old epoch is not part of version 0.
8. **End.** The engine sends `end` (back to the title screen) or closes the
   connection. The host closes the session and keeps the model loaded; a
   later `start` on the same connection opens a new epoch.

**Host views (API 0.2, optional).** An engine that shows a host view sends
`views{views: [...]}` with the views it shows, and again with an empty list
when it stops; a new connection starts with none. The host serves `map`: after
the `gen_frame` messages of a chunk it sends one `view_frame{view: "map"}` per
generated frame, with the same `epoch`, `g` and `tick`, whose image is the
conditioning frame (the world-scenario map) the model generated that frame
from, encoded like the generated frames. The engine can show both at the same
display tick. A host ignores views it does not serve, and an engine that never
sends `views` never receives `view_frame`.

The host drops a connection on which the engine has said nothing for 10 s
(the engine answers every `ping`). An engine should half-close (shut down its
sending side and read until the host closes) after `end`, so that `end` is not
lost to a TCP reset while frames are still in flight.

## State vocabularies

`frame.state` is in the vocabulary named by `hello.state_format`; the host's
`StateDecoder` turns it into FLU metres (`state.py`). Version 0 defines one
engine-neutral vocabulary, `flu/0` (`FluStateCodec`): the rig pose as 16
row-major floats, every `VehicleState` field, every `DriverCommand` field,
actor boxes (`id`, `type`, `position`, `orientation_xyzw`, `size_lwh`) and
events. Engine-specific vocabularies, such as an engine's own state line in Unreal
units, are decoded by their model host with the conversions in `units.py`.

## Recording

`PacketRecorder` writes every message of a session in both directions into a
folder: `packets.jsonl` (a `meta` line, then one line per message with its
direction, stamps and header) and `packets.bin` (payloads; engine payloads
always, host payloads on request). Engine messages are written when the host
consumes them, with `host_sends`, the number of messages the host had sent by
then. `PacketLogSource` replays a recording into a host without the engine and
delivers each message at the same point, so a replay generates the same chunks
of the same epochs from the same states.
