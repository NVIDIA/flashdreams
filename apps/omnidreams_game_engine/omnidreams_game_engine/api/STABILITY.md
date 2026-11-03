# API stability

`omnidreams_game_engine.api` holds the versioned public API of the game-engine
layer. Version 0 is `omnidreams_game_engine.api.v0` with the wire protocol
`omnidreams-engine/0`.

## Within a version

- Every name in `api.v0.__all__`, and the public classes and functions of the
  `api.v0` modules other than `testing`, keep their meaning, signatures and
  behavior.
- Every message type and field of [`v0/PROTOCOL.md`](v0/PROTOCOL.md) keeps
  its name and meaning.
- Additions are allowed: new optional fields with defaults, new message types,
  new optional keyword arguments, new names. Receivers ignore fields and
  message types they do not know, so an older peer keeps working.
- Nothing is removed, renamed or changed in meaning. `API_VERSION` (`0.1`)
  counts additions in its minor number.

## Breaking changes

- A breaking change creates `api.v1` beside `api.v0`, with protocol
  `omnidreams-engine/1`. A host refuses a `hello` whose `api` it does not
  serve with `error{code: "version"}`.
- `api.v0` stays for at least one tagged release after `api.v1` lands.
- The change is announced in this package's README and in the FlashDreams
  release notes.

## Not part of the API

- The modules outside `api/`: `engine.py`, `model.py`, `contracts.py` and
  `types.py` are reached through the re-exports in `api.v0`; their other
  contents, `conditioning.py`, `simulation/`, `game_map/` and the rest are
  implementation. In-tree applications such as Crazy Robotaxi may use them
  and move with them.
- `api.v0.testing` (CPU stand-ins for tests).
- Log messages, metric names in `EngineStep.metrics`, and the stub engine's
  scripted drive.
