# OmniDreams Game Engine

Reusable model-thread simulation, authored-map, physics, and conditioning
components for FlashDreams V2 applications. [Crazy Robotaxi](../crazy_robotaxi/README.md)
uses this package for vehicle simulation, traffic, map rendering, and its
world-model rollout.

## Responsibilities and public API

The package exports `GameEngine`, `EngineStep`, and `WorldModelRollout`:

| Component | Responsibility |
| --- | --- |
| `GameEngine` | Advance vehicle simulation, apply application game rules, then render model conditioning for one block of driver commands. |
| `EngineStep` | Return the trajectory, game-state snapshots, conditioning frames, and engine timing metrics for that block. |
| `WorldModelRollout` | Combine an engine with an already constructed pipeline, generate video, finalize the autoregressive cache, and return video with its matching engine data. |

Applications supply implementations of the `SimulationWorld`, `GameRules`, and
`ConditionRenderer` protocols in [contracts.py](omnidreams_game_engine/contracts.py).
The engine checks that game snapshots and conditioning frames match the number
of driver commands. The rollout uses the pipeline's output-frame count and
checks that generated video matches the engine block.

`WorldModelRollout.reset()` recreates the engine and autoregressive cache while
retaining the pipeline and its model weights. `close()` releases session-local
engine resources. The application owns pipeline construction and lifetime;
the [FlashDreams V2 runtime](../../flashdreams/flashdreams/runtime_v2/README.md)
owns the model and UI loops, threads, client windows, and presentation.

## Shared implementation

| Module | Provides |
| --- | --- |
| `game_map/` | Strict map loading, road-graph resolution, scene compilation, traffic routes, SVG previews, and first-frame spawn previews. |
| `scene_loader.py` | Load compiled scene archives, camera calibration, initial images, prompts, and geometry. |
| `simulation/` | Vehicle kinematics, PhysX integration, ground snapping, traffic control, and simulation components. |
| `conditioning.py` | `LudusConditionRenderer`, including main-camera HD-map conditioning and a top-down view. |
| `input.py` | `DriverInput`, which converts input events to normalized driver commands. |
| `config.py`, `types.py` | Shared rendering, vehicle, scene, trajectory, and command data types. |

## Authored maps

Author node-graph YAML maps using the
[node-graph map format](NODE_GRAPH_MAP_FORMAT.md). Maps define road geometry,
traffic, spawns, and optional race courses. Compilation produces a scene archive
for a selected spawn and caches it under
`$FLASHDREAMS_CACHE_DIR/omnidreams-game-engine/game-maps`, or
`~/.cache/flashdreams/omnidreams-game-engine/game-maps` by default.

Spawn images can be map-relative files or `package://package/resource`
references. If a spawn omits its image, the compiler generates a deterministic
first-person road preview as the initial image. Map validation, compilation,
and previews run without a world model or GPU. The `crazy-robotaxi-map` command
belongs to the Crazy Robotaxi package; see its
[map commands](../crazy_robotaxi/README.md#authored-maps) for runnable examples.

## Settings schemas

Applications choose their own settings tree and how to expose it in menus.
Crazy Robotaxi uses `crazy_robotaxi.settings.SettingsDocument` and its
[`config.yaml` / Options schema](../crazy_robotaxi/README.md#options-and-user-configuration). That tree includes
app-owned settings alongside shared game-engine, OmniDreams, and FlashDreams
settings. Its keyboard, gamepad, and wheel bindings use separate controls files.

This package also provides the independent `EngineSettings` /
`load_engine_settings()` API in
[engine_settings.py](omnidreams_game_engine/engine_settings.py). Its root fields
are `map`, `world_model`, `rendering`, `presentation`, `wheel`, and `runtime`.
It accepts `schema_version: 1` and strict partial YAML overlays on typed defaults
or a supplied base. Unknown fields are rejected, and relative paths resolve
beside the YAML document. Crazy Robotaxi's `--config` loads its own schema;
an `EngineSettings` document cannot be used as that file.
