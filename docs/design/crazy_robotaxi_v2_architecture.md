# Crazy Robotaxi target architecture for FlashDreams V2

This is the implementation guide for the ground-up rewrite. It was derived by
reconciling the current FlashDreams V2 contracts with the behavior recorded in
[`crazy_robotaxi_legacy_architecture.md`](crazy_robotaxi_legacy_architecture.md).
The map branch supplies behavioral details only; its runtime architecture is
not part of this design.

## Design rules

1. FlashDreams V2 owns the runtime loop, event fan-out, reset generation,
   buffering, pacing, client window, and shutdown sequence.
2. There are only two runtime threads: the V2 model thread and the V2 UI
   thread. No game loop, chunk worker, scene-loader worker, MJPEG server, or
   input-device worker is created by the game engine.
3. The model thread exclusively owns mutable simulation, game, conditioning,
   autoregressive-cache, and driving-input state.
4. The UI thread owns all HUD widgets, name-entry state, and presentation.
   Immutable HUD frame batches cross from model to UI through V2
   `invoke_async`; neither thread reads the other's mutable state.
5. The loaded OmniDreams pipeline is application-owned and reusable; every
   session/rollout has its own cache and mutable resources.
6. Reset reconstructs all rollout state together. FlashDreams' generation
   counter discards pre-reset results already in the presentation buffer.
7. Authored maps remain data. Map compilation and preview are CPU tooling, not
   runtime scheduling infrastructure.

## Process ownership

```mermaid
flowchart TD
    HOST["flashdreams-run-v2<br/>ApplicationRunner + client window"]

    subgraph APP["CrazyRobotaxiApplication — process lifetime"]
        ARGS["validated application configuration"]
        PIPELINE["one loaded OmniDreams pipeline<br/>weights + compiled networks"]
    end

    subgraph SESSION["CrazyRobotaxiSession — one run"]
        SCENE["immutable SceneDefinition<br/>map, camera, seed, prompt"]
        REG["register one model thread<br/>register one ImGui UI thread"]
    end

    subgraph MODELTHREAD["FlashDreams V2 model thread"]
        INPUT["DriverInput<br/>held driving keys"]
        ENGINE["GameEngine"]
        CACHE["session-local OmniDreams cache"]
        HUDSTATE["immutable TaxiHudFrame batches"]
    end

    subgraph UITHREAD["FlashDreams V2 UI thread"]
        BUFFER["PresentationManager<br/>generated RGB frames"]
        HUD["CrazyRobotaxiImGUIThread<br/>HUD widgets + name entry"]
        COMPOSE["ImGui compositing"]
        WINDOW["V2 client window<br/>WebRTC / MP4 / supported host"]
    end

    HOST --> APP
    ARGS --> SESSION
    PIPELINE --> SESSION
    SESSION --> MODELTHREAD
    SCENE --> ENGINE
    PIPELINE --> CACHE
    INPUT --> ENGINE
    ENGINE --> HUDSTATE
    ENGINE -->|"generated RGB + optional BEV channels"| BUFFER
    HUDSTATE -->|"invoke_async"| HUD
    HUD -->|"invoke_async game actions"| ENGINE
    BUFFER --> HUD --> COMPOSE --> WINDOW
```

`CrazyRobotaxiApplication.session_desc()` declares the trained output layout,
resolution, and frame rate before model loading. `init()` parses only
application-owned options. `create_session()` validates the requested
description, lazily constructs the shared pipeline, prepares immutable scene
data, and returns an uninitialized session.

The session registers a model thread and a `CrazyRobotaxiImGUIThread`. The
model thread emits generated RGB frames and, when configured, an optional BEV
visualization channel. For each generated frame it also publishes an immutable
`TaxiHudFrame` to the UI thread with `invoke_async`.
The UI thread selects the HUD snapshot aligned with the currently presented
model frame, updates its retained ImGui widgets, and returns the generated
frame as the ImGui back buffer. UI callbacks send validated game actions back
to the model thread with `invoke_async`; they never call game state directly.

## Model-thread step

```mermaid
flowchart LR
    EVENTS["unread V2 input events"]
    REDUCE["DriverInput.reduce"]
    COUNT["pipeline.get_num_frames(step_index)"]
    COMMANDS["per-frame DriverCommand sequence"]
    SIM["SimulationWorld.pose_chunk"]
    RULES["GameRules.advance_frames"]
    SNAPSHOTS["GameFrame snapshots"]
    ACTORS["traffic + passenger actor trajectories"]
    CONDITION["ConditionRenderer.render"]
    TENSOR["HD-map tensor<br/>B,V,T,C,H,W"]
    GENERATE["pipeline.generate"]
    FINALIZE["pipeline.finalize"]
    POST["optional session-local postprocess stream"]
    HUDFRAMES["immutable TaxiHudFrame batch"]
    SEND["invoke_async(UI thread)"]
    RESULTS["list[StepResult]<br/>RGB video + optional BEV"]

    EVENTS --> REDUCE
    COUNT --> COMMANDS
    REDUCE --> COMMANDS
    COMMANDS --> SIM
    SIM --> RULES
    RULES --> SNAPSHOTS
    SIM --> ACTORS
    RULES --> ACTORS
    ACTORS --> CONDITION
    SIM --> CONDITION
    CONDITION --> TENSOR --> GENERATE --> FINALIZE --> POST
    SNAPSHOTS --> HUDFRAMES
    SIM --> HUDFRAMES
    HUDFRAMES --> SEND
    POST --> RESULTS
```

The output-frame count is learned from the pipeline before simulation, so
simulation, conditioning, generated video, game snapshots, and immutable HUD
frames have one authoritative `T`. Input edges update retained held state;
steering and speed commands are advanced at the simulation frame interval to
produce a command for every frame rather than one command per model
invocation.

The direct model contract is:

```text
cache = pipeline.initialize_cache(text, first_frame, view_names)
T = pipeline.get_num_frames(autoregressive_index)
engine_step = engine.step(commands[T])
trajectory = engine_step.trajectory
hdmap = condition_renderer.render(trajectory)       # [B,V,T,3,H,W]
video = pipeline.generate(autoregressive_index, cache, hdmap)
metrics = pipeline.finalize(autoregressive_index, cache)
invoke_async(ui_thread, publish_hud_frames(game_snapshots, poses, video))
return [video StepResult, optional BEV StepResult]
```

The UI thread uses V2's `ImGUIThread` renderer and `presented_model_frames()`;
there is no intermediate render backend, local adapter, model session, chunk
request, private command queue, private frame queue, or legacy
`PresentedFrame` aggregate.

## Game-engine internals

```mermaid
flowchart TD
    MAP["ResolvedGameMap"]
    SCENE["SceneDefinition"]

    subgraph ENGINE["GameEngine — model-thread owned"]
        DRIVER["DriverInput"]
        SIM["SimulationWorld"]
        PHYSX["PhysicsWorld"]
        TRAFFIC["TrafficSystem"]
        GAME["GameRules protocol"]
        COND["ConditionRenderer"]
        OUTPUT["EngineStep<br/>trajectory + game frames + condition frames"]
    end

    MAP --> SCENE
    SCENE --> SIM
    SCENE --> PHYSX
    SCENE --> TRAFFIC
    SCENE --> COND
    DRIVER --> SIM
    SIM <--> PHYSX
    TRAFFIC <--> PHYSX
    SIM --> GAME
    GAME -->|"game actors and immutable per-frame state"| COND
    SIM --> COND
    GAME --> OUTPUT
    COND --> OUTPUT
```

The reusable engine defines narrow game-facing contracts:

- `GameRules` owns mutable rules state, advances it from a trajectory,
  contributes dynamic actors, and returns immutable per-frame snapshots.
- `VehicleDynamics` turns commands into proposed vehicle motion.
- `PhysicsWorld` resolves ego, static, and actor interactions and produces
  synchronized actor trajectories.
- `ConditionRenderer` converts a scene and `EngineStep` into model-ready HD-map
  frames plus optional BEV data used by the HUD.
- `GameEngine` sequences these components; it knows nothing about taxi fares,
  leaderboards, application entry points, V2 windows, or model presets.

The initial implementation may use concrete engine classes where no second
implementation exists, but dependencies still point from generic engine code
to game-injected protocols, never from the engine into `crazy_robotaxi`.

## Crazy Robotaxi specialization

```mermaid
flowchart LR
    MAP["node-graph city map"]
    NAV["TaxiNavigation<br/>lanes + shortest routes"]
    FARES["TaxiGameRules<br/>pickup/dropoff, timers, score"]
    TAXI["ArcadeTaxiDynamics"]
    TAXIPHYS["TaxiPhysicsPolicy<br/>curb response + chassis tuning"]
    TRAFFIC["authored TrafficSystem"]
    PASSENGERS["waiting-passenger actors"]
    SCORES["HighScoreStore"]
    FRAME["TaxiGameFrame"]
    HUDSTATE["TaxiHudFrame<br/>immutable presentation state"]
    IMGUI["CrazyRobotaxiImGUIThread<br/>score, time, navigation,<br/>name entry, leaderboard"]

    MAP --> NAV --> FARES
    TAXI --> TAXIPHYS
    TRAFFIC --> TAXIPHYS
    FARES --> PASSENGERS
    FARES <--> SCORES
    FARES --> FRAME --> HUDSTATE --> IMGUI
    PASSENGERS --> FRAME
```

Taxi-specific code supplies the generic engine with arcade vehicle dynamics,
curb/chassis physics policy, map-derived navigation/fare regions, game rules,
passenger actors, persistent scores, and UI-thread ImGui widgets. It does not
subclass a game-engine application host or presenter.

## State and lifetime table

| Lifetime / owner | State |
| --- | --- |
| Application | resolved pipeline configuration; one loaded OmniDreams pipeline |
| Session, immutable | validated session description; compiled/loaded scene definition; game configuration |
| Model thread, per reset generation | driving-input reducer; simulation/PhysX/traffic; game rules; condition renderer; OmniDreams cache; last generated frame; AR index |
| UI thread, per reset generation | retained ImGui widgets; immutable HUD-frame lookup; name-entry buffer; validation messages |
| Outside runtime | authored map YAML; derived map cache; high-score file |

All simulation, condition-renderer, and world-model CUDA calls that mutate
per-rollout state occur on the model thread. The ImGui renderer creates and
uses its presentation resources only on the UI thread. Immutable scene arrays
may be prepared before thread startup, but the model-thread state factory
performs mutable condition-renderer, PhysX, game, and cache construction
lazily on its first `step`.

## Reset, terminal state, and close

```mermaid
stateDiagram-v2
    [*] --> Playing
    Playing --> Playing: model step
    Playing --> AwaitingName: timer ends and score qualifies
    Playing --> Leaderboard: timer ends without qualification
    AwaitingName --> Leaderboard: name submitted
    Leaderboard --> [*]: terminal output presented
    Playing --> Playing: V2 reset generation rebuilds rollout
    AwaitingName --> Playing: V2 reset generation rebuilds rollout
    Leaderboard --> Playing: V2 reset generation rebuilds rollout
```

While awaiting a name, no new world-model block is generated. The model thread
re-emits the last generated RGB frame and publishes an updated immutable HUD
frame. ImGui owns the editable name buffer and submits a validated name to the
model thread with `invoke_async`. The model thread reports `is_finished()` only
after a leaderboard frame has been emitted (or an explicit finite block limit
is reached), allowing MP4 mode to terminate without a client close event.

On reset, the model thread closes and recreates the simulation, traffic, rules,
condition renderer state, and autoregressive cache as one unit. On close, it
releases all session-local resources; application close releases the shared
pipeline.

## Legacy-to-V2 reconciliation

| Map-branch responsibility | V2 owner / replacement |
| --- | --- |
| `InteractiveDriveApp` lifecycle | `IApplication` + `ISession` |
| custom `run_main_loop` | `run_session` supplied by FlashDreams V2 |
| `ChunkPipeline` worker | V2 model thread |
| private command/frame queues | direct step call + `PresentationManager` |
| generation token on reset | V2 `EventBuffer` generation |
| presenter event polling | V2 client window and shared event buffer |
| presenter pacing and frame replay/drop | V2 UI thread + presentation mode |
| `FlashdreamsWorldModelSession` | direct OmniDreams pipeline cache API |
| `PresentedFrame` metadata bundle | typed engine results plus immutable `TaxiHudFrame` messages |
| keyboard telemetry shared with presenter | UI-owned ImGui input plus `invoke_async` game actions |
| native/MJPEG presenters | V2 client-window modes |
| application-internal scene switching | map/variant application arguments; a different scene starts a new session |
| manual reset queueing | V2 reset event and thread `reset()` |

## Implemented package boundary

```text
apps/omnidreams_game_engine/
  omnidreams_game_engine/
    engine.py                 # GameEngine and EngineStep
    contracts.py              # injected game/dynamics/physics contracts
    model.py                  # direct OmniDreams rollout bridge
    scene.py                  # immutable runtime scene loading
    input.py                  # V2 event reduction to per-frame commands
    conditioning.py           # Ludus conditioning boundary
    simulation/               # generic kinematics, PhysX, traffic
    game_map/                 # authored-map schema/compiler/runtime types

apps/crazy_robotaxi/
  crazy_robotaxi/
    application.py            # IApplication composition root
    session.py                # ISession + model/UI thread wiring
    ui.py                     # ImGui HUD state and UI thread
    rules.py                  # taxi fare state machine
    dynamics.py               # arcade taxi controls
    physics.py                # taxi collision policy
    navigation.py             # map-derived routes/fare sampling
    passengers.py             # conditioning actors
    high_scores.py            # persistent leaderboard
    config.py                 # strict game configuration
    map_tool.py               # offline validate/compile/preview command
```

Runtime responsibilities must remain in the blocks shown here. In particular,
the implementation must not introduce a second application host, custom
runtime loop, worker queue, or presenter abstraction.
