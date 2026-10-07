# LIVE EDIT

[Options index](../OPTIONS.md)

All abilities start disabled. Enabling style, weather, map context, or guided obstacles requires Python transformer hooks and disables native DiT acceleration when it was selected. Enabling mystery items also enables style; rain or snow items also enable weather. Weather cannot change while a non-base style is active.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Sharpen Amount:** | Crazy Robotaxi | — | Unsharp-mask strength on styled frames; `0` disables sharpening. |
| **Sharpen Sigma:** | Crazy Robotaxi | — | Gaussian blur radius used by the unsharp mask. |
| **Perf Log Every Frames:** | Crazy Robotaxi | `--live-edit-perf-log` | Interval for live-edit CPU/GPU cost reports; `0` disables them. |

## STYLE

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-style`, `--no-live-edit-style` | Allows live skin switching. |
| **Lora Checkpoint:** | Crazy Robotaxi | `--live-edit-style-lora` | Pre-merged text-edit LoRA used for skin swaps. |
| **Corrector Checkpoint:** | Crazy Robotaxi | `--live-edit-style-corrector` | Checkpoint for the style drift corrector. |
| **Corrector Gain:** | Crazy Robotaxi | `--live-edit-style-gain` | Strength of the style drift corrector. |
| **Corrector Mode:** | Crazy Robotaxi | `--live-edit-corrector-mode` | `fused`, `unfused`, or `off` deployment of the corrector. |
| **Base Corrector Checkpoint:** | Crazy Robotaxi | `--live-edit-base-corrector` | Optional corrector checkpoint for the base visual state. |
| **Base Corrector Gain:** | Crazy Robotaxi | `--live-edit-base-corrector-gain` | Corrector strength for the base visual state. |
| **Gate Alpha Json:** | Crazy Robotaxi | `--live-edit-gate-alpha-json` | Per-timestep corrector gate profile. |
| **Guidance Scale:** | Crazy Robotaxi | — | Guidance strength for a skin change. |
| **Guidance Chunks:** | Crazy Robotaxi | `--live-edit-skin-guidance-chunks` | Number of chunks in the skin-change guidance window. |
| **Reswap Interval Chunks:** | Crazy Robotaxi | `--live-edit-style-reswap-chunks` | Reapplies the active skin every N chunks; `0` disables refresh. |
| **Skins:** | Crazy Robotaxi | — | List of selectable skins, each with a name and prompt. `--live-edit-skin-first NAME` only rotates the existing list. |

## COINS

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-coins`, `--no-live-edit-coins` | Places collectible coins along lanes. |
| **Spacing M:** | Crazy Robotaxi | — | Distance between coin groups along the road. |
| **Group Offsets M:** | Crazy Robotaxi | — | Lateral positions of coins within a group. |
| **Hover Height M:** | Crazy Robotaxi | — | Height of coin centers above the road. |
| **Coin Diameter M:** | Crazy Robotaxi | — | World-space coin diameter used for rendering. |
| **Pickup Radius M:** | Crazy Robotaxi | — | Distance at which the taxi collects a coin. |
| **Max Render Distance M:** | Crazy Robotaxi | — | Coin visibility limit. |
| **Fade Start Distance M:** | Crazy Robotaxi | — | Distance where coins begin fading out. |
| **Max Visible Sprites:** | Crazy Robotaxi | `--live-edit-coin-max-visible` | Per-frame cap, keeping nearest coins; `0` removes the cap. |
| **Sprite Path:** | Crazy Robotaxi | `--live-edit-coin-sprite` | Optional RGBA coin image; blank uses the procedural coin. |

## ITEMS

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-items`, `--no-live-edit-items` | Places collectible effect items along lanes. |
| **Spacing M:** | Crazy Robotaxi | `--live-edit-item-spacing` | Distance between effect items along the road. |
| **Hover Height M:** | Crazy Robotaxi | — | Height of effect items above the road. |
| **Item Diameter M:** | Crazy Robotaxi | — | World-space item size used for rendering. |
| **Pickup Radius M:** | Crazy Robotaxi | — | Distance at which the taxi collects an effect item. |
| **Max Render Distance M:** | Crazy Robotaxi | — | Item visibility limit. |
| **Fade Start Distance M:** | Crazy Robotaxi | — | Distance where items begin fading out. |
| **Rain Sprite Path:** | Crazy Robotaxi | `--live-edit-item-rain-sprite` | Optional RGBA artwork for each item; blank uses a procedural placeholder. |
| **Snow Sprite Path:** | Crazy Robotaxi | `--live-edit-item-snow-sprite` | Optional RGBA artwork for each item; blank uses a procedural placeholder. |
| **Mystery Sprite Path:** | Crazy Robotaxi | `--live-edit-item-mystery-sprite` | Optional RGBA artwork for each item; blank uses a procedural placeholder. |
| **Nitro Sprite Path:** | Crazy Robotaxi | `--live-edit-item-nitro-sprite` | Optional RGBA artwork for each item; blank uses a procedural placeholder. |
| **Item Types:** | Crazy Robotaxi | `--live-edit-item-types` | Mix of `rain`, `snow`, `mystery`, and `nitro` items placed on the course. |
| **Nitro Boost:** | Crazy Robotaxi | `--live-edit-nitro-boost` | Multiplier applied to maximum speed while nitro is active. |
| **Nitro Duration S:** | Crazy Robotaxi | `--live-edit-nitro-duration-s` | Duration of a nitro boost in simulation time. |
| **Nitro Max Speed Mps:** | Crazy Robotaxi | `--live-edit-nitro-max-speed` | Absolute cap on boosted maximum speed. |
| **Mystery Seed:** | Crazy Robotaxi | `--live-edit-item-mystery-seed` | Repeatable mystery-item selection; blank uses fresh randomness. |
| **Flash Seconds:** | Crazy Robotaxi | — | Duration of the pickup notice on the HUD. |

## WEATHER

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-weather`, `--no-live-edit-weather` | Allows live weather cycling. |
| **Guidance Scale:** | Crazy Robotaxi | `--live-edit-weather-guidance` | Strength of weather-change guidance; guided chunks run an extra model forward. |
| **Guidance Chunks:** | Crazy Robotaxi | `--live-edit-weather-guidance-chunks` | Number of guided chunks after changing weather. |
| **Maintain Interval Chunks:** | Crazy Robotaxi | `--live-edit-weather-maintain-interval` | Chunks between optional weather-guidance refresh pulses; `0` disables them. |
| **Maintain Chunks:** | Crazy Robotaxi | `--live-edit-weather-maintain-chunks` | Guided chunks in each weather-guidance refresh pulse. |
| **Clear Guidance Chunks:** | Crazy Robotaxi | `--live-edit-weather-clear-guidance-chunks` | Guided chunks used when changing back to clear weather. |
| **Corrector Gain:** | Crazy Robotaxi | `--live-edit-weather-corrector-gain` | Strength of optional drift correction while weather is active. |
| **Corrector Checkpoint:** | Crazy Robotaxi | `--live-edit-weather-corrector` | Optional weather corrector checkpoint; blank reuses the style corrector. |
| **Weathers:** | Crazy Robotaxi | — | List of selectable weather presets with names and prompt suffixes. `--live-edit-weather-first NAME` only rotates the existing list. |

## OBSTACLE

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-obstacle`, `--no-live-edit-obstacle` | Enables obstacle spawn events. |
| **Count:** | Crazy Robotaxi | `--live-edit-obstacle-count` | Obstacle vehicles created by one spawn request. |
| **Spacing M:** | Crazy Robotaxi | — | Extra distance ahead for each additional obstacle vehicle. |
| **Stagger Chunks:** | Crazy Robotaxi | `--live-edit-obstacle-stagger-chunks` | Generated chunks between vehicles in one spawn request. |
| **Spawn Ahead M:** | Crazy Robotaxi | `--live-edit-obstacle-ahead-m` | Distance ahead of the taxi for the first obstacle vehicle. |
| **Lateral M:** | Crazy Robotaxi | — | Sideways offset of spawned obstacles from the taxi heading. |
| **Active Chunks:** | Crazy Robotaxi | `--live-edit-obstacle-chunks` | Maximum generated chunks before a moving event despawns. |
| **Min Drift M:** | Crazy Robotaxi | — | Minimum displacement for a moving obstacle template. |
| **Min Coverage S:** | Crazy Robotaxi | — | Minimum duration of a moving obstacle source track. |
| **Length Range M:** | Crazy Robotaxi | — | Accepted vehicle-length range for obstacle templates. |
| **Collision Radius M:** | Crazy Robotaxi | — | Distance used to detect visual-only obstacle hits. |
| **Physics:** | Crazy Robotaxi | `--live-edit-obstacle-physics`, `--no-live-edit-obstacle-physics` | Registers obstacles with the physical simulation. |
| **Placement:** | Crazy Robotaxi | `--live-edit-obstacle-placement` | `ego-relative` or `road-ahead` placement. |
| **Static Count:** | Crazy Robotaxi | `--live-edit-obstacle-static-count` | Number of persistent roadblock cars placed at game start. |
| **Static Ahead M:** | Crazy Robotaxi | `--live-edit-obstacle-static-ahead-m` | Distance ahead of the spawn pose for the first persistent car. |
| **Static Lateral M:** | Crazy Robotaxi | `--live-edit-obstacle-static-lateral-m` | Sideways offset used for alternating persistent cars. |
| **Guide Scale:** | Crazy Robotaxi | `--live-edit-obstacle-guide-scale` | Box-conditioning guidance strength; `0` disables guidance. Guided event chunks do extra model work. |
| **Annotate:** | Crazy Robotaxi | `--live-edit-obstacle-annotate`, `--no-live-edit-obstacle-annotate` | Draws projected obstacle boxes into presented frames. |

## MAP CONTEXT

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-map-context`, `--no-live-edit-map-context` | Adds authored road, landmark, topology, curve, and motion context to the model prompt. |
