# LIVE EDIT

[Options index](../OPTIONS.md)

All abilities start disabled. Enabling style, weather, map context, or guided obstacles requires Python transformer hooks and disables native DiT acceleration when it was selected. Enabling mystery items also enables style; rain or snow items also enable weather. Weather cannot change while a non-base style is active.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Sharpen Amount:** | `live_edit.sharpen_amount` | Crazy Robotaxi | — | Unsharp-mask strength on styled frames; `0` disables sharpening. |
| **Sharpen Sigma:** | `live_edit.sharpen_sigma` | Crazy Robotaxi | — | Gaussian blur radius used by the unsharp mask. |
| **Perf Log Every Frames:** | `live_edit.perf_log_every_frames` | Crazy Robotaxi | `--live-edit-perf-log` | Interval for live-edit CPU/GPU cost reports; `0` disables them. |

## STYLE

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Enabled:** | `live_edit.style.enabled` | Crazy Robotaxi | `--live-edit-style`, `--no-live-edit-style` | Allows live skin switching. |
| **Lora Checkpoint:** | `live_edit.style.lora_checkpoint` | Crazy Robotaxi | `--live-edit-style-lora` | Pre-merged text-edit LoRA used for skin swaps. |
| **Corrector Checkpoint:** | `live_edit.style.corrector_checkpoint` | Crazy Robotaxi | `--live-edit-style-corrector` | Checkpoint for the style drift corrector. |
| **Corrector Gain:** | `live_edit.style.corrector_gain` | Crazy Robotaxi | `--live-edit-style-gain` | Strength of the style drift corrector. |
| **Corrector Mode:** | `live_edit.style.corrector_mode` | Crazy Robotaxi | `--live-edit-corrector-mode` | `fused`, `unfused`, or `off` deployment of the corrector. |
| **Base Corrector Checkpoint:** | `live_edit.style.base_corrector_checkpoint` | Crazy Robotaxi | `--live-edit-base-corrector` | Optional corrector checkpoint for the base visual state. |
| **Base Corrector Gain:** | `live_edit.style.base_corrector_gain` | Crazy Robotaxi | `--live-edit-base-corrector-gain` | Corrector strength for the base visual state. |
| **Gate Alpha Json:** | `live_edit.style.gate_alpha_json` | Crazy Robotaxi | `--live-edit-gate-alpha-json` | Per-timestep corrector gate profile. |
| **Guidance Scale:** | `live_edit.style.guidance_scale` | Crazy Robotaxi | — | Guidance strength for a skin change. |
| **Guidance Chunks:** | `live_edit.style.guidance_chunks` | Crazy Robotaxi | `--live-edit-skin-guidance-chunks` | Number of chunks in the skin-change guidance window. |
| **Reswap Interval Chunks:** | `live_edit.style.reswap_interval_chunks` | Crazy Robotaxi | `--live-edit-style-reswap-chunks` | Reapplies the active skin every N chunks; `0` disables refresh. |
| **Skins:** | `live_edit.style.skins` | Crazy Robotaxi | — | List of selectable `{name, prompt}` skins. `--live-edit-skin-first NAME` only rotates the existing list. |

## COINS

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Enabled:** | `live_edit.coins.enabled` | Crazy Robotaxi | `--live-edit-coins`, `--no-live-edit-coins` | Places collectible coins along lanes. |
| **Spacing M:** | `live_edit.coins.spacing_m` | Crazy Robotaxi | — | Distance between coin groups along the road. |
| **Group Offsets M:** | `live_edit.coins.group_offsets_m` | Crazy Robotaxi | — | Lateral positions of coins within a group. |
| **Hover Height M:** | `live_edit.coins.hover_height_m` | Crazy Robotaxi | — | Height of coin centers above the road. |
| **Coin Diameter M:** | `live_edit.coins.coin_diameter_m` | Crazy Robotaxi | — | World-space coin diameter used for rendering. |
| **Pickup Radius M:** | `live_edit.coins.pickup_radius_m` | Crazy Robotaxi | — | Distance at which the taxi collects a coin. |
| **Max Render Distance M:** | `live_edit.coins.max_render_distance_m` | Crazy Robotaxi | — | Coin visibility limit. |
| **Fade Start Distance M:** | `live_edit.coins.fade_start_distance_m` | Crazy Robotaxi | — | Distance where coins begin fading out. |
| **Max Visible Sprites:** | `live_edit.coins.max_visible_sprites` | Crazy Robotaxi | `--live-edit-coin-max-visible` | Per-frame cap, keeping nearest coins; `0` removes the cap. |
| **Sprite Path:** | `live_edit.coins.sprite_path` | Crazy Robotaxi | `--live-edit-coin-sprite` | Optional RGBA coin image; blank uses the procedural coin. |

## ITEMS

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Enabled:** | `live_edit.items.enabled` | Crazy Robotaxi | `--live-edit-items`, `--no-live-edit-items` | Places collectible effect items along lanes. |
| **Spacing M:** | `live_edit.items.spacing_m` | Crazy Robotaxi | `--live-edit-item-spacing` | Distance between effect items along the road. |
| **Hover Height M:** | `live_edit.items.hover_height_m` | Crazy Robotaxi | — | Height of effect items above the road. |
| **Item Diameter M:** | `live_edit.items.item_diameter_m` | Crazy Robotaxi | — | World-space item size used for rendering. |
| **Pickup Radius M:** | `live_edit.items.pickup_radius_m` | Crazy Robotaxi | — | Distance at which the taxi collects an effect item. |
| **Max Render Distance M:** | `live_edit.items.max_render_distance_m` | Crazy Robotaxi | — | Item visibility limit. |
| **Fade Start Distance M:** | `live_edit.items.fade_start_distance_m` | Crazy Robotaxi | — | Distance where items begin fading out. |
| **Rain Sprite Path:** | `live_edit.items.rain_sprite_path` | Crazy Robotaxi | `--live-edit-item-rain-sprite` | Optional RGBA artwork for each item; blank uses a procedural placeholder. |
| **Snow Sprite Path:** | `live_edit.items.snow_sprite_path` | Crazy Robotaxi | `--live-edit-item-snow-sprite` | Optional RGBA artwork for each item; blank uses a procedural placeholder. |
| **Mystery Sprite Path:** | `live_edit.items.mystery_sprite_path` | Crazy Robotaxi | `--live-edit-item-mystery-sprite` | Optional RGBA artwork for each item; blank uses a procedural placeholder. |
| **Nitro Sprite Path:** | `live_edit.items.nitro_sprite_path` | Crazy Robotaxi | `--live-edit-item-nitro-sprite` | Optional RGBA artwork for each item; blank uses a procedural placeholder. |
| **Item Types:** | `live_edit.items.item_types` | Crazy Robotaxi | `--live-edit-item-types` | Mix of `rain`, `snow`, `mystery`, and `nitro` items placed on the course. |
| **Nitro Boost:** | `live_edit.items.nitro_boost` | Crazy Robotaxi | `--live-edit-nitro-boost` | Multiplier applied to maximum speed while nitro is active. |
| **Nitro Duration S:** | `live_edit.items.nitro_duration_s` | Crazy Robotaxi | `--live-edit-nitro-duration-s` | Duration of a nitro boost in simulation time. |
| **Nitro Max Speed Mps:** | `live_edit.items.nitro_max_speed_mps` | Crazy Robotaxi | `--live-edit-nitro-max-speed` | Absolute cap on boosted maximum speed. |
| **Mystery Seed:** | `live_edit.items.mystery_seed` | Crazy Robotaxi | `--live-edit-item-mystery-seed` | Repeatable mystery-item selection; blank uses fresh randomness. |
| **Flash Seconds:** | `live_edit.items.flash_seconds` | Crazy Robotaxi | — | Duration of the pickup notice on the HUD. |

## WEATHER

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Enabled:** | `live_edit.weather.enabled` | Crazy Robotaxi | `--live-edit-weather`, `--no-live-edit-weather` | Allows live weather cycling. |
| **Guidance Scale:** | `live_edit.weather.guidance_scale` | Crazy Robotaxi | `--live-edit-weather-guidance` | Strength of weather-change guidance; guided chunks run an extra model forward. |
| **Guidance Chunks:** | `live_edit.weather.guidance_chunks` | Crazy Robotaxi | `--live-edit-weather-guidance-chunks` | Number of guided chunks after changing weather. |
| **Maintain Interval Chunks:** | `live_edit.weather.maintain_interval_chunks` | Crazy Robotaxi | `--live-edit-weather-maintain-interval` | Chunks between optional weather-guidance refresh pulses; `0` disables them. |
| **Maintain Chunks:** | `live_edit.weather.maintain_chunks` | Crazy Robotaxi | `--live-edit-weather-maintain-chunks` | Guided chunks in each weather-guidance refresh pulse. |
| **Clear Guidance Chunks:** | `live_edit.weather.clear_guidance_chunks` | Crazy Robotaxi | `--live-edit-weather-clear-guidance-chunks` | Guided chunks used when changing back to clear weather. |
| **Corrector Gain:** | `live_edit.weather.corrector_gain` | Crazy Robotaxi | `--live-edit-weather-corrector-gain` | Strength of optional drift correction while weather is active. |
| **Corrector Checkpoint:** | `live_edit.weather.corrector_checkpoint` | Crazy Robotaxi | `--live-edit-weather-corrector` | Optional weather corrector checkpoint; blank reuses the style corrector. |
| **Weathers:** | `live_edit.weather.weathers` | Crazy Robotaxi | — | List of selectable weather presets with names and prompt suffixes. `--live-edit-weather-first NAME` only rotates the existing list. |

## OBSTACLE

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Enabled:** | `live_edit.obstacle.enabled` | Crazy Robotaxi | `--live-edit-obstacle`, `--no-live-edit-obstacle` | Enables obstacle spawn events. |
| **Count:** | `live_edit.obstacle.count` | Crazy Robotaxi | `--live-edit-obstacle-count` | Obstacle vehicles created by one spawn request. |
| **Spacing M:** | `live_edit.obstacle.spacing_m` | Crazy Robotaxi | — | Extra distance ahead for each additional obstacle vehicle. |
| **Stagger Chunks:** | `live_edit.obstacle.stagger_chunks` | Crazy Robotaxi | `--live-edit-obstacle-stagger-chunks` | Generated chunks between vehicles in one spawn request. |
| **Spawn Ahead M:** | `live_edit.obstacle.spawn_ahead_m` | Crazy Robotaxi | `--live-edit-obstacle-ahead-m` | Distance ahead of the taxi for the first obstacle vehicle. |
| **Lateral M:** | `live_edit.obstacle.lateral_m` | Crazy Robotaxi | — | Sideways offset of spawned obstacles from the taxi heading. |
| **Active Chunks:** | `live_edit.obstacle.active_chunks` | Crazy Robotaxi | `--live-edit-obstacle-chunks` | Maximum generated chunks before a moving event despawns. |
| **Min Drift M:** | `live_edit.obstacle.min_drift_m` | Crazy Robotaxi | — | Minimum displacement for a moving obstacle template. |
| **Min Coverage S:** | `live_edit.obstacle.min_coverage_s` | Crazy Robotaxi | — | Minimum duration of a moving obstacle source track. |
| **Length Range M:** | `live_edit.obstacle.length_range_m` | Crazy Robotaxi | — | Accepted vehicle-length range for obstacle templates. |
| **Collision Radius M:** | `live_edit.obstacle.collision_radius_m` | Crazy Robotaxi | — | Distance used to detect visual-only obstacle hits. |
| **Physics:** | `live_edit.obstacle.physics` | Crazy Robotaxi | `--live-edit-obstacle-physics`, `--no-live-edit-obstacle-physics` | Registers obstacles with the physical simulation. |
| **Placement:** | `live_edit.obstacle.placement` | Crazy Robotaxi | `--live-edit-obstacle-placement` | `ego-relative` or `road-ahead` placement. |
| **Static Count:** | `live_edit.obstacle.static_count` | Crazy Robotaxi | `--live-edit-obstacle-static-count` | Number of persistent roadblock cars placed at game start. |
| **Static Ahead M:** | `live_edit.obstacle.static_ahead_m` | Crazy Robotaxi | `--live-edit-obstacle-static-ahead-m` | Distance ahead of the spawn pose for the first persistent car. |
| **Static Lateral M:** | `live_edit.obstacle.static_lateral_m` | Crazy Robotaxi | `--live-edit-obstacle-static-lateral-m` | Sideways offset used for alternating persistent cars. |
| **Guide Scale:** | `live_edit.obstacle.guide_scale` | Crazy Robotaxi | `--live-edit-obstacle-guide-scale` | Box-conditioning guidance strength; `0` disables guidance. Guided event chunks do extra model work. |
| **Annotate:** | `live_edit.obstacle.annotate` | Crazy Robotaxi | `--live-edit-obstacle-annotate`, `--no-live-edit-obstacle-annotate` | Draws projected obstacle boxes into presented frames. |

## MAP CONTEXT

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Enabled:** | `live_edit.map_context.enabled` | Crazy Robotaxi | `--live-edit-map-context`, `--no-live-edit-map-context` | Adds authored road, landmark, topology, curve, and motion context to the model prompt. |
