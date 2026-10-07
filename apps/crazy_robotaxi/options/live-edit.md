# LIVE EDIT

[Options index](../OPTIONS.md)

All abilities start disabled. Enabling style, weather, map context, or guided obstacles requires Python transformer hooks and disables native DiT acceleration when it was selected. Enabling mystery items also enables style; rain or snow items also enable weather. Weather cannot change while a non-base style is active.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Sharpen Amount:** | Crazy Robotaxi | — | Unsharp-mask strength applied to styled frames (0 disables). |
| **Sharpen Sigma:** | Crazy Robotaxi | — | Gaussian sigma of the unsharp mask. |
| **Perf Log Every Frames:** | Crazy Robotaxi | `--live-edit-perf-log` | Log p50/p95 of the live-edit per-frame costs (coin-update CPU ms, compositor enqueue CPU ms, compositor GPU ms) every N composited frames on the tensor path. 0 disables the report. Exposed as --live-edit-perf-log; LIVE_EDIT_PERF_LOG sets the CLI default. |

## STYLE

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-style`, `--no-live-edit-style` | Whether the style ability is attached to the world-model session. |
| **Lora Checkpoint:** | Crazy Robotaxi | `--live-edit-style-lora` | Pre-merged text-edit LoRA checkpoint (guidance_distill format). |
| **Corrector Checkpoint:** | Crazy Robotaxi | `--live-edit-style-corrector` | Style-drift corrector LoRA checkpoint (train_v2 format). |
| **Corrector Gain:** | Crazy Robotaxi | `--live-edit-style-gain` | Global corrector gain composed with the alpha*(t) gate profile. |
| **Corrector Mode:** | Crazy Robotaxi | `--live-edit-corrector-mode` | Drift-corrector deploy mode: fused, unfused, or off. |
| **Base Corrector Checkpoint:** | Crazy Robotaxi | `--live-edit-base-corrector` | Optional photoreal drift corrector for the BASE world state (fused mode only; the shipped lora_v2_v3_valpeak.pt deploy). None leaves the base world uncorrected. |
| **Base Corrector Gain:** | Crazy Robotaxi | `--live-edit-base-corrector-gain` | Gain for the base-state photoreal corrector (corrgate025). |
| **Gate Alpha Json:** | Crazy Robotaxi | `--live-edit-gate-alpha-json` | Measured per-timestep gate profile (edit_sft/gate_style.py output). |
| **Guidance Scale:** | Crazy Robotaxi | — | Edit-window strength marker for skin swaps. With the pre-merged edit LoRA deployed, any value > 1.0 (together with guidance_chunks > 0) opens the single-branch LoRA window; exactly 1.0 falls back to a plain swap, which deactivates the LoRA. |
| **Guidance Chunks:** | Crazy Robotaxi | `--live-edit-skin-guidance-chunks` | Number of chunks the LoRA edit window stays open after a swap. |
| **Reswap Interval Chunks:** | Crazy Robotaxi | `--live-edit-style-reswap-chunks` | Re-issue the active skin's replace_text every N generated chunks. 0 disables the refresh. |
| **Skins:** | Crazy Robotaxi | — | Selectable skins, cycled by the switch-skin key. |

## COINS

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-coins`, `--no-live-edit-coins` | Whether coins are laid out, rendered, and collectible. |
| **Spacing M:** | Crazy Robotaxi | — | Arc-length spacing between coin groups along each navigation lane. |
| **Group Offsets M:** | Crazy Robotaxi | — | Lateral offsets of the coins in one group, metres across the lane. |
| **Hover Height M:** | Crazy Robotaxi | — | Coin center height above the waypoint ground point. |
| **Coin Diameter M:** | Crazy Robotaxi | — | World-space coin diameter used for sprite scaling. |
| **Pickup Radius M:** | Crazy Robotaxi | — | XY distance at which the ego collects a coin. |
| **Max Render Distance M:** | Crazy Robotaxi | — | Coins farther than this are not composited. |
| **Fade Start Distance M:** | Crazy Robotaxi | — | Alpha ramps to zero between this distance and the render limit. |
| **Max Visible Sprites:** | Crazy Robotaxi | `--live-edit-coin-max-visible` | Composite at most this many coins per frame, keeping the nearest. 0 disables the cap. |
| **Sprite Path:** | Crazy Robotaxi | `--live-edit-coin-sprite` | RGBA coin sprite; None renders a procedural coin. |

## ITEMS

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-items`, `--no-live-edit-items` | Whether effect items are laid out, rendered, and collectible. |
| **Spacing M:** | Crazy Robotaxi | `--live-edit-item-spacing` | Arc-length spacing between items along each navigation lane (items are rare by design; 150-300 m is the intended range). |
| **Hover Height M:** | Crazy Robotaxi | — | Item center height above the waypoint ground point. |
| **Item Diameter M:** | Crazy Robotaxi | — | World-space item height used for sprite scaling (bigger than a coin so the rare pickups read from a distance). |
| **Pickup Radius M:** | Crazy Robotaxi | — | XY distance at which the ego collects an item. |
| **Max Render Distance M:** | Crazy Robotaxi | — | Items farther than this are not composited. |
| **Fade Start Distance M:** | Crazy Robotaxi | — | Alpha ramps to zero between this distance and the render limit. |
| **Rain Sprite Path:** | Crazy Robotaxi | `--live-edit-item-rain-sprite` | RGBA rain-item sprite; None renders a procedural placeholder. Sprite files are local-only paths, never bundled (coin-sprite pattern). |
| **Snow Sprite Path:** | Crazy Robotaxi | `--live-edit-item-snow-sprite` | RGBA snow-item sprite; None renders a procedural placeholder. |
| **Mystery Sprite Path:** | Crazy Robotaxi | `--live-edit-item-mystery-sprite` | RGBA mystery-box sprite; None renders a procedural '?' box. |
| **Nitro Sprite Path:** | Crazy Robotaxi | `--live-edit-item-nitro-sprite` | RGBA nitro-item sprite; None renders a procedural placeholder. |
| **Item Types:** | Crazy Robotaxi | `--live-edit-item-types` | Item kinds included in the course mix, cycled in this order by the layout walk (equal rarity per kind). A subset (e.g. ("nitro",)) makes a single-effect course for scripted captures; the default mixes every kind. Exposed as --live-edit-item-types. |
| **Nitro Boost:** | Crazy Robotaxi | `--live-edit-nitro-boost` | Nitro multiplier applied to the vehicle's max speed while a pickup is active (>= 1). Acceleration is unchanged so the boost does not exaggerate the suspension pitch response. |
| **Nitro Duration S:** | Crazy Robotaxi | `--live-edit-nitro-duration-s` | Nitro boost duration in game time (simulated seconds, accumulated from the physics-tick dt, which is wall time at the shipped realtime recipe). Picking a second nitro while boosted RESETS the timer to this value — no multiplicative stacking. |
| **Nitro Max Speed Mps:** | Crazy Robotaxi | `--live-edit-nitro-max-speed` | Hard ceiling on the boosted max speed. A ceiling below the vehicle's normal speed limit never slows the vehicle. |
| **Mystery Seed:** | Crazy Robotaxi | `--live-edit-item-mystery-seed` | Seed for the mystery-box skin roll (reproducible captures); None draws from the OS entropy pool. Re-seeded per rollout. |
| **Flash Seconds:** | Crazy Robotaxi | — | How long the pickup HUD flash chip stays up. |

## WEATHER

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-weather`, `--no-live-edit-weather` | Whether the weather ability responds to the weather-cycle key. |
| **Guidance Scale:** | Crazy Robotaxi | `--live-edit-weather-guidance` | Two-prompt edit-guidance strength for weather swaps. |
| **Guidance Chunks:** | Crazy Robotaxi | `--live-edit-weather-guidance-chunks` | Number of guided chunks after a weather swap; each costs about twice model time. |
| **Maintain Interval Chunks:** | Crazy Robotaxi | `--live-edit-weather-maintain-interval` | Re-open a short guidance window every N chunks while weather holds. 0 disables maintenance pulses. |
| **Maintain Chunks:** | Crazy Robotaxi | `--live-edit-weather-maintain-chunks` | Guided chunks per maintenance pulse (used when maintain_interval_chunks > 0). Exposed as --live-edit-weather-maintain-chunks. |
| **Clear Guidance Chunks:** | Crazy Robotaxi | `--live-edit-weather-clear-guidance-chunks` | Guided chunks for the weather -> clear landing when the cycle wraps. Slightly longer than the 6-chunk activation landing because dense states (hurricane fog walls) dissipate slower than they land. Exposed as --live-edit-weather-clear-guidance-chunks. |
| **Corrector Gain:** | Crazy Robotaxi | `--live-edit-weather-corrector-gain` | Absolute style-drift-corrector gain while weather is active. 0 (default) keeps the corrector off during weather. |
| **Corrector Checkpoint:** | Crazy Robotaxi | `--live-edit-weather-corrector` | Dedicated corrector checkpoint for the weather state (fused mode). None reuses the style corrector at corrector_gain. |
| **Weathers:** | Crazy Robotaxi | — | Selectable weathers, cycled clear -> rain -> snow -> storm -> hurricane -> clear by default. |

## OBSTACLE

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-obstacle`, `--no-live-edit-obstacle` | Whether the obstacle ability responds to the spawn key. |
| **Count:** | Crazy Robotaxi | `--live-edit-obstacle-count` | Cars per spawn request. Additional cars alternate crossing direction and are staggered by spacing_m and stagger_chunks. |
| **Spacing M:** | Crazy Robotaxi | — | Extra ahead-distance per additional car (count > 1). The default puts a 4-car burst across a 16-40 m band — the model's validated materialization range. |
| **Stagger Chunks:** | Crazy Robotaxi | `--live-edit-obstacle-stagger-chunks` | Chunks between consecutive car spawns in one burst. 0 spawns the whole burst in one chunk; a small stagger both eases the model into the event and spreads the passes out on screen. |
| **Spawn Ahead M:** | Crazy Robotaxi | `--live-edit-obstacle-ahead-m` | Ahead distance for the first event in the selected placement mode. |
| **Lateral M:** | Crazy Robotaxi | — | Meters to the left (+) / right (-) of the ego heading at spawn. |
| **Active Chunks:** | Crazy Robotaxi | `--live-edit-obstacle-chunks` | Despawn each non-static event after this many generated chunks. |
| **Min Drift M:** | Crazy Robotaxi | — | Minimum ground-plane displacement for a moving template. |
| **Min Coverage S:** | Crazy Robotaxi | — | Minimum source-track duration for a moving template. |
| **Length Range M:** | Crazy Robotaxi | — | Inclusive vehicle-length filter for obstacle templates. |
| **Collision Radius M:** | Crazy Robotaxi | — | Ego XY distance at which a visual-only event logs a hit. |
| **Physics:** | Crazy Robotaxi | `--live-edit-obstacle-physics`, `--no-live-edit-obstacle-physics` | Register obstacles with PhysX. False preserves PR494's visual-only conditioning behavior; true makes collisions authoritative. |
| **Placement:** | Crazy Robotaxi | `--live-edit-obstacle-placement` | Placement resolver: ego-relative preserves PR494 behavior; road-ahead walks the compiled directed-lane graph. |
| **Static Count:** | Crazy Robotaxi | `--live-edit-obstacle-static-count` | Static roadblock cars placed ahead of the spawn pose from the first chunk and retained until reset. 0 disables. |
| **Static Ahead M:** | Crazy Robotaxi | `--live-edit-obstacle-static-ahead-m` | Meters ahead of the spawn pose where the first static car sits (nearer slots fight the initial frame hardest and stay ghost). |
| **Static Lateral M:** | Crazy Robotaxi | `--live-edit-obstacle-static-lateral-m` | Lateral offset magnitude of the alternating static-car slots. |
| **Guide Scale:** | Crazy Robotaxi | `--live-edit-obstacle-guide-scale` | Box-axis guidance strength (flow extrapolated along the with-box/without-box conditioning direction). 0 disables the guidance hook entirely (the event may render at ghost strength). Guided event chunks cost about twice model time. Unsupported by the native DiT executor. |
| **Annotate:** | Crazy Robotaxi | `--live-edit-obstacle-annotate`, `--no-live-edit-obstacle-annotate` | Draw each event's projected 3D box outline into presented frames (evidence/demo aid). |

## MAP CONTEXT

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Enabled:** | Crazy Robotaxi | `--live-edit-map-context`, `--no-live-edit-map-context` | Whether road, topology, and motion clauses update the model prompt. |
