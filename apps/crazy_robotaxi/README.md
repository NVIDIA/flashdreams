# Crazy Robotaxi

Crazy Robotaxi is a standalone game built on `omnidreams-game-engine` and the
legacy OmniDreams inference session. It does not import or modify the
Interactive Drive demo.

Launch the native game:

```bash
flashdreams-run crazy-robotaxi
```

Select the bundled performance manifest and load the scene immediately:

```bash
flashdreams-run crazy-robotaxi \
  --world-model-manifest example_world_model_perf.yaml \
  --auto-start True
```

`flashdreams-run` reserves `--manifest` for its launch-manifest format. Use
`--world-model-manifest` for the legacy OmniDreams model manifest, or use the
dedicated `crazy-robotaxi` executable below. Runner booleans use explicit
`True` / `False` values because the shared FlashDreams CLI disables implicit
boolean flag conversion.

The dedicated entry point exposes the complete legacy option surface:

```bash
crazy-robotaxi --help
```

Use `--stream-mjpeg HOST:PORT` with either entry point to run the browser HUD
instead of opening a local Vulkan window.

## Live-edit abilities

Flag-gated, off by default (`--live-edit-*`; see `crazy-robotaxi --help`,
group "live edit"):

- **Coins** (`--live-edit-coins`, toggle with `C`): collectible coins
  composited along the route. Pixel-only — no model hooks; when the model
  frame is CUDA-resident (native window fast path, MJPEG pre-download) the
  sprites/HUD chips are blended on the GPU with pre-uploaded textures
  (< 1 ms per frame), so coins are perf-neutral and work under
  `native_dit_acceleration`.
- **Skins** (`--live-edit-style` + `--live-edit-style-lora`, cycle with
  `K`): prompt swaps realized by a pre-merged text-edit LoRA. The window
  runs single-branch (no per-step extra forward). All skin/weather prompts
  are pre-encoded once at session start, so a swap injects cached
  embeddings — the swap-boundary chunk no longer pays the 0.5-1 s text
  re-encode. `--live-edit-skin-duration-chunks N` turns skins into timed
  power-ups: an activation auto-reverts to the base world after N chunks
  (11 ≈ 3 s), the HUD chip counts down the remaining time, and `K` keeps
  its cycle semantics during an active power-up — next skin, fresh timer
  (the default 0 holds a skin until cycled).
- **Weather** (`--live-edit-weather`, cycle with `V`): guided prompt swaps
  with no LoRA, deployed land-then-release. **Transient cost: the landing
  window costs ~2x per chunk** (a second forward per denoise step) for
  `--live-edit-weather-guidance-chunks` chunks (default 6); the weather
  then holds unguided at ~1x — it persists through the KV history and the
  swapped text (A/B'd: a 27-chunk unguided hold matches the always-guided
  policy). `--live-edit-weather-maintain-interval N` optionally re-opens a
  short rebased window (`--live-edit-weather-maintain-chunks`, default 2)
  every N chunks; the default 0 never does (measured unnecessary).
  Weather is timed by default: any activation (V key or item pickup)
  auto-reverts to clear after `--live-edit-weather-duration-chunks` chunks
  (default 90 ≈ 24 s; 0 holds until cycled), with a HUD countdown. The
  revert lands GUIDED for `--live-edit-weather-clear-guidance-chunks`
  chunks (default 8 — clear is itself a weather transition; a plain swap
  leaves precipitation running on KV momentum). Accepted physics: the
  revert stops new precipitation but does not undo accumulated scene
  change — wet roads dry gradually and snow lingers then fades.
- **Effect items** (`--live-edit-items`): sparse pickup items along the
  lanes — rain/snow icons trigger that weather preset, a mystery box
  grants a random timed skin burst (`--live-edit-item-mystery-burst-chunks`,
  default 11 ≈ 3 s, even when the global skin duration is hold-forever;
  `--live-edit-item-mystery-seed` makes the roll reproducible). Pickups
  dispatch through the same state machines as the K/V keys at the next
  chunk boundary, with a HUD flash; the keys stay fully live alongside.
  Weather items obey the base-world-only rule — picking one up while a
  skin is active shows a "BLOCKED" hint instead of queueing. Item sparsity
  is global over the lane network (`--live-edit-item-spacing`, default one
  item per 200 m neighborhood). Sprites are local-only paths
  (`--live-edit-item-{rain,snow,mystery,nitro}-sprite`); without them the
  items render procedural placeholder icons, like the coin sprite.
  A **nitro** item is the physics-only exception: it applies an INSTANT
  timed speed boost inside the app-authoritative taxi integrator (no
  chunk-boundary wait, no state-machine coupling — it composes with any
  skin/weather/obstacle state). `--live-edit-nitro-boost` (default 1.6)
  multiplies max speed and max acceleration for
  `--live-edit-nitro-duration-s` (default 4 s game time; a re-pickup
  resets the timer, no stacking), with the boosted max speed hard-capped
  at `--live-edit-nitro-max-speed` (default 16 m/s) so the ego does not
  outrun the world model's manifold on the suburb map. A "NITRO x1.6"
  chip counts the boost down next to the other ability chips.
  `--live-edit-item-types` restricts the course mix (e.g. `nitro` for a
  single-effect capture course; default cycles all kinds equally).
- **Obstacle / traffic** (`--live-edit-obstacle`, spawn with `O`): cloned
  scene vehicles. `--live-edit-obstacle-count N` turns one key press into a
  traffic burst of N distinct crossing/oncoming clones staggered ahead of
  the ego (`--live-edit-obstacle-stagger-chunks` apart in time; each
  despawns after its own pass). Conditioning-only unless
  `--live-edit-obstacle-guide-scale > 0`, which adds a second forward per
  step while an event is active. The guidance is CUDA-graph safe (guided
  steps replay the captured graph twice with box/no-box conditioning
  staged in), so the accelerated pipeline stays on; event chunks cost ~2x
  model time, non-event chunks are unchanged.
- **Drift correctors** (`--live-edit-style-corrector`,
  `--live-edit-base-corrector`, `--live-edit-weather-corrector`): optional
  per-state weight-merged correctors. `--live-edit-corrector-mode off`
  disables all of them (no transformer weights are snapshotted or touched)
  even when checkpoints are configured; `fused` (default) keeps CUDA graphs
  and `compile_network` on; `unfused` is the eager fallback (slow,
  graph-free pipeline).

Native-DIT interaction: the prompt-swap abilities (skins, weather) need the
Python transformer forward — `replace_text_embeddings` is not wired for the
native optimized-DiT executor, and the LoRA/corrector weight toggles never
reach the native fp8 weight snapshot — so running them requires
`native_dit_acceleration: disabled` in the world-model manifest. Coins (and
obstacle without guidance) do not touch the model and run at full speed
under native DIT.

## Semantic game maps

Crazy Robotaxi maps are authored as versioned `.robotaxi.yaml` files. The game
engine validates named road profiles, snaps road and intersection ports,
generates directed navigation lanes and curb colliders, and transparently
compiles the result into the private ClipGT archive consumed by OmniDreams.
The generated archive is cached under
`$FLASHDREAMS_CACHE_DIR/crazy-robotaxi/maps/`; it is not an authoring format.

The bundled `minimal_loop.robotaxi.yaml` map is used by default. Select another
map with either entry point:

```bash
flashdreams-run crazy-robotaxi \
  --scene /path/to/city.robotaxi.yaml \
  --auto-start True
```

Validate a map or produce a top-down SVG showing its lanes, connection ports,
and curbs without loading a model:

```bash
crazy-robotaxi-map validate /path/to/city.robotaxi.yaml
crazy-robotaxi-map preview /path/to/city.robotaxi.yaml --output city.svg
```

A map contains one anchored element; every other element attaches a named port
to an existing `element.port`. Additional `connections` close loops and are
validated for position, heading, and road-profile compatibility. Unconnected
road ports receive curb end caps automatically.

Minimal authoring shape:

```yaml
schema_version: 1
id: my-map
name: My Map
profiles:
  neighborhood:
    lane_width_m: 3.6
    curb_offset_m: 0.6
    lanes: [backward, forward]
    speed_limit_mps: 13.4
    curb: true
    lane_marking: {style: DASHED_SINGLE, color: WHITE}
elements:
  - id: main
    type: road_segment
    profile: neighborhood
    geometry: {kind: straight, length_m: 50}
    pose: {x_m: 0, y_m: 0, heading_deg: 0}
  - id: corner
    type: road_segment
    profile: neighborhood
    geometry: {kind: arc, radius_m: 15, sweep_deg: 90}
    attach: {port: start, to: main.end}
spawns:
  - id: taxi_start
    element: main
    lane: 1
    distance_m: 5
    variants:
      default:
        image: seed.png
        prompt: A forward-facing taxi on a neighborhood road.
```

`lane_width_m` controls routing and lane rails. `curb_offset_m` adds paved
roadside clearance beyond the outer lane rail on each side before the physical
curb; it defaults to zero.

Exactly one element uses `pose`; attached element transforms are derived from
their ports. Positive arc sweeps turn left and negative sweeps turn right.
Seed paths are relative to the YAML file. The compiler cache key includes the
YAML, every referenced seed, and the compiler version, so edits rebuild the
private archive automatically on the next load.

The current schema supports straight and constant-radius curved road segments,
T intersections, four-way intersections, flat ground, and per-spawn visual
variants. Driveways, parking lots and openings, boulevards, elevation, and
freeform splines are planned extensions.

The bundled WIP map reuses the existing OmniDreams seed image. Its geometry is
not expected to match that image, so the first generated frames may visibly
adjust toward the semantic map.
