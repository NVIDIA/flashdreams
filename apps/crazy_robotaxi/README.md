# Crazy Robotaxi

Crazy Robotaxi is a standalone game built on `omnidreams-game-engine` and the
OmniDreams inference session.

Launch the native game:

```bash
flashdreams-run crazy-robotaxi
```

Select the bundled performance manifest:

```bash
flashdreams-run crazy-robotaxi \
  --world-model-manifest example_world_model_perf.yaml
```

`flashdreams-run` reserves `--manifest` for its launch-manifest format. Use
`--world-model-manifest` for the OmniDreams model manifest. Runner
booleans use explicit `True` / `False` values because the shared FlashDreams
CLI disables implicit boolean flag conversion.

The dedicated entry point exposes the application options:

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
- **Obstacle events** (`--live-edit-obstacle`, spawn with `O`): generated
  crossing cars owned by a dedicated event controller; they do not depend
  on recorded-scene tracks or participate in NPC routes/recovery.
  The event/archetype layer is general, but the public catalog currently
  exposes only cars.
  `--live-edit-obstacle-count N` creates a staggered burst. Placement is
  ego-relative by default; `--live-edit-obstacle-placement road-ahead`
  resolves the spawn along the compiled directed-lane graph. Events remain
  visual-only by default for PR494 compatibility;
  `--live-edit-obstacle-physics` gives them PhysX bodies that detach from
  their scripts when struck. `--live-edit-obstacle-guide-scale > 0` adds a
  second forward per step while an event is active. The guidance is
  CUDA-graph safe (guided steps replay the captured graph twice with
  box/no-box conditioning staged in), so the accelerated pipeline stays on;
  event chunks cost ~2x model time, non-event chunks are unchanged.
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

## Configuration files

The standalone game keeps its portable configuration in three independent,
strict YAML documents:

- `*.robotaxi.yaml` describes one map, including topology, geometry, profiles,
  compiler settings, spawns, and visual seed variants.
- `default_renderer.yaml` describes primary-camera and BEV rendering.
- `default_game.yaml` describes rules, scoring, controls, taxi dimensions, and
  arcade physics.

The packaged renderer and game files are used when no path is supplied. Select
edited copies independently:

```bash
flashdreams-run crazy-robotaxi \
  --renderer-config /path/to/renderer.yaml \
  --game-config /path/to/game.yaml
```

Existing explicit CLI tuning flags override YAML values. The world-model
manifest remains separate because it configures inference rather than the map,
renderer, or game. All three YAML formats reject missing and unknown fields.

## Node-graph game maps

Crazy Robotaxi maps use schema version 1 of the engine's node-graph format.
Roads are topological edges; intersections, road joints, cul-de-sacs, parking
lots, and driveways are explicitly posed graph nodes. Road joints provide
minimal tangent-continuous degree-two bends and own lane-count or lane-width
tapers between otherwise uniform road edges. Intersections have at least three
road arms and can apply the same taper to each inferred through-road pair
independently. Node footprints follow the bearings and centerline tangents of
the roads connected to them.

The complete, authoritative format is documented in
[`../omnidreams_game_engine/NODE_GRAPH_MAP_FORMAT.md`](../omnidreams_game_engine/NODE_GRAPH_MAP_FORMAT.md).
The bundled `minimal_loop.robotaxi.yaml` is a compact working example, while
`boulevard_district.robotaxi.yaml` recreates the original scene's surface-street
layout at its source scale, including the curved arterial split, neighborhood
grid, eastern commercial loops, cul-de-sacs, and parking lots. The elevated
highway and its on-ramps in the northwest are intentionally omitted.

Select a map with either entry point:

```bash
flashdreams-run crazy-robotaxi \
  --map /path/to/city.robotaxi.yaml
```

Validate a map or produce top-down and spawn-camera previews without loading a
model:

```bash
crazy-robotaxi-map validate /path/to/city.robotaxi.yaml
crazy-robotaxi-map preview /path/to/city.robotaxi.yaml --output city.svg
crazy-robotaxi-map preview-spawn /path/to/city.robotaxi.yaml \
  --spawn taxi_start --output taxi_start.png
```

The engine validates the topology, compiles roads and inferred parking-access
geometry, creates curb colliders, and derives directed navigation lanes. That
lane graph powers fare reachability and future NPC routing; it does not limit
where the player's taxi may physically drive. Parking-lot fare points are
sampled anywhere inside the authored polygon, while their route distance stops
at the connected driveway or intersection.

The engine's private ClipGT archive is cached under
`$FLASHDREAMS_CACHE_DIR/omnidreams-game-engine/game-maps/`. The cache key includes the YAML,
resolved geometry, compiler implementation, and referenced seed assets, so
map or compiler edits rebuild it at the next load. The archive is runtime
output, not an authoring format. Pass `--force-map-recompile` when launching
the game to rebuild each selected map once for that process even if its cache
entry is valid. The standalone equivalent is:

```bash
crazy-robotaxi-map compile /path/to/city.robotaxi.yaml --force-map-recompile
```

Parking-lot surfaces are emitted as green `ROI_POLYGON_ROADNET_MASK` regions.
Parking-space dividers remain intentionally omitted because ClipGT would encode
them as ordinary lane lines, which can condition the model to produce bike or
turn lanes. Lots do not contain inferred navigation aisles or turnarounds.

Each visual variant may omit `image`. In that case, map compilation and scene
selection use a deterministic synthetic view projected from the spawn through
the runtime front camera. It aligns roads, boundaries, curbs, and markings but
does not synthesize buildings, vegetation, traffic, or other scenery. Authors
can inspect that fallback with `preview-spawn` before choosing or generating a
checked-in image.

The bundled maps reuse the existing OmniDreams seed image. Their authored
geometry is not expected to match that image exactly, so the first generated
frames may visibly adjust toward the selected semantic map. Static prompts
describe visual setting and atmosphere rather than duplicating map topology;
the BEV remains the source of truth as maps change.
