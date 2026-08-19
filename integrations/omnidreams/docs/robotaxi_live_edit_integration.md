# Crazy Robotaxi × OmniDreams live-edit: integration plan

Status: proposal. Integrates the OmniDreams live-edit capabilities
([#458](https://github.com/NVIDIA/flashdreams/pull/458) style skins,
[#431](https://github.com/NVIDIA/flashdreams/pull/431) text-edit guidance +
spawned actors + composite overlays) into the Crazy Robotaxi standalone game
([#463](https://github.com/NVIDIA/flashdreams/pull/463)) as optional,
flag-gated gameplay "abilities". Vanilla Robotaxi behavior is unchanged when
the flags are off.

## Why Robotaxi as the host

Crazy Robotaxi already provides the game shell we would otherwise rebuild:
game loop, arcade physics, HUD, keyboard/wheel input, high scores, browser
streaming (`--stream-mjpeg`), and FTheta fisheye camera marker projection —
all on the same student config the live-edit stack targets
(`SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE`). The live-edit stack conversely
provides in-world capabilities the game can consume as events and pickups.

## Abilities

### 1. Live game-skin switching (#458)

Mid-run restyle of the whole world (e.g. arcade-racer, comic-ink) with the
scene, traffic, and controls unchanged.

- Mechanism: multi-style LoRA (r64, prompt-selected, pre-merged weight-set
  toggle — zero steady-state cost) + rank-16 drift corrector (unfused,
  gain 0.15, measured per-timestep gate profile) + unsharp output filter.
  Long-hold stability validated to 20+ chunks past the swap.
- Integration seam: attach LoRA + corrector to the inference session at
  startup (scale 0 = off); a key/HUD toggle performs the prompt swap and
  enables the adapter. New config block `live_edit.style`.
- Cost: none when off; corrector adds a small per-step overhead when on.

### 2. Item pickups: coins (compositor stack)

Collectible items rendered into the world along the drivable lane, with a
coin counter in the HUD.

- Mechanism: item world-positions are laid out along the route and projected
  per frame; sprites are composited with distance scaling, contact shadows,
  and luminance/chroma harmonization. Pickup = ego-pose proximity to the
  item's world position.
- Integration seam: Robotaxi's FTheta marker projection replaces the
  standalone compositor's fitted pinhole camera (strictly better: exact
  fisheye + curves). Compositing happens in the presenter before encode.
- Cost: CPU-only, milliseconds per frame at 30 fps.

### 3. Random obstacle events (#431)

Sudden obstacles the driver must dodge: spawned vehicle clones driven
through the conditioning channel, or photoreal composited vehicles.

- Mechanism A (generative): template-clone spawning + box-axis guidance —
  a real actor track from the scene is cloned into the lane ahead; guidance
  makes moving clones fully opaque. Served as a session command.
- Mechanism B (composited): cutout vehicles pasted at box-projected
  positions (validated look: composited cars + optional renoise-refine).
- Integration seam: a game-event system rolls random encounters and calls
  either mechanism; collision scoring uses the spawned box positions.

### 4. Weather events (#431)

Sudden rain/snow via prompt swap on the text cross-attention KV cache,
optionally strengthened with 2x-cost guidance. Triggered randomly or by key.

## Plan and estimates

| step | est. |
|---|---|
| Bring-up on the #463 branch; locate presenter/runner hook seams | 0.5 day |
| Skin-switch ability behind `live_edit.style` flag + HUD binding | 1 day |
| Coin pickups via FTheta projection + HUD counter | 1 day |
| Obstacle events (spawn command + composite fallback) | 1–1.5 days |
| Weather events | 0.5 day |

Coordination: build against `dev/aidanf/game/crazy-robotaxi-standalone`,
rebase when the #460 API version lands. All abilities behind flags; no
changes to Robotaxi core defaults.

## Evidence (from #431 / #458)

- Style hold: side-by-side before/after videos (residential + highway) with
  the final v5 stack; flat post-swap divergence over 20 held chunks.
- Item compositing: lane-locked animated item courses on highway and
  residential clips (fitted camera incl. curvature), sprite harmonization +
  shadows; sigma≈0.47 renoise-refine for photoreal integration.
- Spawning: guided moving clones (opaque), placement-manifold findings, and
  honest limits (arbitrary-actor materialization is out of scope for the
  2-step student — documented in the spawn-SFT postmortem).
