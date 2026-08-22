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
