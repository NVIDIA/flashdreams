# Crazy Robotaxi V2 performance investigation

This note separates measured facts from hypotheses for the V2 port. Raw JSON
and `/usr/bin/time` captures are local benchmark artifacts and are intentionally
not source files.

## Evidence from the recopied captures

The `crazy-native` and `crazy-webrtc` commands used the `perf` preset at
1280x704, seed 42, 32 visible blocks, and four hidden prewarm blocks. The
statistics below exclude steps 0-3 and report median / p90 over steps 4-31.

| Stage | Native window | WebRTC |
| --- | ---: | ---: |
| Pipeline wall | 436.0 / 439.9 ms | 430.8 / 436.8 ms |
| Engine wall | 54.9 / 59.3 ms | 53.0 / 59.3 ms |
| Rollout wall | 491.1 / 498.7 ms | 484.2 / 511.1 ms |
| Model-thread CPU | 485.5 / 491.5 ms | 477.2 / 486.3 ms |
| Effective 8-frame throughput from median rollout | 16.3 fps | 16.5 fps |

The model produces 8-frame steady-state chunks representing 266.7 ms at the
declared 30 fps. These captures therefore prove a throughput miss, independent
of presentation: a 484-491 ms producer cannot continuously supply 30 fps. The
native and WebRTC generation numbers are close, so WebRTC is not causing the
model-thread chunk gap in this comparison.

The process-wide `/usr/bin/time` captures averaged 136% CPU for native window
and 156% for WebRTC. The model metrics explain about one saturated core:
pipeline CPU time is approximately equal to pipeline wall time (about 430 ms
per chunk) even with synchronized stage profiling disabled. That timing
boundary surrounds only `pipeline.generate` and `pipeline.finalize`, so the
CPU use is not attributable to taxi rules, PhysX, or HUD code. The standalone
artifact does not record thread CPU and still needs a native stack sample to
confirm the same wait there. WebRTC adds roughly one third of a CPU core in
this pair of runs, consistent with presentation and video encoding work.

The synchronized `crazy-pipeline-perf` and `omnidreams-pipeline-perf` captures
provide the pipeline-alone versus pipeline-in-game comparison:

| Synchronized stage | Pipeline alone | Inside Crazy Robotaxi |
| --- | ---: | ---: |
| Encode median | 60.8 ms | 47.2 ms |
| Diffuse median | 362.2 ms | 354.5 ms |
| Decode median | 17.9 ms | 17.9 ms |
| Finalize median | 180.4 ms | 181.2 ms |
| Pipeline total median | 620.7 ms | 601.0 ms |

These independent runs do not prove a speedup inside the game, but they do
rule out a material pipeline regression caused by the Crazy Robotaxi wrapper.
The app adds a measured 52.8 ms median engine stage in the synchronized run.
In the lower-overhead native capture, the steady engine median is 54.9 ms:
about 40.3 ms simulation, 12.8 ms conditioning, and 1.6 ms rules.

The matched `original-perf` captures use 1168x640 and the original demo's
native FP8 DiT settings. They cut median rollout latency to 274.8 ms in the
native window and 272.4 ms over WebRTC, equivalent to 29.1 and 29.4 generated
fps. Their p90 latencies are 277.2 and 279.7 ms. This is a large improvement,
but every steady chunk remains above the 266.7 ms real-time budget. These files
exercise the V2 port with the matched preset; they are not captures from the
non-V2 application.

An older capture contains 127-147 ms CPU spikes from the superseded NumPy
waypoint rasterizer. Waypoints now use an ImGui background draw list, so that
specific hotspot no longer exists and must not be projected onto current runs.

## App-side BEV correction

The Dear ImGui BEV was rendered at 1024x1024, normalized to BF16 on the GPU,
converted back to uint8 on the UI thread, copied to the CPU, expanded to RGBA,
and uploaded to a SlangPy texture. Its default on-screen image is only 234x234
at 1280x704.

The corrected path caps the BEV renderer to the actual image extent while
preserving authored aspect ratio and keeps the renderer's native uint8 pixels
through `ConditionBatch`. This reduces default BEV pixels and upload bytes by
19.15x. A CPU sanity probe measured 0.727 ms for the old 1024x1024 BF16
materialization versus 0.061 ms for the new 234x234 uint8 layout conversion.
Those CPU numbers validate the direction only; GPU renderer, synchronization,
and UI timings still require a target-machine run.

The remaining D2H/H2D round trip is an API boundary: V2's ImGui pixel helper
accepts NumPy-like pixels and calls `texture.copy_from_numpy`. Eliminating it
cleanly requires the ImGui renderer to accept a CUDA tensor or expose a
CUDA-backed SlangPy texture registration hook. The app should not duplicate
the renderer's Vulkan/CUDA resource ownership to work around that missing API.

### CUDA-upload follow-up

The `*-3.json` native-window capture on commit `ece9c52` confirms that the BEV
change removed its original synchronization hotspot. BEV submission fell from
45.68 seconds total and 195.7 ms p90 to 0.32 seconds total and 0.31 ms p90.
UI-step p90 fell from 37.2 ms to 4.2 ms, presented throughput increased from
17.25 to 22.62 fps, and elapsed time after the first publication fell from
104.58 to 79.76 seconds.

The capture also exposes a second synchronization boundary. Native surface
acquisition consumed 22.15 seconds, with 222 of its 227 waits above 10 ms
occurring on a model-frame advance. Those waits cluster near 90--110 ms, once
per generated chunk. The app therefore records a CUDA event after each model
chunk and keeps UI conversion, ImGui interop, composition, and native upload on
a dedicated presentation stream. The stream waits only for the selected
chunk's event instead of making Vulkan depend on subsequently queued model
work. This follow-up remains pending a target-machine `*-4.json` validation and
is not yet a validated performance result.

## App-side PhysX bridge correction

The matched native `original-perf` capture attributes 39.2 ms per chunk to
simulation. PhysX accounts for 36.0 ms: 30.8 ms in the Python/native bridge,
4.0 ms in the solver, and less than 0.1 ms in actor update plus readback.
PhysX already owns a two-worker CPU dispatcher, so adding application worker
threads cannot materially reduce this path. The eight simulated frames are
also state-dependent and cannot run concurrently.

The dominant bridge path tested every active barrier through a Python loop on
every frame. The replacement caches contiguous barrier arrays whenever the
physics vicinity changes, evaluates the broad phase as one NumPy operation,
and applies collision responses sequentially only to the candidate contacts.
Candidate application retains authored barrier order, including corner
response behavior. A deterministic CPU parity sweep covers degenerate,
single-contact, and multi-contact geometry.

An isolated CPU probe with 293 barriers and eight frames measured 21.1 ms for
the scalar scan and 0.51 ms for the vectorized scan, a 41.5x speedup for that
component. This is not a target-GPU headline result. Runtime metrics now split
the bridge into traffic preparation, barrier rebound, traffic update, state
materialization, and residual adapter time so a fresh run can validate where
the saved time lands.

## Backpressure and visible latency

The reported `block` versus `drop_oldest` behavior is consistent with a UI
consumer falling behind the producer. `block` preserves every generated frame,
but V2 may retain the currently presented chunk plus two queued chunks. For
8-frame chunks at 30 fps, that permits roughly 800 ms of model-frame history
before transport latency. `drop_oldest` bounds staleness by evicting complete
chunks, which necessarily creates visible motion jumps.

The app-side BEV correction targets a concrete reason the UI consumer may miss
its cadence. If a matched rerun still fills the queue, the remaining fix is an
API policy rather than a game rule: V2 needs a configurable latency bound (or
frame-based queue capacity) that can keep at most the next ordered chunk
without dropping it. The earlier pre-generation capacity wait did not change
the two-chunk capacity and therefore could not establish that bound.

## Required post-optimization rerun

The original non-V2 demo uses `example_world_model_perf.yaml`. The
`original-perf` preset matches its important settings: 1168x640, native FP8
DiT with cuDNN attention, `skip_finalize_kv_cache`, and denoising timesteps
`[1000, 100]`. Use fresh processes and keep ordered presentation:

```bash
/usr/bin/time -v -o /tmp/crazy-original-native.time \
  uv run --no-sync flashdreams-run-v2 crazy-robotaxi \
  --mode native-window \
  --pixel-width 1168 \
  --pixel-height 640 \
  --backpressure-mode block \
  --presentation-mode only_present_newest \
  --stats-path /tmp/crazy-vectorized-native.json \
  -- \
  --model-preset original-perf \
  --seed 42 \
  --total-blocks 32 \
  --prewarm-blocks 4
```

Repeat with `--mode webrtc` and different artifact names. Do not enable
`--profile-pipeline` for headline throughput because its CUDA synchronization
changes the result. Run a separate diagnostic capture with that flag only when
stage attribution is needed.

Record GPU, driver, CUDA, PyTorch, cuDNN, checkpoint revisions, and compiler
cache state alongside the artifacts. Exclude at least steps 0-3, then report
median and p90. The remaining questions are:

1. Does `physx_barrier_rebound_s` fall near the isolated probe, and does total
   rollout latency move below the 266.7 ms chunk budget?
2. Does matched `original-perf` reach the original demo's measured producer
   throughput on the same machine?
3. After the BEV correction, how much UI-thread CPU remains in native and
   WebRTC modes?
4. If the pipeline still consumes nearly one full CPU core, which native or
   CUDA wait accounts for it? That requires a native stack sample or CPU
   profiler and belongs to the OmniDreams integration/API team rather than the
   game loop.

Current validation status: **Useful app-side optimization, pending target-GPU
measurement**. Physics response parity passes on CPU; the unvalidated part is
the magnitude of the full runtime gain.
