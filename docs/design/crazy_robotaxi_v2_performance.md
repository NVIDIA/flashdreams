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
| Pipeline wall | 420.0 / 421.4 ms | 423.8 / 434.8 ms |
| Engine wall | 54.0 / 57.0 ms | 54.1 / 57.5 ms |
| Rollout wall | 473.7 / 477.8 ms | 477.0 / 491.7 ms |
| Model-thread CPU | 469.4 / 472.0 ms | 474.0 / 487.2 ms |
| Effective 8-frame throughput from median rollout | 16.9 fps | 16.8 fps |

The model produces 8-frame steady-state chunks representing 266.7 ms at the
declared 30 fps. These captures therefore prove a throughput miss, independent
of presentation: a 474-477 ms producer cannot continuously supply 30 fps. The
native and WebRTC generation numbers are close, so WebRTC is not causing the
model-thread chunk gap in this comparison.

The process-wide `/usr/bin/time` captures averaged 126% CPU for native window
and 160% for WebRTC. The model metrics explain about one saturated core:
pipeline CPU time is approximately equal to pipeline wall time (about 420 ms
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
| Encode median | 60.2 ms | 47.9 ms |
| Diffuse median | 358.7 ms | 351.8 ms |
| Decode median | 17.9 ms | 17.9 ms |
| Finalize median | 179.1 ms | 179.4 ms |
| Pipeline total median | 616.0 ms | 596.5 ms |

These independent runs do not prove a speedup inside the game, but they do
rule out a material pipeline regression caused by the Crazy Robotaxi wrapper.
The app adds a measured 55.8 ms median engine stage in the synchronized run.
In the lower-overhead native capture, the steady engine median is 54.0 ms:
about 39.0 ms simulation, 13.1 ms conditioning, and 1.5 ms rules.

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

## Required like-for-like rerun

The existing native/WebRTC captures do **not** test the new `original-perf`
preset. The original non-V2 demo uses `example_world_model_perf.yaml`, whose
important differences include 1168x640, native FP8 DiT with cuDNN attention,
`skip_finalize_kv_cache`, and denoising timesteps `[1000, 100]`. Use fresh
processes and keep ordered presentation for this comparison:

```bash
/usr/bin/time -v -o /tmp/crazy-original-native.time \
  uv run --no-sync flashdreams-run-v2 crazy-robotaxi \
  --mode native-window \
  --pixel-width 1168 \
  --pixel-height 640 \
  --backpressure-mode block \
  --presentation-mode only_present_newest \
  --stats-path /tmp/crazy-original-native.json \
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

1. Does matched `original-perf` reach the original demo's measured producer
   throughput on the same machine?
2. After the BEV correction, how much UI-thread CPU remains in native and
   WebRTC modes?
3. If the pipeline still consumes nearly one full CPU core, which native or
   CUDA wait accounts for it? That requires a native stack sample or CPU
   profiler and belongs to the OmniDreams integration/API team rather than the
   game loop.

Current validation status: **Useful app-side correction, pending target-GPU
measurement**. Ordered frames and the model conditioning path are unchanged;
the unvalidated part is the magnitude of the runtime gain.
