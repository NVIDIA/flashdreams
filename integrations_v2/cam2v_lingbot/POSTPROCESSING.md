<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lingbot post-processing and multi-GPU design

This note records the implementation on `dev/gtong/session-management`, the
single-GPU validation procedure, and the proposed path to overlap Lingbot on
one GPU with FlashVSR on another.

## Current implementation

`Cam2VApplication` resolves an optional postprocess preset and owns both the
Lingbot pipeline and one `VideoPostprocessStream`. The objects live from the
first session that needs them until `IApplication.close()`. A rollout reset or
checkbox transition calls `VideoPostprocessStream.reset()`, which clears only
buffered frames, metadata, the FlashVSR AR index, and nested temporal cache
bookkeeping. It preserves the loaded FlashVSR pipeline, weights, compiled
network, CUDA-graph-bound cache storage, and cache object identity.

The **Post-processing** checkbox sends a message from the UI loop to the model
loop. The transition is applied at a model-step boundary, so the UI thread does
not mutate the postprocessor while it is running. It does not request a new
session.

The configured presentation size is computed from the postprocessor's
`output_spec`. Lingbot still generates and conditions at 832x464, while the
FlashVSR 2x preset presents 1664x896 (FlashVSR aligns the output down to a
multiple of 128). When the checkbox is off, the UI scales the raw frame to the
same presentation dimensions, so WebRTC/native-window output geometry does not
change mid-session. The raw dimensions are retained in `SessionDesc.metadata`
so a browser-driven replacement session cannot accidentally feed the
presentation size back into Lingbot as its next model size.

### Frame cadence

With `--postprocess-chunk-size 8`, FlashVSR consumes 5 frames on its first call
and 8 on every later call. Lingbot emits 9 frames for AR 0 and 12 thereafter.
The stream adapter buffers the remainder and produces this repeating pattern:

| Lingbot AR | Input | Buffered before | FlashVSR calls | Output | Buffered after |
|---:|---:|---:|---|---:|---:|
| 0 | 9 | 0 | 5 | 5 | 4 |
| 1 | 12 | 4 | 8 + 8 | 16 | 0 |
| 2 | 12 | 0 | 8 | 8 | 4 |
| 3 | 12 | 4 | 8 + 8 | 16 | 0 |

At a normal rollout start, AR 0 supplies the real 5-frame cold-start call. If
post-processing is enabled in the middle of a rollout, the next model chunk is
exactly 12 frames; the adapter starts with a real 5-frame call and retains the
remaining 7. If another producer starts a reset stream with exactly 8 frames,
the adapter primes the required 5-frame decoder state from the first frame and
discards only that synthetic prime output before processing the real 8 frames.
End-of-rollout partial input is replicate-padded for execution and trimmed so
the total visible frame count is preserved.

## Run and benchmark

Interactive WebRTC:

```bash
uv sync --package flashdreams-cam2v-lingbot --inexact
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
    --mode webrtc --host 0.0.0.0 --port 8089 -- \
    --example-data \
    --postprocess-preset flashvsr-v1.1-sparse-1.5 \
    --postprocess-chunk-size 8
```

Use `--no-postprocess-compile` for a development smoke test that should not pay
Triton/Inductor autotuning time. It is not representative of rollout
performance.

For comparable measurements, use the same input and number of blocks and ask
the runtime for JSON metrics. `drop_oldest` prevents a slow software MP4
encoder from blocking on the one-chunk presentation queue, although a
device-wide profiling synchronization can still include same-GPU presentation
work. Use `block` for a frame-exact artifact.

```bash
# Baseline
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
    --mode mp4 --output-path /tmp/lingbot-baseline.mp4 \
    --stats-path /tmp/lingbot-baseline.json \
    --backpressure-mode drop_oldest --presentation-mode on_demand -- \
    --example-data --total-blocks 10 --warmup-blocks 4

# FlashVSR
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
    --mode mp4 --output-path /tmp/lingbot-flashvsr.mp4 \
    --stats-path /tmp/lingbot-flashvsr.json \
    --backpressure-mode drop_oldest --presentation-mode on_demand -- \
    --example-data --total-blocks 10 --warmup-blocks 4 \
    --postprocess-preset flashvsr-v1.1-sparse-1.5 \
    --postprocess-chunk-size 8 --postprocess-profile
```

Interpret these metrics separately:

- `model_step_wall_s`: synchronized Lingbot generation only.
- `postprocess_step_wall_s`: synchronized wall time after Lingbot completion,
  including a final tail flush when applicable.
- `model_loop_wall_s`: current serial critical path, generation plus
  post-processing.
- `postprocess_s`: FlashVSR CUDA-event time for `process()` only when
  `--postprocess-profile` is set. The model reports milliseconds and the JSON
  metrics sink normalizes it to seconds.
- `postprocess_output_frames`: the variable 5/8/16 output cadence above.

Discard compilation/capture steps and report both the 8-output and 16-output
steady cases. Averaging them without weighting by output frames hides the
cadence cost. Also record peak GPU memory and whether MP4/WebRTC presentation
was active.

### 2026-09-03 single-GPU result

The implementation was exercised on one NVIDIA GB300 with the sparse-1.5
preset, compiled/CUDA-graph mode, 832x464 Lingbot input, and 1664x896 FlashVSR
presentation. The baseline used AR 3--9; the table reports its median. The
postprocess run needed AR 0--3 to compile/capture every alternating cadence, so
AR 4 and the FlashVSR portion of AR 5 are the first reusable steady samples.

| Case | Lingbot | FlashVSR | Serial step | Output | Effective output rate |
|---|---:|---:|---:|---:|---:|
| No postprocess (median) | 1.004 s | - | 1.004 s | 12 | 11.95 fps |
| FlashVSR, one 8-frame window (AR 4) | 0.979 s | 0.287 s | 1.267 s | 8 | 6.31 fps |
| FlashVSR, two-window cadence (AR 5) | presentation-contaminated | 0.300 s | invalid | 16 | invalid |

Using the baseline median model time for both halves of the alternating
8/16-output pair gives a projected serial steady rate of
`24 / (2 * 1.004 + 0.287 + 0.300) = 9.24 fps`. This is an inference from the
isolated stage timings, not a measured overlapped rate.

The AR-5 Lingbot measurement was 31.48 s because the benchmark's device-wide
synchronization also waited for the same-GPU presentation/MP4 path. It is not a
model regression and is excluded. The compiled FlashVSR event remained 0.300 s.
This contamination is itself evidence for giving FlashVSR and presentation
separate devices/streams and for collecting production metrics asynchronously.
The process held both model pipelines concurrently; sampled device memory rose
from roughly 111 GiB for Lingbot to about 137 GiB with FlashVSR resident.

The completed model path emitted `5, 16, 8, 16, 8, 16` frames and the output
artifact probed as H.264 at 1664x896. The long compile run caused continuous
presentation to repeat frames into the software encoder; the encoder was
interrupted after all six model blocks completed, so the JSON inference metrics
are complete but the MP4 duration is not a useful latency measurement.

A separate two-block smoke run used eager execution, `on_demand`, and the
normal (non-profiling) path. It exited cleanly at its interactive-session
timeout, recorded model chunks with 5 and 16 postprocessed frames, and produced
a playable 1664x896 H.264 file. The interactive runtime intentionally remains
open after model generation so it can continue presenting and accept input;
therefore even this MP4 repeats the terminal frame until `--timeout` and is a
functional artifact, not a throughput artifact.

## Runtime architecture findings

The v2 runtime intentionally has two long-lived execution threads:

1. The model thread calls `IModelLoop.step()` and publishes a list of
   `StepResult`s.
2. The UI thread polls input, takes model chunks from `PresentationManager`,
   selects one frame per presentation tick, composites the UI, and writes to
   the client window.

`PresentationManager` has one active chunk and a bounded one-chunk publish
queue. `block` applies producer backpressure; `drop_oldest` preserves
interactivity at the cost of output chunks. A `StepResult` records a CUDA event
after its output is submitted, allowing another CUDA stream to consume it
without a host synchronization. The presentation manager already uses a
separate high-priority CUDA stream.

Today Cam2V invokes `VideoPostprocessStream.process()` inside the model step,
after synchronizing Lingbot, then waits for the final output before returning.
That makes the runtime presentation clock follow the combined Lingbot and
FlashVSR critical path. Benchmark profiling additionally records FlashVSR CUDA
event timing. The two models execute serially on the same GPU. Moving a
synchronous FlashVSR call to the UI thread would freeze input and rendering, so
that is not an acceptable overlap mechanism.

The existing FlashVSR gRPC uplift server is useful for a remote service: it has
bounded receive/GPU/send queues and one cache per stream. It is not the right
same-host fast path because requests are converted through NumPy and raw RGB or
JPEG byte payloads, and returned frames are materialized on the CPU. The local
runtime should keep tensors on CUDA and use peer-to-peer copies.

## Proposed two-GPU pipeline

Keep the two-thread runtime. The model thread can enqueue work asynchronously
on two devices and let CUDA provide the overlap:

```text
model thread       GPU 0                         GPU 1
------------       ---------------------------   ----------------------------
step N             enqueue Lingbot N
                   record generated[N] event
postprocess N      P2P copy waits on event  ---> enqueue FlashVSR N
publish N                                         record uplifted[N] event
step N+1           enqueue Lingbot N+1            execute FlashVSR N
UI thread                                         wait on uplifted[N], present
```

The steady-state stage time becomes approximately
`max(lingbot_time, p2p_copy + flashvsr_time, presentation_time)` instead of the
current sum, once the pipeline is full.

### API changes

1. Split heavy resources from rollout state in the generic postprocess API.
   `VideoPostProcessorRuntime` owns weights, compiled code, CUDA graphs, device,
   and an explicit execution stream. `start_rollout()` returns lightweight
   temporal state (buffer spans, AR index, encoder/transformer/decoder cache).
   The current in-place `reset()` is the compatibility step toward this split.

2. Add an asynchronous stream operation:

   ```python
   pending = postprocess_stream.submit(
       output,
       autoregressive_index=step_index,
       producer_event=model_ready_event,
   )
   # pending.tensor exists immediately; pending.ready_event orders consumers.
   ```

   `submit()` must not synchronize either device. It schedules a non-blocking
   copy on the FlashVSR stream, waits there on the GPU-0 producer event, runs
   FlashVSR, and records a GPU-1 completion event. `StepResult` should accept an
   existing readiness event rather than always recording one on an implicit
   current stream.

3. Add `--postprocess-device` (for example `cuda:1`) and resolve it independently
   from the model device. Do not reuse the current torchrun `LOCAL_RANK`
   override for this single-process, two-device topology. Validate device
   existence and P2P access before loading the second model.

4. Remove the model-device host synchronization in asynchronous mode. Record a
   GPU-0 event after Lingbot output, submit FlashVSR on GPU 1, publish the
   pending result, and let the next model iteration enqueue on GPU 0. Keep the
   current synchronized path behind the benchmark/profiling option.

5. Extend async metrics. CUDA elapsed time is valid only after the completion
   event fires, while the metrics sink currently writes at publish time. Add a
   delayed metrics record or completion callback for `generation_gpu_ms`,
   `p2p_ms`, `postprocess_gpu_ms`, queue wait, and end-to-end chunk latency.
   Never synchronize production work merely to populate metrics.

6. Make source and presentation specs first-class. The Cam2V metadata bridge
   currently distinguishes 832x464 model input from 1664x896 output locally.
   A generic runtime contract should expose `model_video_spec` and
   `presentation_video_spec` so other applications and no-UI sinks do not need
   an application-specific metadata key.

### Buffering, backpressure, and toggle safety

- Preallocate a two-slot GPU-1 input ring for the 12-frame Lingbot cadence.
  If CUDA peer access is available, copy directly. Otherwise use a bounded ring
  of pinned host staging buffers; never silently fall back to pageable copies.
- Use `block` for the postprocess benchmark and frame-exact output. Dropping a
  submitted chunk cannot cancel its GPU work and would break FlashVSR temporal
  continuity. A future live mode may drop only *before* submission and must
  reset temporal state at that boundary.
- Apply checkbox changes only between model submissions. Before resetting the
  temporal cache, wait for the last submitted FlashVSR event (not a device-wide
  synchronize), discard pending presentation chunks by generation, then reset
  in place. This keeps toggles fast without racing cache mutation against GPU-1
  kernels.
- Keep output geometry fixed for a session. Bypass frames are scaled to the
  presentation spec; enabling FlashVSR must never reopen WebRTC or recreate the
  native window.

### Delivery sequence and acceptance checks

1. Land the current resident single-GPU implementation and cadence/reset tests.
2. Add independent device selection and P2P capability diagnostics.
3. Add `submit()` plus externally supplied `StepResult` readiness events; test
   ordering with two CUDA streams before involving two devices.
4. Enable the two-device Cam2V path and add generation tokens around toggles and
   rollout resets.
5. Benchmark warm steady state on a real two-GPU host. Confirm Nsight Systems
   shows Lingbot N+1 overlapping FlashVSR N, GPU memory remains flat across ten
   resets/toggles, output frame count equals input frame count, and the same
   FlashVSR pipeline/cache allocation identities persist for the application
   lifetime.
