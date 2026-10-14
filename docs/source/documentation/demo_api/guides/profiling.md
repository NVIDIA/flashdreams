---
title: 'Profiling a demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 Praneeth Samineni. -->

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

Nsight Systems shows the model thread, the UI thread and the GPU on one
timeline, so you can see what each of them was doing at the same moment. If you
only need numbers for each step, such as stage timings, frame rates and input
latency, use `--stats-path` instead.

## Collecting a report

First install the model package, then launch the app through
`flashdreams-profile`:

```bash
uv sync --package flashdreams-self-forcing
uv run flashdreams-profile -- t2v-self-forcing-wan2.1-t2v-1.3b --timeout 300 \
    --output-path artifacts/run.mp4 -- --prompt "A city street at night" --total-blocks 7
```

If the model takes longer to compile, increase `--timeout`.

The report is saved in `artifacts/profiles/`. To save it somewhere else, pass
`--report-path`. By default the report records CUDA, NVTX, OS runtime and
Vulkan activity; to record something different, pass `--trace`.

To time each stage, `--stats-path` (or `FLASHDREAMS_SYNC_AND_PROFILE=1`)
makes the CPU wait for the GPU twice per step. This adds a slight delay to each
step, so the timeline differs slightly from a normal run.

## Reading the report

To read the report, open the `.nsys-rep` file in the Nsight Systems GUI. For a
quick summary in the terminal instead, run:

```bash
nsys stats --report nvtx_sum artifacts/profiles/<report>.nsys-rep
```

Each row of the summary is one of the ranges below, with how many times it ran
and how long it took. These durations are CPU time and include waiting. For the
GPU work that each range launched, use `--report nvtx_gpu_proj_sum` instead.

| Range | Meaning |
| --- | --- |
| `model.pace` | Waiting to hold the step rate. |
| `model.step[i]` | One autoregressive step. |
| `model.publish` | Handing the chunk to presentation. |
| `pipeline.encode` / `diffuse` / `decode` | Encode, denoising loop, VAE decode. |
| `pipeline.finalize` | KV-cache update for the next step. |
| `ui.step` | One UI loop iteration. |
| `input.wait` | From an input arriving to the first frame that shows it. |

Besides these ranges, the timeline has a mark each time a frame is generated
(`model.frame`) or shown (`present.frame`). Each mark includes the step that
produced the frame, such as `present.frame [step 5]`, so you can follow a frame
on screen back to the step that made it.

FlashVSR and SwiftVR use their own `generate` method, so their reports have no
`pipeline.*` ranges.

## Measuring source-input host latency

Run the demo with its existing `--stats-path artifacts/stats.json` option.
This enables `InputLatencyProfiler` alongside stage timings and frame rates.
Programmatic sessions can select the same backend through
`FLASHDREAMS_INPUT_LATENCY=1`, then pass a `MetricsOutputSink` to the runner.

The input source supplies the monotonic origin for `UserInputEvent.timestamp`,
which counts microseconds from the session start. Native-window uses its input
clock. WebRTC uses the server receipt clock and rebases it for each replacement
session. A custom source exposes that origin through `TimestampedInputSource`.
Sources without a shared clock origin leave the two source-input metrics absent.

| Profiler metric | Host checkpoint |
| --- | --- |
| `input.ui_claim_ms` | The UI loop begins a run that claims the input. |
| `input.window_write_ms` | The first successful `window.write` return after that claim, within the same reset generation. |

Both metrics average all inputs completed since the previous collection.
`_max_ms` and `_count` report that collection's maximum and count. The
`_avg_ms`, `_session_max_ms` and `_session_count` forms cover the session.
Integer timestamp sums preserve precision and keep pending storage constant
even when many UI ticks produce no frame. A reset discards pending inputs from
the preceding generation. A failed write leaves completion unreported.

These samples reach the next model record or the final session summary. Their
collection interval can span earlier model steps. `MetricsOutputSink` converts
`_ms` names and values to `_s`, so the file contains `input.ui_claim_s` and
`input.window_write_s`. Session metadata identifies the window type, both frame
rates, resolution, presentation mode and backpressure mode. Reopening the stats
sink keeps the existing behavior of retaining the final replacement session.

The existing `input.latency_ms`, `input.queue_ms` and `input.present_ms` follow
runtime receipt, model consumption and frame selection. Source-input host
metrics use the separate checkpoints above. Selecting a model frame precedes
the UI's actual window write. The UI may also write a held pre-input frame.

Native-window completion includes the presenter call. WebRTC completion covers
host-side output handling and sender-mailbox admission when a track exists.
Transport delivery, browser decode, physical display and a causal response to
the input require their own telemetry. Profiling adds host work, and
`--stats-path` also enables synchronized CUDA stage timing. Compare runs with
the same profiling settings.

## Using your own profiler

A session runs with the profiler the application runner creates. To use your
own, assign it at the start of `init()`, before registering a loop:

```python
def init(self):
    self._profiler = MyProfiler()
```

The session's loops get it as `self.profiler`, so a model loop can record its
own ranges and events:

```python
with self.profiler.range("my_stage"):
    ...
self.profiler.event("my_event")
```

To profile a pipeline on its own, see
[Profiling a pipeline](../../inferencing_api/guides/profiling.md).
