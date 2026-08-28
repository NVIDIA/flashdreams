# Crazy Robotaxi V2 input-latency handoff

## Status

Crazy Robotaxi uses the V2 runtime's shared real-time input timeline and
keyboard-state track, and offers opt-in UI-to-model-frame diagnostics. The V2
WebRTC sender is bounded and drops stale unsent frames under congestion. The
remaining held-input latency is bounded by synchronous autoregressive model
steps: input cannot change a chunk whose inference is already in flight.

## Reproduction and evidence

Run the app at its trained 1280×704, 30 fps output contract:

```bash
uv run flashdreams-run-v2 crazy-robotaxi --mode webrtc \
  --stats-path /tmp/crazy-robotaxi-latency.json -- \
  --model-preset perf \
  --profile-input-latency
```

The investigation observed:

- A held steering key took more than 1.5 seconds to affect displayed model
  video across model presets.
- A 100–200 ms tap was often absent before timestamped transition reduction
  was added.
- `--presentation-mode on_demand` did not materially improve the held
  steering delay, so repeated UI frames were not its dominant cause.
- A supplied synchronized diagnostic run produced eight-frame chunks covering
  266.7 ms of video in 733.6–788.7 ms after warmup, averaging 760.1 ms. Pipeline
  profiling perturbs CPU behavior, so these numbers describe that captured
  stack rather than a universal model benchmark.

V2 reads model events immediately before calling the synchronous model-loop
`step`. An input arriving during the measured 760 ms step therefore waits for
that step to finish and for the next conditioned step to finish. This creates
an application/model response window of roughly 760–1520 ms before final
WebRTC delivery. Smaller or preemptible model work is required to reduce that
floor.

## V2 transport behavior

The V2 WebRTC client window now materializes UI frames at `window.write()`,
uses a bounded two-frame sender queue, and replaces the oldest unsent frame
during congestion. CUDA events order model output, UI composition, and WebRTC
transfer work without device-wide synchronization. Continuous presentation
can redraw input-responsive UI while model inference is still running, but it
cannot alter the already-conditioned model chunk.

## Remaining API decision

Specify the input contract for long synchronous model steps. Sub-generation
response requires smaller chunks, cancellation/preemption, or an explicit
asynchronous conditioning mechanism; event polling alone cannot modify an
in-flight conditioned chunk.

## Runtime acceptance tests

- Exercise real 1280×704 content at 30 fps, not only the existing 16×16 smoke
  test.
- Simulate a slow encoder or consumer and prove the interactive queue remains
  bounded, stale work is flushed according to policy, and frame age recovers
  after a stall.
- Record browser-key-to-runtime-event and UI-write-to-browser-display latency;
  target p95 transport overhead below 100 ms on a local loopback after warmup.
- Report best, average, p90, and worst latency together with encoder backend,
  resolution, host, GPU, driver, and warmup policy.
- Verify reset and disconnect discard stale media without replaying frames from
  the previous generation.
