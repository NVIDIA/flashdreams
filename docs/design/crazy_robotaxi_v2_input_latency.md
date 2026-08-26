# Crazy Robotaxi V2 input-latency handoff

## Status

Crazy Robotaxi preserves timestamped input transitions on the application side
and offers opt-in UI-to-model-frame diagnostics. The remaining held-input
latency cannot be removed through the public application interfaces alone.
Shared V2 runtime changes are intentionally deferred to the API and
architecture owners.

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
- `--presentation-mode only_present_new` did not materially improve the held
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

## V2 transport boundary

The current V2 WebRTC client window has a second, independent latency risk:

- Its private video track uses an unbounded `asyncio.Queue` and appends every
  submitted frame.
- `SessionDesc.backpressure_mode` configures `PresentationManager`, but does
  not configure or bound the WebRTC queue.
- Every UI result is converted to CPU RGB and encoded by aiortc's software VP8
  path. The app cannot select the encoder, inspect queue depth or frame age,
  flush stale frames, or observe browser presentation.
- The existing non-V2 WebRTC serving stack already contains bounded tracks,
  stale-delivery flushing, pacing re-anchoring, delivery metrics, and optional
  NVENC. V2 does not currently use those facilities.

The application cannot safely work around this boundary without constructing
or reaching into a private client window, which would violate V2 ownership and
couple the game to one presentation mode.

## Requested API decisions

1. Define interactive transport backpressure end to end. Either propagate the
   session policy through the client window or introduce a separate transport
   policy with a bounded low-latency mode.
2. Reuse the shared WebRTC encoder/bridge abstractions, including optional
   NVENC, rather than maintaining a second unbounded software-only path.
3. Expose encoder backend, conversion/encode time, queue depth, dropped or
   flushed frames, and estimated frame age through runtime metrics.
4. Specify the input contract for long synchronous model steps. Sub-generation
   response requires smaller chunks, cancellation/preemption, or an explicit
   asynchronous conditioning mechanism; event polling alone cannot modify an
   in-flight conditioned chunk.
5. Keep ordered/blocking delivery available for file and quality workflows,
   while making the interactive low-latency policy explicit rather than
   silently accumulating stale frames.

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
