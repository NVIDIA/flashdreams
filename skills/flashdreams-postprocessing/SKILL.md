---
name: flashdreams-postprocessing
description: Add or modify FlashDreams video post-processing processors, sessions, presets, and runner stream wiring. Use when implementing a new VideoPostProcessorConfig / VideoPostProcessor / VideoPostProcessorSession, registering a --postprocess.preset entry point, changing VideoPostprocessStream behavior, or reasoning about streaming buffering, layouts, per-view processing, distributed execution, or postprocess tests.
---

# FlashDreams Post-Processing

Use this skill when adding a video post-processor or changing the runner
post-processing stream. The reference implementation is
`integrations/flashvsr/flashvsr/postprocess.py`.

## Mental Model

A post-processor is usually three classes, not one class inheriting everything:

- `VideoPostProcessorConfig`: serializable config and CLI surface. It sets
  `_target` to the processor factory and declares fields, `output_spec()`,
  `requires_all_ranks()`, and `validate_execution()`.
- `VideoPostProcessor`: lightweight factory created from config. Its job is
  `start(spec) -> VideoPostProcessorSession`.
- `VideoPostProcessorSession`: mutable per-stream runtime. It owns buffers,
  caches, lazy model instances, counters, and `process()` / `flush()`.

Keep stream state in the session. Do not store per-rollout mutable state on the
config or processor factory.

## Implementation Steps

1. Pick a home:
   - Generic reusable post-processing belongs under `flashdreams/flashdreams/infra/postprocess/`.
   - Model-specific processors belong in their integration, for example
     `integrations/<name>/<pkg>/postprocess.py`.

2. Define a config subclass:

   ```python
   @dataclass(kw_only=True)
   class MyPostProcessorConfig(VideoPostProcessorConfig):
       _target: type["MyPostProcessor"] = field(
           default_factory=lambda: MyPostProcessor
       )

       scale: int = 2

       def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
           return VideoSpec(
               height=input_spec.height * self.scale,
               width=input_spec.width * self.scale,
               fps=input_spec.fps,
               channels=input_spec.channels,
           )
   ```

   Override:
   - `output_spec()` when spatial size, channels, or timing changes.
   - `requires_all_ranks()` when the processor must run on nonzero ranks under
     `torchrun`.
   - `validate_execution()` to reject unsupported distributed or shape modes
     early.

3. Define the processor factory:

   ```python
   class MyPostProcessor(VideoPostProcessor[MyPostProcessorConfig]):
       def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
           return _MyPostProcessorSession(self.config, spec)
   ```

4. Define the session:

   ```python
   class _MyPostProcessorSession(VideoPostProcessorSession):
       def __init__(self, config: MyPostProcessorConfig, spec: VideoSpec) -> None:
           self._config = config
           self._spec = spec
           self._buffer: Tensor | None = None

       def process(self, chunk: VideoChunk) -> list[VideoChunk]:
           ...

       def flush(self) -> list[VideoChunk]:
           ...
   ```

   `process()` is synchronous but may return `[]`: that means it consumed the
   input chunk and is buffering frames until a later chunk or `flush()` can
   complete an output window.

5. Handle layouts at the boundary:
   - Accept `VideoChunk.tensor` in `chunk.layout`.
   - Use `to_bvtchw()` only as a generic boundary helper.
   - Convert once into the processor's native layout, make it contiguous if the
     model kernels require that, and keep internal buffers in that native
     layout.
   - Document any forced `.contiguous()` because it can copy.

6. Return `VideoChunk`s:
   - Preserve `[-1, 1]` value range unless the API is intentionally changed.
   - Set the correct `layout`.
   - Carry metadata only if it helps downstream processors or provenance.

7. Register presets when users should select it from CLI:

   ```toml
   [project.entry-points."flashdreams.postprocess_presets"]
   "my-postprocessor-v1" = "my_pkg.postprocess:POSTPROCESS_PRESET_MY_V1"
   ```

   The exported object must be a `VideoPostProcessorConfig`, for example:

   ```python
   POSTPROCESS_PRESET_MY_V1 = MyPostProcessorConfig(...)
   ```

   Users select it with `--postprocess.preset my-postprocessor-v1`.

## Runner Interaction

Runners create a `VideoPostprocessStream` through
`create_runner_postprocess_stream()`. The stream:

- creates one chain session for whole-stream processing, or one session per
  view when `postprocess_per_view=True`;
- calls `session.process(VideoChunk(...))` for each AR output;
- turns `[]` into a zero-frame tensor so `process()` remains tensor-only;
- skips collecting zero-time tensors in `_append_if_nonempty()`;
- calls `flush()` once at end-of-stream and appends any tail output.

Use `postprocess_output_layout` to describe the runner's decoded output layout.
Use `postprocess_per_view=True` for `bvtchw` outputs when each camera/view needs
an independent processor session.

## Tests

Add CPU-safe tests unless the behavior genuinely requires a GPU:

- Config/preset discovery: `flashdreams/tests/test_postprocess_presets.py`.
- Stream contract and buffering: `flashdreams/tests/test_postprocess_stream.py`.
- Processor-specific CPU fakes: `integrations/<name>/tests/test_postprocess.py`.
- Runner distributed skip/all-rank behavior:
  `flashdreams/tests/test_runner_postprocess.py`.

Every pytest test must use exactly one marker: `ci_cpu`, `ci_gpu`, or `manual`.
Prefer fake processor builders for CPU tests instead of loading checkpoints.

Useful focused validation:

```bash
uv run pytest flashdreams/tests/test_runner_postprocess.py \
  flashdreams/tests/test_postprocess_stream.py \
  flashdreams/tests/test_postprocess_presets.py \
  integrations/<name>/tests/test_postprocess.py
```
