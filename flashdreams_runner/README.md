# FlashDreams Runner

`flashdreams-runner` is the model-neutral shell for FlashDreams applications.
It loads an application from the active Python environment, creates its runtime,
initializes that runtime with a selected I/O mode, creates sessions, and drives
their generation loops.

```bash
uv run flashdreams-runner t2v-app mp4 --output o.mp4 --prompt "A waterfall"
uv run flashdreams-runner t2v-app replay --output o.mp4 --prompt "A waterfall"
uv run flashdreams-runner t2v-app webrtc --prompt "A waterfall"
uv run flashdreams-runner t2v-app none --steps 4 --prompt "A waterfall"
```

## Architecture

```text
application module              flashdreams-runner                 I/O mode
+------------------+       +--------------------------+       +---------------+
| create_runtime() | ----> | initialize Runtime       | ----> | Replay / MP4  |
|                  |       | create Session           |       | WebRTC        |
| Runtime          |       |                          |       | None          |
|  model weights   |       | loop:                    |       +-------+-------+
|  global state    |       |   input = mode.read()    |               |
|                  |       |   output = Session.      |<--------------+
| Session          |<------|       generate(input)    |
|  prompt/cache    |       |   mode.write(output)     |-------------->+
|  game state      |       | destroy Session/Runtime  |
+------------------+       +--------------------------+
```

The application owns inference. The runner owns orchestration and I/O.

## Application ABI

An installed application module exposes one function:

```python
create_runtime(arguments: flashdreams_runner.ApplicationArguments) \
    -> flashdreams_runner.Runtime
```

The function adds application-specific arguments to `arguments.parser`, calls
`arguments.parse_args()`, resolves application configuration, and returns an
uninitialized runtime. This single factory is the ABI between an application
package and `flashdreams-runner`.

`Runtime` owns model weights and other one-time or process-global state:

- `config` exposes the application's `AppConfig` to runner-owned modes.
- `initialize(device=..., io_handler=...)` performs model construction and
  one-time setup.
- `create_session(initial_input)` creates isolated per-user state.
- `destroy()` releases model and process resources.

`Session` owns the application loop implementation and all per-user state, such
as prompts, K/V caches, world state, and step counters:

- `generate(inputs)` performs exactly one application iteration and returns a
  `StepResult`.
- `destroy()` releases session-local resources.

The base classes also expose compatibility spellings for the shared
FlashDreams serving stack, so WebRTC consumes an application runtime directly
without a runner-specific adapter.

## I/O modes

Modes are runner-owned I/O handlers. They never construct model pipelines or
implement application generation logic.

| Mode | Input/output behavior |
|---|---|
| `mp4` | Compatibility name for a finite replay written to MP4. |
| `replay` | Runs a finite deterministic input sequence and writes MP4. |
| `webrtc` | Creates a live server and one application session per admitted client. |
| `none` | Runs a finite input sequence and discards output. |

`--steps` overrides the finite iteration count for `mp4`, `replay`, and `none`.
If omitted, those modes use `Runtime.config.default_steps`. Native-window and
additional interactive handlers can implement the same `IOHandler` boundary.

## Minimal application

```python
from flashdreams.runtime import InferenceInput, StepResult
from flashdreams_runner import (
    ApplicationArguments,
    AppConfig,
    IOHandler,
    Runtime,
    Session,
)


class MySession(Session):
    def __init__(self, model, prompt: str) -> None:
        self.model = model
        self.prompt = prompt
        self.cache = model.create_cache(prompt)
        self._step_index = 0

    @property
    def step_index(self) -> int:
        return self._step_index

    def generate(self, inputs: InferenceInput) -> StepResult:
        video = self.model.generate(self.cache, inputs.step)
        result = StepResult.from_video_chunk(
            step_index=self._step_index,
            video_chunk=video,
            layout="tchw",
        )
        self._step_index += 1
        return result

    def destroy(self) -> None:
        self.cache = None


class MyRuntime(Runtime):
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.model = None

    @property
    def config(self) -> AppConfig:
        return AppConfig(
            model_id="my-app",
            fps=24,
            output_layout="tchw",
            video_width=832,
            video_height=480,
            default_steps=4,
        )

    def initialize(self, *, device: str, io_handler: IOHandler) -> None:
        del io_handler
        self.model = load_model(device)

    def create_session(self, initial_input=None) -> Session:
        return MySession(self.model, self.prompt)

    def destroy(self) -> None:
        self.model = None


def create_runtime(arguments: ApplicationArguments) -> Runtime:
    arguments.parser.add_argument("--prompt", required=True)
    options = arguments.parse_args()
    return MyRuntime(prompt=options.prompt)
```
