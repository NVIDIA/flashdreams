# FlashDreams Cam2V application

Reusable interactive camera-to-video application infrastructure. Model packages
provide adapters under `integrations_v2/<model>/apps/cam2v` and register an
application named `cam2v-<model-config-name>`.

## Controls

| Keys | Action |
| --- | --- |
| `W` / `S`, `Up` / `Down` | Move forward / backward |
| `A` / `D`, `J` / `L` | Yaw left / right |
| `Q` / `E` | Strafe left / right |
| `I` / `K` | Pitch up / down |

Losing browser focus clears held keys.

## Usage

Concrete launch commands live with each model adapter:

- [Lingbot](../../integrations_v2/lingbot/apps/cam2v/README.md)
- [HY-WorldPlay](../../integrations_v2/hy_worldplay/apps/cam2v/README.md)

The general command shape is:

```bash
uv run --no-sync flashdreams-run-v2 cam2v-<model-config-name> \
  [runtime arguments] -- [application arguments]
```

Runtime arguments, before `--`:

| Argument | Purpose |
| --- | --- |
| `-h`, `--help` | Show runtime help and installed application names |
| `--mode {mp4,webrtc,native-window}` | Select file, browser, or native-window output |
| `--stats-path PATH` | Write model-step measurements as JSON |
| `--output-path PATH` | MP4 destination; required in `mp4` mode |
| `--host HOST`, `--port PORT` | WebRTC bind address |
| `--window-title TITLE` | Native-window title |
| `--pixel-width INT`, `--pixel-height INT` | Override output dimensions |
| `--fps INT` | Override output frame rate |
| `--layout {tchw,btchw,bcthw,bvtchw}` | Override output tensor layout |
| `--backpressure-mode {block,drop_oldest}` | Select presentation-queue behavior |
| `--presentation-mode {on_demand,continuous}` | Select UI update behavior |

Cam2V application arguments, after `--`:

| Argument | Purpose |
| --- | --- |
| `-h`, `--help` | Show model application help and defaults |
| `--prompt TEXT`, `--prompt-path PATH` | Set the text prompt directly or from a file |
| `--image-path PATH` | Set the first frame |
| `--pose-path PATH` | Set a pose trace used by the model adapter |
| `--intrinsic-path PATH` | Set camera calibration |
| `--world-scale FLOAT` | Set the camera-motion scale |
| `--example-data`, `--no-example-data` | Enable or disable packaged/example inputs |
| `--example-idx INT` | Select an example input |
| `--device DEVICE` | Select the model device |
| `--total-blocks INT` | Set autoregressive chunks per rollout |
| `--warmup-blocks INT` | Set chunks excluded from steady-state FPS |
| `--ui`, `--no-ui` | Enable or disable the controls/status overlay |
| `--compile`, `--no-compile` | Enable or disable model compilation |
| `--seed INT` | Override the diffusion seed |

For UI development without loading a model:

```bash
uv run --no-sync flashdreams-run-v2 cam2v-dummy --mode webrtc \
  --host 0.0.0.0 --port 8089 -- \
  --step-wait-seconds 0.9 --frames-per-chunk 12
```

The dummy-only arguments are `--step-wait-seconds FLOAT` and
`--frames-per-chunk INT`.

## Tests

```bash
uv sync --package flashdreams-cam2v --group test --inexact
uv run --no-sync pytest apps/cam2v/tests -m ci_cpu
```

Run `-m ci_gpu` instead for the CUDA UI-composition test.
