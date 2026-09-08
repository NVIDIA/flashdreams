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
- [SANA-WM](../../integrations_v2/sana_wm/apps/cam2v/README.md)

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
| `--sync-and-profile`, `--no-sync-and-profile` | Enable or disable synchronized per-stage pipeline timings |
| `--seed INT` | Override the diffusion seed |

### Defaults

Defaults supplied by each registered Cam2V application:

| Setting | `cam2v-lingbot` | `cam2v-hy-worldplay` | `cam2v-sana-wm-streaming` | `cam2v-dummy` |
| --- | --- | --- | --- | --- |
| Output size | 832 x 464 | 1280 x 704 | 1280 x 704 | 640 x 360 |
| Model FPS | 16 | 16 | 16 | 16 |
| UI FPS | 60 | 60 | 60 | 60 |
| `--device` | `cuda` | `cuda` | `cuda` | `cuda` |
| `--total-blocks` | 20 | 20 | 10 | 10,000 |
| `--warmup-blocks` | 5 | 5 | 5 | 1 |
| `--world-scale` | Unset; inferred from `--pose-path` | 2.5 | 1.0 | Fixed at 1.0 |
| Output layout | `tchw` | `tchw` | `tchw` | `tchw` |
| Backpressure | `block` | `block` | `block` | `block` |
| Presentation | `continuous` | `continuous` | `continuous` | `continuous` |

For all four applications, the controls/status UI is enabled,
`--example-data` is disabled, and `--example-idx` is 0. Prompt and input paths
default to unset. HY-WorldPlay falls back to its built-in prompt and computed
intrinsics; Lingbot requires an image and intrinsics, and also a pose trace when
`--world-scale` is not given. SANA-WM can download its official image and prompt
with `--example-data`; intrinsics
default to a 90-degree horizontal field of view. `--compile`,
`--sync-and-profile`, and `--seed` default to the selected pipeline
configuration values.

For UI development without loading a model:

```bash
uv run --no-sync flashdreams-run-v2 cam2v-dummy --mode webrtc \
  --host 0.0.0.0 --port 8089 -- \
  --step-wait-seconds 0.9 --frames-per-chunk 12
```

The dummy-only arguments are `--step-wait-seconds FLOAT` (default 0.9) and
`--frames-per-chunk INT` (default 12).

## Tests

```bash
uv sync --package flashdreams-cam2v --group test --inexact
uv run --no-sync pytest apps/cam2v/tests -m ci_cpu
```

Run `-m ci_gpu` instead for the CUDA UI-composition test.
