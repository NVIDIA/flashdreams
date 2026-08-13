# FlashDreams text-to-video app

A model-neutral text-to-video shell built on the shared
`flashdreams.runtime.demo` session lifecycle. Each integration owns its pipeline
and checkpoint configuration and registers a slug under the
`flashdreams.applications` entry-point group; this app owns only prompt input,
output selection, and the thin runtime adapter.

## Setup

Install the workspace from the repo root, which registers every application
slug below, then set a token for checkpoint downloads:

```bash
uv sync --extra runners
export HF_TOKEN=<your-hf-token>
```

Developers and maintainers should use `uv sync --extra dev --extra runners`
instead. To build only one model's dependencies rather than the whole workspace,
see that model's page under `docs/source/models/`.

## Available applications

Every slug below launches through `flashdreams-run` and carries its model's
defaults, so a bare invocation needs no flags.

| Slug | Preset | Blocks | Size | FPS |
| --- | --- | --- | --- | --- |
| `causal-forcing-t2v` | `causal-forcing-wan2.1-t2v-1.3b-chunkwise` | 60 | 832x480 | 16 |
| `cosmos-predict2-t2v` | `cosmos2-t2v-2b-720p` | 1 | 1280x720 | 16 |
| `fastvideo-causal-wan22-t2v` | `fastvideo-causal-wan2.2-t2v-14b` | 60 | 832x480 | 16 |
| `self-forcing-t2v` | `self-forcing-wan2.1-t2v-1.3b` | 60 | 832x480 | 16 |
| `wan21-t2v` | `wan21-t2v-1.3b-480p` | 1 | 832x480 | 16 |

Models with one block generate a single chunk; the rest roll out
autoregressively for `total_blocks` chunks.

## Running

The launch mode is positional: `flashdreams-run <slug> <mode>`, where `<mode>`
is `mp4`, `null`, or `webrtc`.

Write an MP4, defaulting to `outputs/<slug>.mp4`:

```bash
uv run flashdreams-run wan21-t2v mp4
```

Serve streamed WebRTC output, defaulting to `127.0.0.1:8080`:

```bash
uv run flashdreams-run self-forcing-t2v webrtc
```

The browser UI accepts a prompt before opening a generation session, plays
emitted chunks as they arrive, and records the received stream for download.

Generate without writing anything, which is the cheapest way to exercise a
model end to end:

```bash
uv run flashdreams-run causal-forcing-t2v null
```

## Overrides

`--scenario.KEY VALUE` overrides generation settings: `prompt`, `total_blocks`,
`pixel_height`, `pixel_width`, and `fps`.

```bash
uv run flashdreams-run fastvideo-causal-wan22-t2v mp4 \
    --scenario.prompt "a red fox padding through fresh snow" \
    --scenario.total_blocks 8
```

`--output.KEY VALUE` overrides sink settings, which vary by mode. MP4 accepts
`path`, `fps`, `output_layout`, and `move_to_cpu`; `null` accepts
`store_results`; WebRTC accepts `host`, `port`, `video_width`, `video_height`,
`warmup_chunks`, and `client_liveness_timeout_s`, among others. WebRTC also
takes `--host` and `--port` directly.

```bash
uv run flashdreams-run wan21-t2v mp4 --output.path outputs/fox.mp4
uv run flashdreams-run self-forcing-t2v webrtc --host 0.0.0.0 --port 8099
```

Append `--no-instantiate` to resolve and print a launch without loading the
model, which validates a command in seconds:

```bash
uv run flashdreams-run wan21-t2v mp4 --scenario.total_blocks 2 --no-instantiate
```

## Image-to-video

The app shell conditions on a prompt only, so the image-to-video presets are not
available as application slugs. Run them through their runner slugs instead,
which accept `--prompt` and `--image-path`:

```bash
uv run flashdreams-run wan21-i2v-14b-480p --image-path first_frame.png
```

The same applies to `causal-forcing-wan2.1-i2v-1.3b-framewise` and
`cosmos2-i2v-2b-720p`.
