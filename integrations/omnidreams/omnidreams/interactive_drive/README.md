# interactive-drive

`interactive-drive` is an interactive demo for exploring Omniverse Dreams live.

![interactive-drive screenshot](screenshot.jpg)

Drive a single scene with the keyboard and switch between the two views that
matter most:

- the generated driving view from the world model
- the HD map view that conditions that generated output

This sample uses the flashdreams Alpadreams pipeline for world-model inference,
uses Ludus to render the HD map view, and uses SlangPy for local windowing.

Runs on Windows or Linux with a native host toolchain; the prebuilt Docker
image option is Linux-only.

The implementation is intentionally narrow:

- one scene loaded at startup
- one camera view
- ego-only kinematic controls from the keyboard
- one UI thread and one simulation thread
- explicit WSL-safe CPU staging between Vulkan and CUDA when needed

## Install

### 1. Prerequisites

`interactive-drive` lives in the
[`NVIDIA/flashdreams`](https://github.com/NVIDIA/flashdreams) monorepo as
the `omnidreams.interactive_drive` subpackage of the
`flashdreams-omnidreams` workspace member at
`integrations/omnidreams/omnidreams/interactive_drive/`. Its `flashdreams`,
`flashdreams-omnidreams`, and `ludus-renderer` deps resolve from the
workspace, so no separate SSH-keyed `git clone` of those projects is needed
beyond cloning `flashdreams` itself.

**Hugging Face token.** Scenes and the Cosmos-Reason1 text encoder are
downloaded from Hugging Face, so `HF_TOKEN` must be set in any shell where
you run `omnidreams-prepare` or `interactive-drive`. Create a token at
[`huggingface.co/settings/tokens/new`](https://huggingface.co/settings/tokens/new)
if you don't already have one, and request access to the
[`nvidia/omni-dreams-scenes`](https://huggingface.co/datasets/nvidia/omni-dreams-scenes)
dataset before the first run.

```bash
export HF_TOKEN=<your-hf-token>              # gates nvidia/omni-dreams-scenes
```

On Windows, set the same value as a user or session environment variable
named `HF_TOKEN` instead of using `export`.

If your environment uses another authorized Hugging Face org, pass
`--hf-org <YOUR-HF-ORG>` to `omnidreams-prepare` and `interactive-drive`, or set
`OMNI_DREAMS_HF_ORG=<YOUR-HF-ORG>` once in your shell. OmniDreams scene URLs
read from the world-model manifest are rewritten to the selected org;
unrelated upstream repos (`lightx2v/Autoencoders`, `nvidia/Cosmos-Reason1-7B`)
stay untouched.

### 2. Pick an environment

Pick **one** of the two paths below, then continue to step 3.

#### Option A — Native (host-installed toolchain)

The simplest path if you already have `uv`, a CUDA toolkit, and SDL/Vulkan
on the host (see the [flashdreams root README](../../../../README.md) for the recommended
hardware and CUDA setup). No additional work in this step — proceed to
step 3 from the flashdreams workspace root.

#### Option B — Docker (prebuilt `flashdreams` image)

If you'd rather skip installing CUDA, SDL, and the EGL toolchain on the host,
the prebuilt `flashdreams` image gives you an end-to-end Linux environment.
Additional prerequisites:

- Linux host (Wayland-based local windowing assumes Linux; Windows users
  should use the native path above)
- Docker + `nvidia-container-toolkit`
- GitHub PAT with `read:packages` for `ghcr.io/nvidia/flashdreams`

1. **Pull the image.**

   ```bash
   echo "$GITHUB_PAT" | docker login ghcr.io -u <github-username> --password-stdin
   docker pull ghcr.io/nvidia/flashdreams:base-v0.3-20260430-7985764
   ```

2. **Launch the container** from the `flashdreams` repo root. The repo
   bind-mount lands at `/workspace/flashdreams` and the workdir leaves
   you at the flashdreams workspace root (the same place you'd run uv
   from on a native host):

   ```bash
   docker run --rm -it \
     --gpus all --ipc=host --network=host \
     -v "$PWD":/workspace/flashdreams \
     -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
     -w /workspace/flashdreams \
     -e NVIDIA_DRIVER_CAPABILITIES=all \
     -e HF_TOKEN="$HF_TOKEN" \
     -e HF_HOME=/root/.cache/huggingface \
     -e UV_PROJECT_ENVIRONMENT=/root/.venv \
     -v /run/user/$(id -u)/wayland-0:/run/user/0/wayland-0:rw \
     --device /dev/dri \
     -e WAYLAND_DISPLAY=wayland-0 \
     -e XDG_RUNTIME_DIR=/run/user/0 \
     -e SDL_VIDEODRIVER=wayland \
     ghcr.io/nvidia/flashdreams:base-v0.3-20260430-7985764 \
     bash
   ```

3. **Install EGL inside the container.** The world-model backend renders
   frames via EGL on both run paths (HUD and `--no-hud` bare Vulkan
   window), so this step is required regardless of which one you use:

   ```bash
   apt-get update && apt-get install -y --no-install-recommends \
       libegl-dev libgl-dev && \
   mkdir -p /usr/share/glvnd/egl_vendor.d && \
   cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
   {
       "file_format_version": "1.0.0",
       "ICD": { "library_path": "libEGL_nvidia.so.0" }
   }
   EOF
   ```

> [!TIP]
> If `uv sync` in step 3 below ever appears to hang, re-run it with `-vv` —
> the most common cause is an SSH host-key prompt for a remote that isn't
> yet in `known_hosts`.

### 3. Sync and stage assets

Run everything from the **flashdreams workspace root** — the standard
flashdreams convention. The `interactive-drive` extra pulls in `slangpy`
(the Vulkan-backed local windowing runtime) on top of the base
`flashdreams-omnidreams` deps; server-only users (`omnidreams.webrtc` /
`omnidreams.grpc`) skip it.

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run --package flashdreams-omnidreams omnidreams-prepare \
  --scene-uuid clipgt-01d503d4-449b-46fc-8d78-9085e70d3554
```

`uv sync --extra interactive-drive` installs the full demo runtime (slangpy + Ludus
front end, flashdreams + flashdreams-omnidreams pipeline).
`omnidreams-prepare` stages the requested scene USDZ from the
resolved scenes dataset
(`nvidia/omni-dreams-scenes` by default, or another authorized org when
`OMNI_DREAMS_HF_ORG` / `--hf-org` points there) and pre-warms the
Cosmos-Reason1 text encoder used at runtime (~14 GB of Hugging Face cache),
so the first setup can take a while depending on your network. Flashdreams
owns video checkpoint selection and cache layout for the selected recipe.

> **Tip:** You can skip the `omnidreams-prepare` step for the
> *default* scene — `interactive-drive` will auto-stage it from
> `nvidia/omni-dreams-scenes` on first launch as long as `HF_TOKEN` is
> set. Run `omnidreams-prepare` explicitly when you want to stage
> multiple scenes ahead of time or pre-warm the Cosmos-Reason1 text
> encoder (~14 GB) so the first launch isn't blocked on it.

Common `omnidreams-prepare` flags:

- `--scene-uuid <clipgt-...>` — stage only one specific scene instead of
  all of them. Useful on bandwidth-constrained links (and a good first
  choice inside the container). Browse available UUIDs on the
  [scenes dataset page](https://huggingface.co/datasets/nvidia/omni-dreams-scenes/tree/main/scenes).
- `--skip-hf-prewarm` — skip pre-warming Hugging Face model repos;
  flashdreams will pull assets lazily on first use.
- `--skip-text-encoder` — skip the ~14 GB text-encoder prewarm when you're
  using a precomputed prompt embedding or want a lighter first-time setup.
- `--skip-scene` — don't stage any scene (for when you're supplying your
  own USDZ via `interactive-drive --scene`).

If `omnidreams-prepare` fails with `401`, `403`, or a gated-repo
error, verify `HF_TOKEN` and confirm access to `nvidia/omni-dreams-scenes`.

Once done, you should see this binary asset under the shared scenes cache:

- `~/.cache/flashdreams/omnidreams-scenes/clipgt-<scene-uuid>.usdz`

That directory (`$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/`) is shared
with the WebRTC server: the desktop demo keeps the archive there and
the WebRTC session pipeline extracts each scene's `clipgt/` payload
under `<uuid>/` next to it. Both demos go through the same
`huggingface_hub` content-addressed cache for the actual download, so
the HF round-trip happens at most once per scene UUID regardless of
which demo you launch first. Hugging Face model snapshots live in the
normal HF cache; flashdreams manages its own video checkpoint
locations.

### First-run behavior

The first world-model launch can spend several minutes in loading and
optimization before the view becomes interactive. In the browser stream this
shows up as `Loading world model...` followed by `Optimizing world model...`.
That phase includes checkpoint loading, torch compilation / CUDA graph setup,
and Triton autotuning. Subsequent runs are usually much faster because cached
kernels and model assets are reused.

During Triton autotuning you may see non-fatal messages such as
`Runtime error during autotuning: permute(sparse_coo)...`. Those indicate that
an autotuner candidate was rejected for the current tensor shape; the runtime
continues with a valid candidate.

## Run

All commands below are run from the **flashdreams workspace root**. The
`interactive-drive` CLI's defaults for `--scene`, `--manifest`,
`--scene-dir`, and `--wheel-profiles-dir` resolve to the bundled assets in
this subpackage (via `__file__`), so you don't have to pass long
`integrations/omnidreams/omnidreams/interactive_drive/...` paths unless
you want to override them.

There is one entry point — `interactive-drive` — and two modes selected by
flags:

| Mode | When to use | How |
|---|---|---|
| **HUD (default)** | You have a graphical desktop session and want the full demo: scene/variant selector, steering wheel + pedals overlay, BEV minimap, keyboard *and* wheel input. | `interactive-drive ...` |
| **Bare backend, local window** | You want the lightweight setup: a single Vulkan window showing the world-model output, no HUD chrome. | `interactive-drive --no-hud ...` |

For browser / remote streaming, use the separate
`omnidreams.webrtc.server` entry point (see [`integrations/omnidreams/README.md`](../../README.md))
— it ships a WebRTC viewer with a polished frontend and lower latency
than an in-process HTTP stream would offer.

The HUD itself uses pygame/SDL2 for rendering, which keeps the demo responsive
in fullscreen at high display resolutions (press `F11` to toggle). It supervises
the headless backend as a subprocess on `127.0.0.1:<--port>` so the world model
keeps running across scene / variant switches without restarting the entire
process.

### HUD mode (default)

```bash
uv run --package flashdreams-omnidreams interactive-drive
```

The default `--scene` resolves to
`$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/clipgt-01d503d4-449b-46fc-8d78-9085e70d3554.usdz`
(staged on first launch via the HF auto-stage flow described above);
`--manifest` resolves to the bundled `configs/example_world_model.yaml`.
Pass a different `--scene` or `--manifest` to override.

`--cuda-visible-devices` defaults to `auto`, which leaves whatever
`CUDA_VISIBLE_DEVICES` you already exported alone (or, on hosts without
the env var set, lets CUDA see every GPU). The HUD does not auto-pick a
GPU — on multi-GPU hosts where the default-zero pick is wrong, export
`CUDA_VISIBLE_DEVICES=<idx>` before launching or pass
`--cuda-visible-devices <idx>` directly. `--cuda-visible-devices ""`
force-unsets the env var.

The HUD also subscribes to the backend's `/bev_stream` and shows a top-down
BEV minimap below the steering and pedal controls; pass `--no-bev` to skip
the extra rasterizer dispatch when you don't need it.

**Steering wheel support.** Drop a profile YAML (axis map, FFB settings,
device-name match patterns) into `configs/wheels/` and the HUD will pick it
up at startup. With `--wheel-profile auto` (the default), the HUD scans
`/dev/input/by-id` first, then `/dev/input/event*`, and matches the
detected device name against each profile's `detection_patterns`. To name a
specific profile use `--wheel-profile <name>` (matching the YAML filename);
to bind a known device path directly use `--wheel-device /dev/input/eventX`;
to disable wheel input entirely use `--no-wheel`. No profiles ship with the
repo — keyboard-only driving works fine without one.

### `--no-hud`: bare backend, local Vulkan window

This is the lighter-weight path that matches the older standalone
`interactive-drive` script: a single Vulkan window for the omnidreams
output, no HUD chrome, no scene selector.

```bash
uv run --package flashdreams-omnidreams interactive-drive --no-hud
```

You should initially see the generated driving view. Press `2` to switch to the
HD map view (conditioning input) and `1` to switch back to the photorealistic
output.

Controls (apply in both modes):

- `W` throttle
- `S` brake / reverse drag
- `A` steer left
- `D` steer right
- arrow keys mirror `W/A/S/D`
- `Space` stop
- `1` generated driving view
- `2` HD map view
- `R` reset rollout
- `Esc` quit

The browser control hint is static today, so it does not confirm every keydown
visually. If the world-model backend is still producing a chunk, input can be
accepted before the visual response arrives.

### Rollout drift and resets

OmniDreams generates video autoregressively, so long rollouts can accumulate
artifacts such as diagonal striping, color bleeding, or distorted geometry,
especially after extended driving without a reset. This does not necessarily
mean the demo is broken. Press `R` to restart from the scene's initial clean
state. For long demos, reset every 30-50 generated chunks or whenever visual
quality starts to drift.

### Without HD-map data (synthetic scene)

Use `--synthetic-scene` to skip the USDZ download entirely. interactive-drive
builds a procedural 2-lane road with a single intersection at startup and
feeds it to the same loader the regular flow uses:

```bash
uv run --package flashdreams-omnidreams omnidreams-prepare --skip-scene
uv run --package flashdreams-omnidreams interactive-drive \
  --synthetic-scene \
  --synthetic-initial-rgb path/to/forward_facing_road_photo.jpg
```

The world model is trained on natural driving frames, so passing your own
forward-facing road photo through `--synthetic-initial-rgb` makes the
generation start from a believable RGB instead of the scene_fixture's debug
gradient. Any aspect ratio works; the loader resizes to the raster
resolution. `--synthetic-prompt` similarly overrides the embedded text prompt.

The procedural road is a **20 km** golden track lined with periodic
streetlamp-style poles (~50 m spacing), parked cars on alternating
shoulders (~150 m spacing), and traffic signs on alternating shoulders
(~200 m spacing) so the scene doesn't look empty. The centerline is a
sum-of-sines so the road varies naturally instead of feeling like the
same kilometre on repeat. There's no per-session length knob -- the
track is generous enough that no demo realistically reaches the end.

## Develop

Lint and type-check via the workspace-wide pre-commit hooks
([`flashdreams/.pre-commit-config.yaml`](../../../../.pre-commit-config.yaml));
install once from the flashdreams workspace root with
`uv run pre-commit install`.

Run the subpackage's test suite (CPU subset) from the flashdreams workspace
root:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --extra dev
uv run --no-sync --package flashdreams-omnidreams pytest \
  integrations/omnidreams/tests/interactive_drive -m "not gpu and not xvfb"
```

Root CI's `pytest -m ci_cpu` / `pytest -m ci_gpu` also pick up these tests
automatically — the subpackage's `tests/conftest.py` auto-stamps each test
with the right workspace CI-tier marker based on its existing
`gpu` / `xvfb` markers.

The hook does not auto-fix. Use `./scripts/check.sh --fix` to clean up lint and
format issues explicitly.
