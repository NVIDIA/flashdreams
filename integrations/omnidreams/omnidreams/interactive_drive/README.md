# interactive-drive

`interactive-drive` is an interactive demo for exploring NVIDIA OmniDreams live.

![interactive-drive screenshot](screenshot.jpg)

Drive a single scene with the keyboard and switch between the two views that
matter most:

- the generated driving view from the world model
- the HD map view that conditions that generated output

This sample uses the flashdreams Alpadreams pipeline for world-model inference,
uses Ludus to render the HD map view, and uses SlangPy for local windowing.

Runs on Windows or Linux with a native host toolchain; the optional
Docker path is Linux-only.

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

#### Option B — Docker (build a local image)

If you'd rather skip installing CUDA, SDL, and the EGL toolchain on the
host, the bundled `docker/Dockerfile` builds an end-to-end Linux
environment locally. See [`docker/README.md`](../../../../docker/README.md)
for the full build docs; the short version is below. Additional
prerequisites:

- Linux host (Wayland-based local windowing assumes Linux; Windows users
  should use the native path above)
- Docker + `nvidia-container-toolkit`

1. **Build the image** from the `flashdreams` repo root:

   ```bash
   docker build -t flashdreams:local -f docker/Dockerfile .
   ```

2. **Launch the container** from the same `flashdreams` repo root. The
   repo bind-mount lands at `/workspace/flashdreams` and the workdir
   leaves you at the flashdreams workspace root (the same place you'd
   run uv from on a native host):

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
     flashdreams:local \
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
  --scene-uuid clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4
```

`uv sync --extra interactive-drive` installs the full demo runtime (slangpy + Ludus
front end, flashdreams + flashdreams-omnidreams pipeline).
`omnidreams-prepare` stages the requested scene USDZ from the
resolved scenes dataset
(`nvidia/omni-dreams-scenes` by default, or another authorized org when
`OMNI_DREAMS_HF_ORG` / `--hf-org` points there), pre-warms the
Cosmos-Reason1 text encoder used at runtime (~14 GB of Hugging Face cache),
and pre-downloads the world-model DiT checkpoint from
`nvidia/omni-dreams-models` (a separate gated repo) so the first launch
isn't blocked on the multi-GB fetch — and so a missing-access error surfaces
here with an actionable hint rather than mid-demo. The first setup can take a
while depending on your network. Flashdreams owns video checkpoint selection
and cache layout for the selected recipe.

> **Tip:** You can skip the `omnidreams-prepare` step for the
> *default* scene — `interactive-drive` will auto-stage it from
> `nvidia/omni-dreams-scenes` on first launch as long as `HF_TOKEN` is
> set. Run `omnidreams-prepare` explicitly when you want to stage
> multiple scenes ahead of time or pre-warm the Cosmos-Reason1 text
> encoder (~14 GB) so the first launch isn't blocked on it.

Common `omnidreams-prepare` flags:

- `--scene-uuid <clipgt-...>` — stage only one specific scene instead of
  all of them (every published weather variant of it). Useful on
  bandwidth-constrained links (and a good first choice inside the
  container). Browse available UUIDs on the
  [scenes dataset page](https://huggingface.co/datasets/nvidia/omni-dreams-scenes/tree/main/scenes).
- `--scene-variant <default|rain|snow>` — with `--scene-uuid`, stage only
  that weather variant instead of all of them.
- `--skip-hf-prewarm` — skip pre-warming Hugging Face model repos;
  flashdreams will pull assets lazily on first use.
- `--skip-text-encoder` — skip the ~14 GB text-encoder prewarm when you're
  using a precomputed prompt embedding or want a lighter first-time setup.
- `--skip-scene` — don't stage any scene (for when you're supplying your
  own USDZ via `interactive-drive --scene`).

If `omnidreams-prepare` fails with `401`, `403`, or a gated-repo
error, verify `HF_TOKEN` and confirm you have requested (and been granted)
access to **both** `nvidia/omni-dreams-scenes` and
`nvidia/omni-dreams-models` — access to one is not access to the other.

Once done, you should see the scene's variant archive(s) under the shared
scenes cache, one per weather:

- `~/.cache/flashdreams/omnidreams-scenes/clipgt-<scene-uuid>.usdz` (and
  `-rain` / `-snow` siblings when published)

That directory (`$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/`) is shared
with the WebRTC server: the desktop demo keeps the archives there and
the WebRTC session pipeline extracts each scene's `clipgt/` payload
under `<uuid>[-<variant>]/` next to them. Both demos go through the same
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

There is one entry point — `interactive-drive` — and three modes selected by
flags:

| Mode | When to use | How |
|---|---|---|
| **HUD (default)** | You have a graphical desktop session and want the full demo: scene/variant selector, steering wheel + pedals overlay, BEV minimap, keyboard *and* wheel input. | `interactive-drive ...` |
| **Bare backend, local window** | You want the lightweight setup: a single Vulkan window showing the world-model output, no HUD chrome. | `interactive-drive --no-hud ...` |
| **Bare backend, browser** | The demo machine has no graphics-capable GPU (e.g. compute-only GB300) or you want to view from a laptop browser while the model runs elsewhere. Implies `--no-hud`. | `interactive-drive --stream-mjpeg [HOST:]PORT ...` |

For a richer remote-viewing experience with a polished frontend and lower
latency than an in-process MJPEG stream, prefer the separate
`omnidreams.webrtc.server` entry point (see
[`integrations/omnidreams/README.md`](../../README.md)).

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
`$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz`
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

**Steering wheel support.** Drop a profile YAML (devices, axis map, FFB
settings) into `configs/wheels/` and the HUD will pick it up at startup. With
`--wheel-profile auto` (the default), the HUD scans `/dev/input/by-id` first,
then `/dev/input/event*`, and matches detected device names against each
profile's devices. To name a specific profile use `--wheel-profile <name>`
(matching the YAML filename); to bind a known device path directly use
`--wheel-device /dev/input/eventX` (this names the wheel/steering device);
to disable wheel input entirely use `--no-wheel`. No profiles ship with the
repo — keyboard-only driving works fine without one.

**Multi-device profiles.** A profile binds one or more devices, so steering,
throttle, brake, and buttons can each live on a different device — a wheel
base plus a separate-brand or separately-connected pedal set, for example.
Each device is listed under `devices` with its own `detection_patterns`, and
every axis/button is a `{device, code}` binding naming the device by index:

```yaml
devices:
  - {display_name: Base, detection_patterns: ["Fanatec CSL DD"]}
  - {display_name: Pedals, detection_patterns: ["Heusinkveld"]}
axis_map:
  steering: {device: 0, code: 0}   # wheel base
  throttle: {device: 1, code: 0}   # separate pedals
  brake:    {device: 1, code: 1}
```

At launch each device is matched independently by name; the steering device
is required, while a device used only by other controls degrades gracefully
(a warning, those controls inactive) if unplugged. Older single-device
profiles (top-level `detection_patterns` + bare integer codes) still load.
Device 0 (the steering device) is the one that produces force feedback.

**Force feedback (multi-vendor).** A profile's `ffb.mode` selects how the
centering force is rendered, and defaults to `auto`, which inspects the
device's advertised Linux FF effects and picks the right backend:

- `autocenter` — a driver-managed spring written via `FF_AUTOCENTER`.
  Used by Thrustmaster and Logitech wheels.
- `constant_force` — a self-rendered spring uploaded via the `FF_CONSTANT`
  effect, where the app computes the force each tick. This is the universal
  path and is required for Fanatec wheels, whose `hid-fanatecff` driver does
  not expose `FF_AUTOCENTER`.

With `mode: auto` (or the wizard's "Auto"), a Fanatec base automatically
falls back to constant force while Thrustmaster/Logitech keep their managed
autocenter. Set `mode: constant_force` explicitly if a Logitech's in-kernel
autocenter feels too weak. Driver prerequisites: most modern Thrustmaster
wheels (T300RS, T248, TX, T-GT II, TS-PC, TS-XW, …) need the out-of-tree
[`hid-tmff2`](https://github.com/Kimplul/hid-tmff2) module plus a wheel-mode
init (`hid-tminit`, or `tmdrv` for TX/TS-XW); Fanatec needs the
[`hid-fanatecff`](https://github.com/gotzl/hid-fanatecff) module with the
base in PC mode (red LED); Logitech G29/G27/G923-PS use the in-kernel
`hid-lg4ff` or [`new-lg4ff`](https://github.com/berarma/new-lg4ff), while the
G920 and Xbox/PC G923 use the HID++ driver (kernel >= 6.3). FFB writes need
access to `/dev/input/*` (add your user to the `input` group).

**Generate an input profile (wheel or game controller).** Instead of
hand-writing that YAML, run the calibration wizard:

```bash
uv run --package flashdreams-omnidreams interactive-drive-configuration
```

It shows a live panel -- a steering-wheel and pedal visualization plus a
per-axis activity strip -- so you can confirm the right device and watch each
control move. Ctrl+click to select more than one device when your controls
are split across devices (e.g. a wheel base plus a separate pedal set); the
wizard listens to all selected devices at once and binds each control to
whichever device it actually moved on. It then listens while you move each
control to capture the correct axes and directions (self-centering sticks and
force-feedback wheels work because it peak-holds each axis' range rather than
snapshotting after you let go), lets you bind reverse / reset / exit-scene
buttons and test force feedback, then writes the profile to
`$FLASHDREAMS_CACHE_DIR/interactive-drive/wheels/` (by default under
`~/.cache/flashdreams/`). The next `interactive-drive` launch discovers it
automatically through the same `--wheel-profile auto` detection. The wizard
supports both steering wheels (with pedals) and game controllers (analog
stick plus triggers); the generated file stays on your machine and is never
committed. It needs a graphical session and read access to `/dev/input/*`
(add your user to the `input` group if no devices are found).

The opening screen also lists your saved profiles so you can edit their
settings (display name, steering range and deadzone, inversion, force
feedback mode and gain, detection patterns), choose which one is the
default, or delete them. Steering range and deadzone are most useful for game controllers,
whose sticks are sensitive and tend to drift -- lower the range to make
steering less twitchy and raise the deadzone to ignore a drifting stick at
rest.

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

### `--stream-mjpeg`: bare backend served over HTTP

Use this when the demo machine has no graphics-capable GPU (e.g. a
compute-only GB300 in a DGX Station), when you're connecting over the
network, or when you want to demo from a laptop browser while the model
runs elsewhere. Implies `--no-hud` because the user is then viewing
through a browser, not a local Vulkan window — the slangpy HUD itself
is a Vulkan presenter, so it can't run on the same hosts that need
`--stream-mjpeg`.

```bash
uv run --package flashdreams-omnidreams interactive-drive \
  --stream-mjpeg 8080
```

Open `http://<host-ip>:8080/` in a browser on the same network; keyboard
events posted from the page are forwarded to the demo over the same socket.
The flag accepts `8080`, `:8080`, or `0.0.0.0:8080` (all equivalent — bind
on all interfaces); pass an explicit host (`127.0.0.1:8080`) to restrict
the listener to a single interface.

The browser viewer ships an HTML/CSS HUD overlay so the headless demo
matches the desktop modes' affordances:

- A **scene picker** in the upper-right lists the same scenes the
  slangpy HUD discovers (anything under `--scene-dir`, defaulting to
  `$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/`). Pick a scene and a
  variant, click *Load Scene*, and the demo tears down the current
  rollout and rebuilds with the new scene without dropping the
  browser session — the stream pauses briefly during the rebuild and
  resumes the moment the new pipeline produces its first chunk. The
  picker is hidden when no scenes were discovered.
- A **speed readout** in the lower-left, rendered in MPH, polls the
  server's `/state` endpoint at 10 Hz. Reads `--` until the simulation
  has produced its first chunk; numeric the moment chunks start
  arriving.
- **WASD chiclets** light up while the corresponding direction key is
  held. The page tracks the `keydown`/`keyup` set locally so the
  highlight is zero-latency (no server round-trip); arrow keys light
  the same chiclets as their letter equivalents.
- **Auto-crawl**: releasing throttle keeps the ego creeping toward
  ~10 mph (4.47 m/s) instead of coasting to a stop, matching the
  alpasim manual-driver behaviour and the slangpy HUD's keyboard
  path. The `--stream-mjpeg` presenter routes browser keypresses
  through the same `KeyboardDriveState` integrator the desktop HUD
  uses, so it posts `DriverCommand(manual_control=True, ...)` which
  unlocks the creep branch in `EgoVehicleKinematics.integrate_vehicle`.

If you're running on a remote box (cloud GPU, lab machine, headless
server), the demo port may not be reachable directly from your laptop.
Forward the port over SSH (or your provisioner's CLI) and open the
forwarded URL instead — for example:

```bash
ssh -L 8080:localhost:8080 <user>@<host>
```

Then open `http://localhost:8080/`.

For a richer browser frontend with lower latency, prefer the separate
`omnidreams.webrtc.server` entry point.

#### Fully headless: `--stream-mjpeg` with `--auto-start`

By default the streaming mode waits for the browser scene picker before it
loads anything, so a freshly launched server idles on "Select a scene to
begin" until someone opens the page. To run with no GUI/browser interaction
add `--auto-start`. It skips the scene selection and immediately loads `--scene`
(or the first scene discovered under `--scene-dir` when `--scene` is unset
or not yet staged):

```bash
uv run --package flashdreams-omnidreams interactive-drive \
  --stream-mjpeg 8080 --auto-start --scene <clip-id-name-or-path>
```

The interactive-drive CUDA fast path is enabled by default. HDMap raster frames
stay CUDA-backed for world-model conditioning, and both Vulkan presenters use
SlangPy CUDA interop for generated RGB frames when the model output is still on
CUDA:

```bash
uv run --no-sync --package flashdreams-omnidreams interactive-drive
```

Set `INTERACTIVE_DRIVE_DISABLE_CUDA_INTEROP=1` to force the conservative host
path for both HDMap raster conditioning and presenter CUDA interop. In HUD mode,
chrome is still rendered with PIL on the CPU, then uploaded as an alpha overlay;
the generated camera frame stays lazy on CUDA and is resized/composited into the
shared presentation buffer on the CUDA stream. Use `--no-hud` with the same
environment variable for the bare presenter.

Controls (apply in all three modes):

- `W` throttle
- `S` brake / reverse drag
- `A` steer left
- `D` steer right
- arrow keys mirror `W/A/S/D`
- `Space` stop
- `1` generated driving view
- `2` HD map view
- `R` reset rollout
- `X` exit scene (return to the scene selector; HUD mode only)
- `Esc` quit

The browser control hint is static today, so it does not confirm every keydown
visually. If the world-model backend is still producing a chunk, input can be
accepted before the visual response arrives.

### Generated-frame e2e profiling

Set `INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT=1` to log generated-frame
input-to-present timing while the demo runs:

```bash
INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT=1 \
  uv run --no-sync --package flashdreams-omnidreams interactive-drive --auto-start
```

The log line is `[profile] e2e ...`. `wall_present_fps` counts only frames
consumed from the model pipeline queue, so loading-frame or hold-frame
re-presents are excluded. `avg_adj_control_to_present_ms` subtracts the
intentional per-frame spacing inside a generated chunk; the raw value is also
printed for debugging. Set
`INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT_INTERVAL_S` to adjust the report
period; the default is `2`.

### Rollout drift and resets

OmniDreams generates video autoregressively, so long rollouts can accumulate
artifacts such as diagonal striping, color bleeding, or distorted geometry,
especially after extended driving without a reset. This does not necessarily
mean the demo is broken. Press `R` to restart from the scene's initial clean
state. For long demos, reset every 30-50 generated chunks or whenever visual
quality starts to drift.

### Out-of-bounds warning and auto-respawn

The demo also auto-resets when you drive off the navigable area. The
implementation mirrors alpasim's ``is_ego_off_map`` algorithm: at scene
load time the demo computes an axis-aligned bounding box of **every**
spatial layer in the scene (lane markers, drivable triangles,
polygons, vehicle tracks, ground mesh) and prints it to stderr, e.g.

```
[ego_vehicle_kinematics] map bounds: x=[-127.4, 218.9] (346.3 m), y=[-89.6, 142.3] (231.9 m). Adds 50 m margin + 100 m warning zone for OOB.
```

Each chunk, the simulation computes how far the ego is from this AABB
expanded by a 50 m margin and the loop reacts to it in two stages:

- **Approaching the edge** (proximity ramps `0.0 → 1.0` across the
  100 m warning zone inside the AABB+margin edge): the warning text
  *"Approaching map edge, turn back to avoid respawn"* is overlaid on
  the current frame. Steering back into the navigable area clears it
  on the next chunk.
- **Out of bounds** (proximity = `2.0`, set when the ego has actually
  crossed the AABB+margin boundary): the overlay flips to
  *"Respawning..."* and the loop triggers the same reset path that `R`
  uses.

Crucially, the respawn is a **binary** trigger — it fires only when
the ego is actually past the AABB+margin edge, not when it's somewhere
in the warning ramp. Driving on a sidewalk, brushing curbs, or
crossing sparse-mesh patches *inside* the navigable area never trigger
a teleport; only flying off the entire mapped area does. Because the
AABB is the union of all geometry (not just the ground mesh, which is
often a small strip representing only the road surface), it covers the
full extent of where the scene "contains" content.

The loop logs every state transition to stderr so you can confirm the
thresholds are firing at the right time:

```
[loop] oob 'in-bounds' -> 'Approaching map edge…' proximity=0.620 streak=0 action=warning
[loop] oob 'Approaching map edge…' -> 'Respawning...' proximity=2.000 streak=1 action=firing respawn
```

Five CLI flags expose the OOB knobs:

| Flag | Default | Effect |
|---|---|---|
| `--oob-margin-m` | `50` | Margin (m) added around the geometry AABB. Bigger = more room before the boundary. |
| `--oob-warning-zone-m` | `100` | Depth (m) of the warning-ramp band inside the AABB+margin edge. Set to `0` to disable the ramp. |
| `--oob-warn-proximity` | `0.6` | Lower values warn earlier. |
| `--oob-respawn-proximity` | `2.0` | Default `2.0` matches alpasim's binary "off map" sentinel. Set to `2.5` (or any value > `2.0`) to disable auto-respawn entirely while keeping the warning overlay. |
| `--oob-respawn-debounce-chunks` | `1` | Default `1` matches alpasim. Higher values add a per-chunk buffer. |

Both messages render through the standard `status_message` overlay, so
they look identical across the HUD, `--no-hud`, and `--stream-mjpeg`
presenters. Scenes that ship no spatial geometry report proximity
`0.0` and never auto-respawn -- you get the same behaviour as the
older builds.

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

Pass `--synthetic-model` (or `synthetic_model: true` in the manifest) to run
the world model on local random-initialized weights, so latency/code-path
tests run with no checkpoint downloads.

The hook does not auto-fix. Use `./scripts/check.sh --fix` to clean up lint and
format issues explicitly.
