# Ludus Renderer

GPU-native F-theta renderer for autonomous vehicle simulation, with two
interchangeable backends:

- **CUDA software rasterizer** (`LudusCudaTimestampedContext`): always
  available, no graphics driver required, built on the HPG 2011
  CudaRaster triangle pipeline.
- **Vulkan mesh-shader backend** (`LudusTimestampedContext`): opt-in,
  uses `VK_EXT_mesh_shader` for hardware-accelerated procedural geometry
  with CUDA-Vulkan external-memory interop.

## Features

- **F-theta Camera Model**: Native support for fisheye lens distortion using polynomial projection
- **Two rendering backends**: CUDA software rasterizer or Vulkan mesh shaders
- **Timestamped Rendering**: Efficient temporal queries for simulation playback
- **Adaptive Tessellation**: Automatic subdivision based on distortion error
- **MSAA**: 4x antialiasing (2x supersampling on CUDA, hardware MSAA on Vulkan)
- **Mirror Augmentation**: Extend scenes by tiling reflected copies for longer driving sequences
- **GPU Spatial Culling**: Per-element AABB/sphere culling for large scenes

## Primitives

- **Polylines**: Thick line strips with configurable width and round caps
- **Polygons**: Filled polygons with pre-triangulation
- **Cubes**: Oriented bounding boxes with 9-DOF transform

## Requirements

**Always:**
- NVIDIA GPU (Turing or later)
- CUDA 11+
- Python 3.10+
- ffmpeg (for MP4 muxing with `--output-format mp4`)

**For the Vulkan backend additionally:**
- Vulkan 1.3 SDK or `libvulkan-dev` + `libvulkan1` (Debian/Ubuntu)
- An NVIDIA GPU + driver that exposes `VK_EXT_mesh_shader` (Ada generation
  and later on the latest production drivers; consult `vulkaninfo`).
- To rebuild shaders from source: `glslangValidator` from the Vulkan SDK.

## Installation

```bash
uv sync
```

Dependencies installed:
- PyTorch 2.11+
- NumPy, Pandas, SciPy
- PyArrow (for parquet scene files)
- Pillow, ImageIO (for image handling)

## Usage

The default CUDA backend just works everywhere:

```python
from ludus_renderer import LudusCudaTimestampedContext
ctx = LudusCudaTimestampedContext(device="cuda")
```

The Vulkan backend has the same public API and is selected at construction:

```python
from ludus_renderer import LudusTimestampedContext
ctx = LudusTimestampedContext(device="cuda")
# ... ctx.upload_cameras / upload_scene / render_batch ...
```

If Vulkan headers / loader / `VK_EXT_mesh_shader` are missing, the
constructor raises a friendly `ImportError`. The CUDA backend stays
fully usable.

### Choosing a backend

| Aspect | CUDA backend | Vulkan backend |
| --- | --- | --- |
| Driver requirement | CUDA only | CUDA + Vulkan 1.3 + `VK_EXT_mesh_shader` |
| Geometry generation | CUDA kernels (CPU-side dispatch) | Task/Mesh shaders on the GPU |
| Render dispatch | One CUDA kernel per query (Python loop) | Single Vulkan submission per batch |
| MSAA | 2x supersampling | Hardware 4x MSAA |
| First-call cost | Plugin JIT (~10s once) | Plugin JIT (~10s once) + Vulkan context init |

### Shader build (Vulkan only)

The Vulkan backend ships with pre-compiled SPIR-V embedded in
`_cpp/render/shaders_spv.h`. The GLSL sources in `ludus_renderer/shaders/`
are hand-maintained `GL_EXT_mesh_shader` style; regenerate after editing them
with:

```bash
bash ludus_renderer/shaders/compile.sh    # GLSL -> SPIR-V -> embed in shaders_spv.h
```

### v1 caveats for the Vulkan backend

- Dot-primitive rendering (`PRIM_DOT_*`) is implemented in the CUDA
  backend but not yet plumbed through the Vulkan task/mesh pipeline.
- The CUDA-Vulkan handoff uses opaque file-descriptor external memory,
  which is Linux-only; the Vulkan plugin currently refuses to build on
  Windows. The CUDA backend remains cross-platform.
- Diagnostics: set `LUDUS_VK_DEBUG=1` to enable internal `[Vulkan] ...`
  trace logs, and `LUDUS_VK_CLEAR_RED=1` to clear the framebuffer to
  opaque red instead of transparent black (useful to verify the
  render-pass and readback path are alive independent of the shaders).

## Examples

### HDMap Scene Renderer

Render clipgt HDMap scenes with road geometry, lane lines, obstacles, and traffic elements:

```bash
# Render a single frame
uv run python examples/render_hdmap_scene.py --scene example_data/test_hdmap --frame 12

# Render bird's eye view
uv run python examples/render_hdmap_scene.py --scene example_data/test_hdmap --frame 12 --bev

# Render full sequence to PNG frames
uv run python examples/render_hdmap_scene.py --scene example_data/test_hdmap --sequence

# Render all cameras at 30fps as MP4
uv run python examples/render_hdmap_scene.py --scene example_data/test_hdmap --sequence --all-cameras --fps 30 --output-format mp4

# Render specific camera with JPEG output
uv run python examples/render_hdmap_scene.py --scene example_data/test_hdmap --sequence --camera camera:front:wide:120fov --output-format jpg

# Enable 4x antialiasing
uv run python examples/render_hdmap_scene.py --scene example_data/test_hdmap --sequence --msaa 4
```

**Key options:**
- `--msaa N`: MSAA sample count (`0` = disabled, `4` = 4x antialiasing)
- `--camera NAME`: Render from a specific scene camera (use `--list-cameras` to see available)
- `--all-cameras`: Render from all available cameras in the scene
- `--fps N`: Output frame rate in Hz (default: 10)
- `--output-format`: `png` (default), `jpg` (nvJPEG hardware encode), or `mp4` (H264 via ffmpeg libx264)
- `--batch-size N`: Number of frames to render per GPU batch (default: all frames at once)
- `--quality N`: JPEG quality 1-100 (default: 90)
- `--bitrate N`: MP4 bitrate in bps (default: 10Mbps)

Scene elements rendered:
- Road boundaries, lane lines (solid/dashed/dotted, white/yellow)
- Crosswalks, road markings, wait lines
- Traffic lights, traffic signs, poles
- Dynamic obstacles (vehicles, pedestrians)
- Ego trajectory and BEV ego vehicle

### Video Overlay

Composite rendered HD map elements on top of an input video (50:50 blend). Supports all output formats:

```bash
# Overlay as JPEG frames (GPU-accelerated via nvjpeg)
uv run python examples/render_hdmap_scene.py --scene example_data/debug_021926 \
    --overlay-video example_data/debug_021926/av_ec2fb4fa-3530-4a6a-b431-f06779a0537a.camera_front_wide_120fov.mp4 \
    --output-format jpg

# Overlay as MP4 video
uv run python examples/render_hdmap_scene.py --scene example_data/debug_021926 \
    --overlay-video example_data/debug_021926/av_ec2fb4fa-3530-4a6a-b431-f06779a0537a.camera_front_wide_120fov.mp4 \
    --output-format mp4

# Overlay as PNG frames
uv run python examples/render_hdmap_scene.py --scene example_data/debug_021926 \
    --overlay-video example_data/debug_021926/av_ec2fb4fa-3530-4a6a-b431-f06779a0537a.camera_front_wide_120fov.mp4 \
    --output-format png
```

Blending is performed on GPU using PyTorch integer arithmetic. For JPEG output, encoding uses nvjpeg hardware acceleration. Frame count is capped to `min(rendered frames, video frames)`.

### Mirror Augmentation

Extend a scene by mirror-stitching it N times at load time. Two canonical tiles (original + single reflection) are placed alternately via rigid body transforms, producing an `[original]-[mirror]-[original]-[mirror]-...` pattern without rotational drift on curved roads.

```python
from ludus_renderer import load_clipgt_scene, mirror_augment_scene

scene = load_clipgt_scene("example_data/clipgts/clipgt-0300edb0-...", device=device)
extended = mirror_augment_scene(scene, n_mirrors=10, lookahead_m=50.0)
```

- `n_mirrors`: number of augmentation iterations (total segments = n_mirrors + 1)
- `lookahead_m`: distance (metres) beyond the ego endpoint to place the first mirror plane

GPU-side spatial culling ensures that rendering cost stays constant regardless of the augmented scene size. Culling is enabled by default (1.5x `depth_max`) and can be adjusted via:

```python
ctx.set_cull_radius(scale=1.5)  # 0 disables culling
```

### Benchmarking

```bash
# Single scene benchmark
uv run python examples/benchmark_renderer.py --scene example_data/test_hdmap --iters 10

# Multi-camera benchmark (8 cameras per timestamp)
uv run python examples/benchmark_renderer.py --scene example_data/test_hdmap --multicam
```

# Contributing

Contributions are welcome, thank you. This project only accepts contributions under the
Apache License, Version 2.0. All contributions must be signed off in accordance with the
[Developer Certificate of Origin (DCO)](CONTRIBUTING).
