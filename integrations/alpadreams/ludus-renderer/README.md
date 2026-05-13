# Ludus Renderer

GPU-native F-theta mesh shader rendering for autonomous vehicle simulation.

## Features

- **F-theta Camera Model**: Native support for fisheye lens distortion using polynomial projection
- **Mesh Shader Rendering**: High-performance GPU rendering using NVIDIA mesh shaders
- **Timestamped Rendering**: Efficient temporal queries for simulation playback
- **Adaptive Tessellation**: Automatic subdivision based on distortion error
- **MSAA**: 4x hardware antialiasing
- **CUDA-GL Interop**: Direct GPU memory sharing between rendering and compute
- **Mirror Augmentation**: Extend scenes by tiling reflected copies for longer driving sequences
- **GPU Spatial Culling**: Per-element AABB/sphere culling in task shaders for large scenes

## Primitives

- **Polylines**: Thick line strips with configurable width and round caps
- **Polygons**: Filled polygons with pre-triangulation
- **Cubes**: Oriented bounding boxes with 9-DOF transform

## Requirements

- NVIDIA GPU with mesh shader support (Turing or later)
- OpenGL 4.6 with EGL
- CUDA 11+
- Python 3.10+
- ffmpeg (for MP4 muxing with `--output-format mp4`)

## Installation

```bash
uv sync
```

Dependencies installed:
- PyTorch 2.0+
- NumPy, Pandas, SciPy
- PyArrow (for parquet scene files)
- Pillow, ImageIO (for image handling)

## Usage

```python
from ludus_renderer import LudusTimestampedContext
ctx = LudusTimestampedContext(device="cuda")
```

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
- `--output-format`: `png` (default), `jpg` (nvJPEG hardware encode), or `mp4` (H264 hardware encode)
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

## License

This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for the full license text.

## Contributing

Contributions are welcome, thank you. This project only accepts contributions under the
Apache License, Version 2.0. All contributions must be signed off in accordance with the
[Developer Certificate of Origin (DCO)](CONTRIBUTING).

Sign off your commits by adding a `Signed-off-by` line to each commit message
(`git commit -s`):

```
Signed-off-by: Your Name <your.email@example.com>
```

Third-party attributions for any code bundled in this repository are listed in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).
