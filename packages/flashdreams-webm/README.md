# FlashDreams WebM bindings

In-process AV1 recording with SVT-AV1 4.2.0, libwebm 1.0.0.32 and
dav1d 1.5.3. The extension statically links these libraries. It neither builds
nor ships recorder executables, FFmpeg, or GStreamer.

Build with `uv build --wheel packages/flashdreams-webm`. The isolated build
installs scikit-build-core, pybind11, CMake, Ninja and Meson, downloads
hash-pinned upstream sources, and includes their notices in the wheel.
Installing a compatible prebuilt wheel requires no compiler or special shell.
Building from source on Windows requires Visual Studio C++ Build Tools and the
Windows SDK. Package builds automatically discover Visual Studio with `vswhere`
and initialize its compiler environment for CMake, Ninja and Meson. Run `uv sync`
or `uv build` from your usual terminal; no developer prompt or IDE is needed.
YASM is downloaded automatically on Windows.
Linux requires a C/C++ toolchain and YASM on x86; ARM64 uses native assembly.
SVT-AV1 uses YASM for optimized x86 encoding. dav1d uses its C decoder on x86
because its assembly requires NASM; decoding during canvas growth may be slower.

`Writer(path, width, height, fps_numerator, fps_denominator)` accepts packed
8-bit I420 bytes through `write()`. Its `width` and `height` properties give
the padded input dimensions: SVT-AV1 needs even dimensions of at least 64.
WebM crop/display metadata preserves the requested visible dimensions.
`close()` drains delayed frames and records duration including the last frame.
`abort()` releases the encoder without finalizing. Writers are serialized and
release the Python GIL during encoding and finalization.

`write_rgb(frame)` accepts a contiguous `[H, W, 3]` uint8 buffer. It converts
RGB to limited-range BT.709 I420, pads smaller frames, and submits the frame in
one native call with the GIL released. The runtime uses this path by default;
`WebmClientWindow(path, native_rgb=False)` retains the NumPy reference path.
Frame conversion and submission are synchronous; SVT retains its internal
encoder workers.

`Reader(path)` exposes `width`, `height`, `duration_ns`, and `read()`, which
returns visible Y/U/V plane bytes or `None` at EOF. Call `close()` to release
the file. The reader supports the single-track AV1 files produced here.

The encoder uses preset 10, CRF 35, limited-range BT.709 with sRGB transfer.
These are initial recording settings, not an equal-quality match to x264 CRF 18.
Encoder workers and lookahead buffer frames; the entire video is not held
until close. The parser retains block metadata while decoding.

Native RGB conversion is **Recommended on the measured Windows x64 stack**.
On 61 saved SwiftVR frames at 2560x1280, the median of three recordings after
12 warmup frames fell from 8.00 s to 1.38 s. Timings include conversion,
encoder startup and finalization; they exclude input loading, model inference,
GPU transfer and validation. Settings remained preset 10 / CRF 35 with YASM.
Decoded YUV PSNR against the same NumPy-converted input was 42.409 dB versus
42.408 dB for the reference path; outputs are not bit-identical. All 61 frames
and their 5.083333 s duration were preserved, including the last frame.
