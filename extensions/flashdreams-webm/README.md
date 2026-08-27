<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams native WebM companion

`flashdreams-webm` is the optional native output wheel for FlashDreams V2. It
statically links pinned builds of libvpx, libopus, and libwebm into one Python
extension. The FlashDreams runtime imports it only for `--mode webm`; the core
package and external-FFmpeg MP4 path do not depend on it.

The initial wheel target is Linux and requires CMake, a C/C++ toolchain, GNU
Make, and NASM or Yasm. Build it from the FlashDreams workspace with:

```bash
python -m pip wheel ./extensions/flashdreams-webm --no-deps
```

The build downloads checksum-pinned upstream source archives and produces a
self-contained wheel. Exact upstream license and patent notices are installed
as wheel license files.

Install the companion beside FlashDreams with
`pip install 'flashdreams[webm]'`, then select it with
`--mode webm --output-path clip.webm`. The existing `--mode mp4` remains the
default and continues to invoke a host `ffmpeg` for H.264/AAC, providing the
initial native-viewer and legacy fallback without linking FFmpeg into this
wheel.

## Codec selection

VP9+Opus is preferred. On first use, the package encodes a deterministic
768×768/24-fps VP9 workload, excludes eight warmup frames, and compares the 24
measured frames' p90 native latency with the 41.667 ms frame interval. It caches
the native-library and CPU-specific decision below the user cache directory.
VP8 is selected only when VP9 exceeds that budget.

Pre-run or refresh the benchmark and retain both machine-readable and reviewable
evidence with:

```bash
flashdreams-webm-benchmark --refresh \
  --json-path webm-codec.json \
  --markdown-path webm-codec.md
```

Set `FLASHDREAMS_WEBM_CODEC=vp8` or `vp9` for a deliberate deployment override.

## Runtime contract

RGB frames are converted to I420 and encoded incrementally into a private VPx
packet spool. FlashDreams stages normalized interleaved PCM with timeline gaps,
padding, and truncation before native Opus encoding. libwebm muxes packets in
timestamp order at close. Only a successfully finalized staged file is
atomically moved to the requested destination; abort removes the packet, PCM,
and container staging without replacing an existing target.
