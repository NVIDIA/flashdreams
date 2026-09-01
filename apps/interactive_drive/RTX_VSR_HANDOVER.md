<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Interactive Drive RTX VSR handover

## Scope

This branch exposes the existing `rtx-super-resolution` post-processing preset
in the regular OmniDreams Interactive Drive HUD. The checkbox starts off, and a
change is applied on the model-owned path while marking the current rollout for
a restart. This keeps processed and unprocessed chunks out of the same rollout.

The implementation reuses `VideoPostprocessStream`, `VideoOutputStream`, and
the existing NVIDIA VFX-backed processor. It does not duplicate RTX VSR model
or tensor-conversion code in the v2 application.

The performance variants do not select RTX VSR by default. The generic
`--postprocess-preset` and `--[no-]postprocess-enabled` options remain available
for explicit runs.

## Platform finding

The development host was:

```text
CPU architecture: aarch64
Python: 3.12.13
GPU: NVIDIA GB300, compute capability 10.3
Driver: 595.71.05
```

The correct workspace package is `flashdreams-omnidreams`, not `omnidreams`.
The following command resolved the workspace but could not install RTX VSR on
this host:

```bash
uv sync --package flashdreams-omnidreams \
    --extra interactive-drive --extra rtx-postprocess
```

`nvidia-vfx==0.1.0.1` reported wheels for Linux x86_64 and Windows, but no
Linux ARM64 wheel. Consequently, `nvvfx` is unavailable on the Grace/GB300
host. The base application environment installs successfully without the RTX
extra:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive
```

This is an external binary-distribution limitation, not a failed source build
inside FlashDreams. Real RTX VSR execution was therefore not attempted on the
ARM64 host.

## x86_64 validation procedure

From the repository root on a supported Linux x86_64 NVIDIA system:

1. Install Interactive Drive and the VFX binding:

   ```bash
   uv sync --package flashdreams-omnidreams \
       --extra interactive-drive --extra rtx-postprocess --group test
   ```

2. Confirm that the binding imports:

   ```bash
   uv run python -c "import nvvfx; print(nvvfx.VideoSuperRes)"
   ```

3. Run the focused CPU contract tests:

   ```bash
   uv run --no-sync pytest \
       apps/interactive_drive/interactive_drive/tests/test_application.py \
       flashdreams/tests/test_rtx_super_resolution_postprocess.py \
       -m ci_cpu -q
   ```

4. Start the interactive application:

   ```bash
   uv run flashdreams-run-v2 interactive-drive-omnidreams \
       --mode webrtc --port 8089
   ```

   Add `--host 0.0.0.0` for a remote browser, or pass a local scene after the
   application separator:

   ```bash
   uv run flashdreams-run-v2 interactive-drive-omnidreams \
       --mode webrtc --host 0.0.0.0 --port 8089 -- \
       --scene /path/to/scene.usdz
   ```

5. Verify the HUD behavior:

   - **Post-processing** is visible and initially unchecked.
   - Enabling it reports that the rollout is restarting.
   - The following rollout uses RTX VSR for every generated chunk.
   - Disabling it restarts again and returns to unprocessed chunks.
   - Repeated toggles close the previous VFX session without stale frames or
     CUDA errors.

## Validation completed on ARM64

The following passed using the CPU fake VFX backend:

```text
34 passed
```

Coverage includes application configuration, HUD toggle propagation, rollout
restart state, RTX VSR layout/range conversion, VFX lifecycle, and error paths.
The v2 CLI also discovered `interactive-drive-omnidreams` and exposed:

```text
--postprocess-preset
--postprocess-enabled / --no-postprocess-enabled
```

## Items to inspect on x86_64

- Measure first-enable latency. The VFX effect is lazy-loaded on the first
  processed chunk.
- Confirm VFX output and PyTorch run on the intended CUDA device and stream.
- Watch VRAM across several on/off cycles to confirm `flush()` releases each
  VFX effect.
- Validate displayed resolution and image quality. The shipped
  `rtx-super-resolution` preset scales frames by 2, while the Interactive Drive
  HUD renders at the session dimensions. The UI preparation path may resize
  the enhanced frame back to the HUD dimensions; verify that this produces the
  intended quality, or decide whether model and presentation dimensions need
  separate configuration.
- Record post-processing latency and confirm the model/presentation queues do
  not accumulate unexpected backpressure.

No GPU generation, quality comparison, or VFX performance benchmark was run on
the ARM64 development host.
