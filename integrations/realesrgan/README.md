<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# flashdreams-realesrgan

Real-ESRGAN frame and video upsampling packaged as a FlashDreams workspace
integration. It provides:

- a reusable `RealESRGANUpsampler` Python API,
- a `RealESRGANPostProcessorConfig` for FlashDreams generated RGB video chunks,
- a `realesrgan-upsample` CLI for image and video files.

The architecture definitions follow the public Real-ESRGAN / BasicSR RRDBNet
and compact SRVGG network layouts so public checkpoints can be loaded directly.
Weights are downloaded at first use into
`$FLASHDREAMS_CACHE_DIR/realesrgan` or `~/.cache/flashdreams/realesrgan`.

## Install

From the repository root:

```bash
uv sync --package flashdreams-realesrgan --extra dev
```

The root workspace glob includes `integrations/realesrgan`, so editable
workspace usage works without extra path setup.

## CLI

Upsample one image:

```bash
uv run --package flashdreams-realesrgan realesrgan-upsample \
  --input /path/to/input.png \
  --output /path/to/input_2x.png \
  --scale 2
```

Upsample a video:

```bash
uv run --package flashdreams-realesrgan realesrgan-upsample \
  --input /path/to/input.mp4 \
  --output /path/to/input_2x.mp4 \
  --scale 2 \
  --tile 256 \
  --compile
```

Use Real-ESRGAN as a `flashdreams-run` postprocessor:

```bash
uv run --package flashdreams-realesrgan flashdreams-run \
  --postprocess.preset realesrgan \
  wan21-t2v-1.3b-480p
```

Useful flags:

| flag | description |
| --- | --- |
| `--scale {2,4}` | Output scale. Defaults to `2`. |
| `--model-name` | Public checkpoint name such as `RealESRGAN_x2plus` or `RealESRGAN_x4plus`. |
| `--model-path` | Local checkpoint path. Skips public download. |
| `--tile` | Tile size for lower peak VRAM. `0` processes whole frames. |
| `--fp32` | Disable fp16 CUDA inference. |
| `--compile` | Enable `torch.compile` with `mode="reduce-overhead"`. |
| `--compile-mode` | Override the `torch.compile` mode. |
| `--profile-warmup-frames` | Frames excluded from steady FPS metrics. Defaults to `10`. |
| `--max-frames` | Video-only frame cap for quick smoke tests. |

For `flashdreams-run`, Real-ESRGAN is selected with the registered
`--postprocess.preset realesrgan` preset. The preset uses the default 2x public
checkpoint on CUDA. Use the standalone `realesrgan-upsample` CLI or configure a
`RealESRGANPostProcessorConfig` directly when you need custom scale, tiling,
precision, checkpoint, or compile settings.

Video runs print two profile rows. `model_fps` is CUDA-event timing for the
Real-ESRGAN model call only. `pipeline_fps` includes image conversion,
CPU/GPU copies, padding/cropping, the model call, and output conversion.
`video_loop_fps` adds video frame read/write around the pipeline call.
`end_to_end_fps` is the whole video pass wall-clock throughput.

## FlashDreams Postprocess API

```python
from flashdreams.infra.postprocess import VideoPostprocessChainConfig
from realesrgan import RealESRGANPostProcessorConfig

postprocess = VideoPostprocessChainConfig(
    processors=(
        RealESRGANPostProcessorConfig(
            scale=2,
            model_name="RealESRGAN_x2plus",
            tile=256,
            compile_model=True,
            device="cuda",
        ),
    )
)
```

The postprocessor accepts generated RGB chunks in any FlashDreams video layout,
runs frame-local Real-ESRGAN inference, and returns `bvtchw` chunks in
`[-1, 1]`.

## Tests

CPU-safe tests use random weights and do not download checkpoints:

```bash
uv run --package flashdreams-realesrgan pytest integrations/realesrgan/tests/test_realesrgan.py -m ci_cpu
```

Checkpoint-backed GPU smoke test:

```bash
uv run --package flashdreams-realesrgan python integrations/realesrgan/scripts/gpu_smoke.py \
  --scale 2 \
  --device cuda
```

The smoke script downloads `RealESRGAN_x2plus` once, runs one CUDA fp16
forward pass on a `16x16` RGB frame, checks the `32x32` output, and prints
model load time, inference time, and peak allocated VRAM.
