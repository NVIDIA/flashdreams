<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# flashdreams-lingbot-va

LingBot-VA Image-to-Action-Video (I2AV) integration, packaged as
a [`flashdreams`](../..) plugin.

This plugin adapts [LingBot-VA](https://github.com/robbyant/lingbot-va) into the
standard flashdreams runner/pipeline interface, achieving **2.3× speedup** over
the original repository implementation. Note that the upstream repo wraps models
with FSDP; compared to the original implementation with FSDP removed, the
speedup is **1.48×**.


## Install

This plugin is a workspace member in the repo-root `pyproject.toml`, which
is included automatically when you set up the flashdreams environment:

```bash
uv sync
```

No separate install step is needed.

## Run

```bash
uv run flashdreams-run lingbot-va-robotwin-i2av \
    --input-image-dir assets/example_data/lingbot-va/robotwin \
    --output-dir outputs/lingbot_va/robotwin_i2av \
    --checkpoint-root /path/to/lingbot-va-posttrain-robotwin \
    --num-chunks 10 \
    --benchmark True
```

### CLI arguments

| flag | type | default | description |
| --- | --- | --- | --- |
| `--checkpoint-root` | str | `robbyant/lingbot-va-posttrain-robotwin` | Local path or HuggingFace repo ID for model weights. Must contain `transformer/`, `vae/`, `text_encoder/`, `tokenizer/` subdirs. |
| `--input-image-dir` | path | `assets/example_data/lingbot-va/robotwin` | Directory containing three observation camera PNGs (see below). |
| `--output-dir` | path | `outputs/lingbot_va/robotwin_i2av` | Where to write `demo.mp4`, `actions.npy`, `latents.pt`, and timing JSON. |
| `--prompt` | str | `"Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack."` | Text prompt describing the manipulation task. Can also be a path to a `.txt` file. |
| `--num-chunks` | int | `10` | Number of autoregressive chunks to generate. Each chunk produces `frame_chunk_size` (2) video frames and `action_per_frame × frame_chunk_size` (32) action steps. |
| `--seed` | int | `42` | Random seed for diffusion sampling. |
| `--benchmark` | bool | `False` | Print per-chunk and total pipeline timing, and save `timing_flashdreams.json`. |
| `--compile-network` | bool | `True` | Apply `torch.compile` to the DiT for faster inference. Set `False` for debugging. |
| `--enable-offload` | bool | `False` | Offload VAE/text-encoder to CPU after use to reduce VRAM (slower). |
| `--save-video` | bool | `True` | Decode latents and save `demo.mp4`. |
| `--save-actions` | bool | `True` | Save predicted actions to `actions.npy`. |
| `--num-inference-steps` | int | `25` | Diffusion steps for video denoising. |
| `--action-num-inference-steps` | int | `50` | Diffusion steps for action denoising. |
| `--guidance-scale` | float | `5.0` | Classifier-free guidance scale for video. |
| `--action-guidance-scale` | float | `1.0` | Classifier-free guidance scale for actions. |
| `--snr-shift` | float | `5.0` | Flow-match sigma shift for video scheduler. |
| `--action-snr-shift` | float | `1.0` | Flow-match sigma shift for action scheduler. |

### Input images

The runner expects these files under `--input-image-dir`:

- `observation.images.cam_high.png`
- `observation.images.cam_left_wrist.png`
- `observation.images.cam_right_wrist.png`

### Outputs

| file | description |
| --- | --- |
| `demo.mp4` | Decoded video (all chunks concatenated). |
| `actions.npy` | Predicted actions array, shape `(num_chunks × action_per_frame × frame_chunk_size, action_dim)`. |
| `latents.pt` | Raw latent tensors before VAE decode. |
| `timing_flashdreams.json` | Per-chunk and total timing (only when `--benchmark True`). |
