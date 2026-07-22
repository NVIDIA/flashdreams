# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthesize first-frame images from prompts with the Wan 2.2 TI2V-5B base.

Keeps the whole corrector stack zero-real-data (the av2s regime): prompts ->
base-model T2V chunk 0 (no history, single-shot within the chunk) -> frame 0
saved as PNG. These images seed :mod:`build_pairs` rollouts (train split) and
the held-out eval set.

Prompt source: ``MovieGenVideoBench_extended.txt`` (same family the reference
av2s eval used); train frames draw from the file's head, eval frames from
index 700+ so neither overlaps the reference's 128-prompt eval indices.

Run from the repo root::

    N_FRAMES=40 SPLIT=train .venv/bin/python \
        integrations/hy_worldplay/drift_correction/gen_first_frames.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Must precede the first CUDA allocation (shared-GPU headroom).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

## Generation configuration

PROMPTS_FILE = Path(os.environ.get("PROMPTS_FILE", "prompts.txt"))
"""One prompt per line (e.g. Self-Forcing's
``MovieGenVideoBench_extended.txt``); set via the ``PROMPTS_FILE`` env var."""

N_FRAMES = int(os.environ.get("N_FRAMES", "40"))
SPLIT = os.environ.get("SPLIT", "train")
"""``train`` draws prompts from line 0 up; ``eval`` from line 700 up."""

OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        f"integrations/hy_worldplay/drift_correction/outputs/first_frames/{SPLIT}",
    )
)

HEIGHT, WIDTH = 480, 832
"""Generation resolution. Below HY-WorldPlay's native 704x1280 on purpose:
the runner resizes/crops seed images anyway, and the full-res eager decode
does not fit next to co-tenant jobs (~17 GiB single permute)."""

SEED = 7000 if SPLIT == "train" else 8000
"""Base diffusion seed; frame ``i`` uses ``SEED + i``."""

_EVAL_PROMPT_OFFSET = 700


def main() -> None:
    torch.set_grad_enabled(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = [
        line.strip() for line in PROMPTS_FILE.read_text().splitlines() if line.strip()
    ]
    start = 0 if SPLIT == "train" else _EVAL_PROMPT_OFFSET
    picked = prompts[start : start + N_FRAMES]
    assert len(picked) == N_FRAMES, (
        f"prompt file has only {len(prompts)} lines; cannot draw {N_FRAMES} "
        f"from offset {start}."
    )

    from flashdreams.infra.config import derive_config
    from wan22.config import WAN22_TI2V_5B_DIT_DIFFUSERS_PATH, PIPELINE_WAN22_TI2V_5B

    # T2V mode: the base pipeline asserts encoder-None when no image is
    # given; VAE graphs off for co-tenant headroom. Upstream sharded the
    # diffusers repo, so the single-file constant 404s -- point at the
    # sharded index (already in the local HF cache), which the loader
    # supports natively.
    cfg = derive_config(
        PIPELINE_WAN22_TI2V_5B,
        encoder=None,
        decoder=dict(use_cuda_graph=False),
        diffusion_model=dict(
            transformer=dict(
                checkpoint_path=WAN22_TI2V_5B_DIT_DIFFUSERS_PATH + ".index.json",
                # I2V-only mechanisms; both require an I2VCtrl input.
                stamp_image_latent=False,
                ti2v_first_frame_per_token_timestep=False,
            ),
        ),
    )
    pipe = cfg.setup().to("cuda").eval()
    dm = pipe.diffusion_model
    device = next(pipe.parameters()).device

    from PIL import Image

    for i, prompt in enumerate(picked):
        out_path = OUT_DIR / f"frame_{start + i:04d}.png"
        if out_path.exists():
            print(f"{out_path.name}: exists, skipping", flush=True)
            continue
        dm._rng = torch.Generator(device=device).manual_seed(SEED + i)
        # Release the ~11 GiB text encoder after each prompt: the 704x1280
        # eager VAE decode needs the headroom next to co-tenant jobs, and
        # the per-prompt reload (~15 s) is cheap at this batch size.
        cache = pipe.initialize_cache(
            text=[prompt],
            image=None,
            height=HEIGHT // 16,
            width=WIDTH // 16,
        )
        video = pipe.generate(0, cache)  # [*, T, C, H, W] in [-1, 1]
        pipe.finalize(0, cache)
        frame = video[0, 0] if video.ndim == 5 else video[0]
        frame = ((frame.float().clamp(-1, 1) + 1) * 127.5).round().byte()
        Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()).save(out_path)
        print(f"{out_path.name}: saved ({prompt[:60]}...)", flush=True)
        del cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
