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

"""Static-background demo rollouts: locked-off camera, moving foreground.

The flagship corrector use case: a static camera aligns *with* the anchoring
pull (background stability is the point), so the demo checks that foreground
motion and story flow survive it. Each scene from ``demo_prompts.txt`` rolls
under an all-identity pose (action class 0 throughout) for every gain in
``GAINS``, seeded by the matching ``first_frames/demo`` image. The last
scene is the new-element entrance probe (the anchoring pull resists novel
content — verify it doesn't). Scoring is a separate pass (``score_drift.py``
with ``EVAL_OUT`` pointed here); labeled side-by-sides come from
``make_sbs.py``.

Run from the repo root::

    GAINS=0,0.7,1.0 NUM_CHUNK=24 \
        .venv/bin/python integrations/hy_worldplay/drift_correction/demo_static.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from _rollout import (
    build_runner,
    capture_rollout,
    install_alpha_gate,
    parse_gain_token,
)

## Demo configuration

_BASE = Path("integrations/hy_worldplay/drift_correction")

PROMPTS_FILE = Path(os.environ.get("PROMPTS_FILE", str(_BASE / "demo_prompts.txt")))
"""One scene prompt per line; line ``i`` pairs with ``frame_{i:04d}.png``."""

FRAMES_DIR = Path(
    os.environ.get("FRAMES_DIR", str(_BASE / "outputs/first_frames/demo"))
)
"""Seed frames from ``gen_first_frames.py`` run on ``PROMPTS_FILE``."""

LORA = os.environ.get("LORA", str(_BASE / "outputs/lora_v2.pt"))
"""Corrector checkpoint; gain 0 rows double as the base config."""

GAINS = tuple(
    parse_gain_token(g)
    for g in os.environ.get("GAINS", "0,0.7,1.0").split(",")
    if g.strip()
)
"""Deployed LoRA gains; 0 = base, ``gate``/``gate0.5`` = per-step
``alpha*(t) x scale``. The entrance scene needs corrector-off/on/strong."""

NUM_CHUNK = int(os.environ.get("NUM_CHUNK", "24"))
"""Rollout horizon (24 chunks matches the T1 eval, ~19 s)."""

SEED = int(os.environ.get("SEED", "5042"))
"""Diffusion seed, matched across configs."""

OUT_DIR = Path(os.environ.get("DEMO_OUT", str(_BASE / "outputs/demo_static")))

_DEFAULT_INTRINSIC = [
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
]
"""Same 1920x1080 intrinsic the pose-string parser stamps on every frame."""


def write_static_pose(n_latents: int, path: Path) -> Path:
    """Write an all-identity pose JSON covering ``n_latents`` latents.

    Identity extrinsics give zero relative motion, so the action labels
    resolve to class 0 (no translation, no rotation) — a locked-off camera
    in the upstream grammar, which has no explicit "stay" token.
    """
    eye = np.eye(4).tolist()
    poses = {
        str(i): {"extrinsic": eye, "K": _DEFAULT_INTRINSIC}
        for i in range(n_latents + 1)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(poses))
    return path


def main() -> None:
    torch.set_grad_enabled(False)
    prompts = [
        line.strip() for line in PROMPTS_FILE.read_text().splitlines() if line.strip()
    ]
    frames = [FRAMES_DIR / f"frame_{i:04d}.png" for i in range(len(prompts))]
    missing = [f.name for f in frames if not f.exists()]
    assert not missing, (
        f"seed frames missing under {FRAMES_DIR}: {missing}; run "
        "gen_first_frames.py with PROMPTS_FILE/OUT_DIR pointed at the demo set."
    )

    pose_json = write_static_pose(NUM_CHUNK * 4, OUT_DIR / "static_pose.json")

    runner = None
    network = None
    mode: dict = {"gain": 0.0}
    for gain in GAINS:
        if isinstance(gain, tuple):
            config = (
                "corrgate"
                if gain[1] == 1.0
                else f"corrgate{gain[1]:.2f}".replace(".", "")
            )
        else:
            config = "base" if gain == 0 else f"corr{gain:.2f}".replace(".", "")
        for i, (prompt, image_path) in enumerate(zip(prompts, frames)):
            mp4 = OUT_DIR / config / f"scene{i}_s{SEED}.mp4"
            if mp4.exists():
                print(f"{config}/{mp4.stem}: exists, skipping", flush=True)
                continue
            if runner is None:
                runner = build_runner(
                    num_chunk=NUM_CHUNK,
                    pose=str(pose_json),
                    output_dir=OUT_DIR,
                    image_path=image_path,
                )
                from _lora import apply_lora, load_lora, unwrap_compiled

                network = unwrap_compiled(
                    runner.pipeline.diffusion_model.transformer.network
                )
                apply_lora(network)
                load_lora(network, LORA)
                install_alpha_gate(runner, network, mode)
            else:
                runner.config.image_path = image_path
            from _lora import set_lora_scale

            mode["gain"] = gain
            if not isinstance(gain, tuple):
                assert network is not None  # bound with the runner on first build
                set_lora_scale(network, gain)
            runner.config.prompt = prompt
            print(f"{config}/{mp4.stem}: rolling {NUM_CHUNK} chunks ...", flush=True)
            capture_rollout(runner, noise_seed=SEED, mp4_path=mp4)
    print(f"done -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
