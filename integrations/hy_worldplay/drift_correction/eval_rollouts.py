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

"""Paired closed-loop eval rollouts: frozen base vs +corrector at scale 1.

Generates long OOD rollouts (held-out first frames, non-loop trajectories,
matched seeds) for each config and writes MP4s under
``outputs/eval/{base,corr}/``. Scoring (Delta-MUSIQ, dynamic-degree guard,
frame strips) is a separate pass -- see ``score_drift.py``.

Run from the repo root::

    NUM_CHUNK=40 LORA=outputs/lora_v1_pilot.pt \
        .venv/bin/python integrations/hy_worldplay/drift_correction/eval_rollouts.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from _rollout import build_runner, capture_rollout, install_alpha_gate

## Eval configuration

_BASE = Path("integrations/hy_worldplay/drift_correction")

LORA = os.environ.get("LORA", str(_BASE / "outputs/lora_v1_pilot.pt"))
"""Corrector checkpoint for the ``corr`` config."""

NUM_CHUNK = int(os.environ.get("NUM_CHUNK", "40"))
"""Rollout horizon (40 chunks = 160 latents = ~32s at 16 fps)."""

POSES = (
    "w-40, right-3, w-40, left-3, w-73",
    "w-30, d-10, w-30, a-10, w-79",
    "w-60, right-2, w-50, right-2, w-45",
)
"""Non-loop eval trajectories (159 steps each = ``NUM_CHUNK * 4 - 1``);
distinct from the training strafe loops."""

IMAGES_DIR = os.environ.get("IMAGES_DIR", "")
"""Held-out first frames; empty -> the upstream sample image."""

SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "5042,5043").split(","))
"""Diffusion seeds; every (pose, image, seed) cell runs in every config."""

N_IMAGES = int(os.environ.get("N_IMAGES", "0"))
"""Cap on held-out images (0 = all); finals trade breadth for horizon."""

PROMPT = os.environ.get("PROMPT", "")
"""Text-prompt override. The integration default ("ancient Athens") is
mismatched to most seed images and tugs content toward off-scene
structures/figures; pass a scene-matched or neutral prompt for evals."""

OUT_DIR = Path(os.environ.get("EVAL_OUT", str(_BASE / "outputs/eval")))


def main() -> None:
    torch.set_grad_enabled(False)
    images = [None]
    if IMAGES_DIR:
        p = Path(IMAGES_DIR)
        images = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
        assert images, f"no images under {p}"
        if N_IMAGES:
            images = images[:N_IMAGES]

    # Config name -> deployed LoRA gain. ``GAINS=0.7,0.85`` adds partial-gain
    # rows (named ``corr070`` etc.); ``TGATE=1`` adds the per-step
    # alpha*(t)-gated row, ``TGATE=1,0.5`` also the gate x 0.5 composition
    # (``corrgate050``).
    configs: dict[str, float | tuple[str, float]] = {"base": 0.0, "corr": 1.0}
    for g in os.environ.get("GAINS", "").split(","):
        if g.strip():
            configs[f"corr{float(g):.2f}".replace(".", "")] = float(g)
    for s in os.environ.get("TGATE", "").split(","):
        if s.strip():
            scale = float(s)
            name = (
                "corrgate" if scale == 1.0 else f"corrgate{scale:.2f}".replace(".", "")
            )
            configs[name] = ("gate", scale)

    runner = None
    network = None
    mode: dict = {"gain": 0.0}
    for config, gain in configs.items():
        for pi, pose in enumerate(POSES):
            for ii, image_path in enumerate(images):
                for seed in SEEDS:
                    name = f"p{pi}_i{ii}_s{seed}"
                    mp4 = OUT_DIR / config / f"{name}.mp4"
                    if mp4.exists():
                        print(f"{config}/{name}: exists, skipping", flush=True)
                        continue
                    if runner is None:
                        runner = build_runner(
                            num_chunk=NUM_CHUNK,
                            pose=pose,
                            output_dir=OUT_DIR,
                            image_path=image_path,
                        )
                        from _lora import apply_lora, load_lora, set_lora_scale

                        network = runner.pipeline.diffusion_model.transformer.network
                        if hasattr(network, "_orig_mod"):
                            network = network._orig_mod
                        apply_lora(network)
                        load_lora(network, LORA)
                        if PROMPT:
                            runner.config.prompt = PROMPT
                        install_alpha_gate(runner, network, mode)
                    else:
                        runner.config.pose = pose
                        if image_path is not None:
                            runner.config.image_path = image_path
                    from _lora import set_lora_scale

                    mode["gain"] = gain
                    if not isinstance(gain, tuple):
                        set_lora_scale(network, gain)
                    print(
                        f"{config}/{name}: rolling {NUM_CHUNK} chunks ...", flush=True
                    )
                    capture_rollout(runner, noise_seed=seed, mp4_path=mp4)
    print(f"done -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
