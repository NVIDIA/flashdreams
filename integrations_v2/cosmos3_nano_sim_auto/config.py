# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the checkpoint wants, as against what a caller might prefer.

Each of these describes the weights rather than how hard the model is being
asked to work. Run the schedule, the cache or the scope at anything else and
the result is a coherent but visibly degraded picture, with nothing in the
output or the log to say so. That is why they live in one module that every
front end reads: a value restated in a second place is a value that can drift
silently, and silently is the only way this particular mistake arrives.

The settings here are the ones more than one caller needs. A default that only
the batch command's argument parser means anything to stays with the parser.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from cosmos3_nano_sim_auto.impl.transformer.mask import AttentionScope

__all__ = [
    "CHECKPOINT_REPO_ID",
    "DEFAULT_ATTENTION_SCOPE",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_DENOISE_STEPS",
    "DEFAULT_HISTORY_FRAMES",
    "DEFAULT_SCENE",
    "DEFAULT_TEMPORAL_WINDOW_SECONDS",
    "denoise_schedule",
]

CHECKPOINT_REPO_ID = "nvidia/Cosmos3-Nano-Sim-Auto"
"""Hosted Sim-Auto transformer and companion LiDAR VAE repository."""

DEFAULT_CHECKPOINT = f"https://huggingface.co/{CHECKPOINT_REPO_ID}/resolve/main/model.safetensors.index.json"
"""GA transformer index; the checkpoint loader downloads and caches its shards."""

DEFAULT_SCENE = Path("clipgt-residential-iter1500")
"""Built-in prepared scene downloaded during application initialization."""

DEFAULT_ATTENTION_SCOPE: AttentionScope = "decomposed"
"""How the checkpoint this demo now runs was trained.

A scope is a property of the weights rather than a preference: a checkpoint
reads whatever it was trained to read, and one run under the wrong scope
generates a coherent but visibly degraded picture rather than an error. Nothing
in the output or the log says which happened, so the default has to name the
checkpoint in use and cannot stay on a conservative choice that is simply wrong
for it. ``all_views`` is the older stage 3 checkpoint's, and is still reachable
by flag.
"""

DEFAULT_DENOISE_STEPS = 4
"""Denoising steps per chunk, the reference schedule the model team baselines on.

The published checkpoint uses four distilled SDE steps with shift 5.
Shorter schedules are diagnostic overrides and are not the guide's configuration.
0 asks for the checkpoint's full schedule, which is also four.
"""

DEFAULT_HISTORY_FRAMES = 3
"""Generated frames the key/value cache keeps, besides the pinned opening frame.

Three is one chunk. Upstream's ``rolling_kv_cache_chunks=2`` includes its one
sink chunk, so it retains the opening frame plus the latest generated chunk.
An unbounded cache grows with elapsed time.
"""

DEFAULT_TEMPORAL_WINDOW_SECONDS = 0.4
"""How far back a query reads the other cameras under the decomposed scope.

The final stage-3 recipe uses this causal window and deduplicates the query's
own view between attention folds. Earlier weights retain their original overlap.
"""


def denoise_schedule(full: Sequence[float], steps: int | None) -> list[float] | None:
    """The checkpoint's sigma schedule, trimmed to its first ``steps``.

    A prefix and not a subsample: the sampler appends 0.0 and walks the list in
    pairs, so two sigmas leave ``[1.0, 0.9375, 0.0]``, two steps ending in the
    jump to clean that every schedule ends in. The model predicts a clean latent
    at each sigma, so a short schedule means a coarser final jump, not a latent
    left part-noisy.

    Checked here rather than in the parser because only the loaded checkpoint
    knows how many sigmas it has, and a schedule cannot be lengthened.

    Returns:
        None when nothing was asked for, leaving the choice to the checkpoint.
    """
    if steps is None:
        return None
    if steps > len(full):
        raise ValueError(
            f"--denoise-steps {steps} is more than the {len(full)} this checkpoint "
            "was distilled for; a schedule cannot be lengthened."
        )
    return list(full[:steps])
