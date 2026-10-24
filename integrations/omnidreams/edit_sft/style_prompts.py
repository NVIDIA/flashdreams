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

"""Style-slug -> edit-prompt bank for the style-SFT run (Tier-2b, restyles).

Each JoyAI style slug (``style_pairs/<uuid>__<slug>.mp4``) maps to the
DECLARATIVE scene-description phrasing of its edit instruction
(``style_pairs/instructions.json`` / the audition job files): the DiT's
text conditioning was trained on scene descriptions, not imperative edit
commands, so "Restyle the scene as a bright arcade racing game world ..."
becomes "A bright arcade racing game world ...". The ``_v2`` audition
slugs reran the same styles with a layout-preservation suffix; the suffix
is a JoyAI-side control, not a scene property, so they share their base
slug's prompt (the tensors alias, so ``torch.save`` stores them once).
"""

from __future__ import annotations

STYLE_PROMPTS: dict[str, str] = {
    "arcade_racer": (
        "A bright arcade racing game world with exaggerated saturated "
        "colors, clean stylized surfaces, and a cheerful sunny palette."
    ),
    "cartoon_cel": (
        "A colorful cartoon video game world with cel-shaded surfaces, "
        "bold outlines, and vivid saturated colors."
    ),
    "anime": (
        "A Japanese anime style world with painted textures, soft "
        "shading, and vibrant colors."
    ),
    "toy_world": (
        "A miniature toy world where everything has glossy plastic "
        "surfaces and bright playful colors."
    ),
    "lowpoly": (
        "A low-poly stylized video game environment with flat-shaded "
        "geometric surfaces and vivid colors."
    ),
    "comic_ink": (
        "A comic book style world with bold black ink outlines, halftone "
        "shading, and vivid flat colors."
    ),
    "pixel_art": (
        "Retro 16-bit pixel art video game graphics with visible pixels "
        "and a bright limited color palette."
    ),
    "cyberpunk_neon": (
        "A neon-lit cyberpunk night city with glowing signs and rain-slicked streets."
    ),
    "watercolor": (
        "A soft watercolor painting of the scene with visible brush "
        "strokes and paper texture."
    ),
}
STYLE_PROMPTS["anime_v2"] = STYLE_PROMPTS["anime"]
STYLE_PROMPTS["cartoon_cel_v2"] = STYLE_PROMPTS["cartoon_cel"]

DEFAULT_SKIP_STYLES: tuple[str, ...] = ("cyberpunk_neon", "watercolor")
"""Styles rejected at the audition (scene re-imagined / washed out) —
excluded from training by default even if the VLM filter passes them."""


def clip_key(uuid: str) -> str:
    """Return the embedding-dict key of a sample clip's own prompt.

    Args:
        uuid: ``nvidia/omni-dreams-samples`` single-view clip UUID.

    Returns:
        The key under which ``precompute_style.py`` stores the clip
        prompt's text embeddings (style prompts live under the raw slug).
    """
    return f"clip:{uuid}"
