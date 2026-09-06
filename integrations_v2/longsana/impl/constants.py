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

"""Released LongSana model constants and immutable artifact locations."""

from pathlib import Path

DEFAULT_VIDEO_HEIGHT = 480
"""Native LongSana release height in pixels."""

DEFAULT_VIDEO_WIDTH = 832
"""Native LongSana release width in pixels."""

DEFAULT_VIDEO_FPS = 16
"""Frame rate used by the public LongSana release."""

FIRST_LATENT_BLOCK_FRAMES = 11
"""First block size for the release's 261-frame latent rollout."""

LATENT_BLOCK_FRAMES = 10
"""Steady-state number of latent frames generated per AR block."""

MAX_ROPE_POSITION = 1024
"""Largest absolute latent position supported by the released RoPE table."""

MAX_ROLLOUT_BLOCKS = (
    1 + (MAX_ROPE_POSITION - FIRST_LATENT_BLOCK_FRAMES) // LATENT_BLOCK_FRAMES
)
"""Maximum complete rollout that fits the released absolute RoPE table."""

MOTION_SCORE = 10
"""Motion-score suffix used during LongSana self-forcing post-training."""

LONGSANA_REVISION = "48283a1b034cecdfaf412a01be2ae202d2432a85"
"""Immutable Hugging Face revision used for the public distilled checkpoint."""

LONGSANA_CHECKPOINT_PATH = (
    "https://huggingface.co/Efficient-Large-Model/"
    "LongSANA_2B_480p_self_forcing/resolve/"
    f"{LONGSANA_REVISION}/checkpoints/LongSANA_2B_480p_self_forcing.pt"
)
"""Public four-step LongSana generator checkpoint."""

SANA_VIDEO_REVISION = "7dd4f2fcddc7db57597238d728e1f430129827ff"
"""Immutable SANA-Video revision containing the Wan 2.1 VAE."""

LONGSANA_VAE_CHECKPOINT_PATH = (
    "https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p/resolve/"
    f"{SANA_VIDEO_REVISION}/vae/Wan2.1_VAE.pth"
)
"""Wan 2.1 VAE checkpoint paired with the LongSana release."""

LONGSANA_TEXT_CONFIG_PATH = str(
    Path(__file__).resolve().parent.parent / "resources" / "longsana_text.yaml"
)
"""Packaged text-encoder settings copied from the upstream release config."""

DEFAULT_DENOISING_TIMESTEPS = [1000, 960, 889, 727]
"""Four raw self-forcing timesteps from the public LongSana checkpoint."""
