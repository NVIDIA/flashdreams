# SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
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
"""Robotwin constants for the LingBot-VA FlashDreams integration."""

from __future__ import annotations

RUNNER_NAME_ROBOTWIN_I2AV = "lingbot-va-robotwin-i2av"
DEFAULT_CHECKPOINT_ROOT = "robbyant/lingbot-va-posttrain-robotwin"
DEFAULT_PROMPT = (
    "Grab the medium-sized white mug, rotate it, place it on the table, "
    "and hook it onto the smooth dark gray rack."
)

ROBOTWIN_HEIGHT = 256
ROBOTWIN_WIDTH = 320
ROBOTWIN_COMPOSITE_HEIGHT = ROBOTWIN_HEIGHT + ROBOTWIN_HEIGHT // 2
ROBOTWIN_VAE_TEMPORAL_SCALE = 4
ROBOTWIN_ATTENTION_WINDOW = 72
ROBOTWIN_FRAME_CHUNK_SIZE = 2
ROBOTWIN_ACTION_DIM = 30
ROBOTWIN_ACTION_PER_FRAME = 16
ROBOTWIN_GUIDANCE_SCALE = 5.0
ROBOTWIN_ACTION_GUIDANCE_SCALE = 1.0
ROBOTWIN_VIDEO_INFERENCE_STEPS = 25
ROBOTWIN_ACTION_INFERENCE_STEPS = 50
ROBOTWIN_SNR_SHIFT = 5.0
ROBOTWIN_ACTION_SNR_SHIFT = 1.0
ROBOTWIN_PATCH_SIZE = (1, 2, 2)
ROBOTWIN_OBS_CAM_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ROBOTWIN_USED_ACTION_CHANNEL_IDS = tuple(
    list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
)
ROBOTWIN_ACTION_Q01 = (
    -0.06172713458538055,
    -3.6716461181640625e-05,
    -0.08783501386642456,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -0.3547105032205582,
    -1.3113021850585938e-06,
    -0.11975435614585876,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    *([0.0] * 16),
)
ROBOTWIN_ACTION_Q99 = (
    0.3462600058317184,
    0.39966784834861746,
    0.14745532035827624,
    1.0,
    1.0,
    1.0,
    1.0,
    0.034201726913452024,
    0.39142737388610793,
    0.1792279863357542,
    1.0,
    1.0,
    1.0,
    1.0,
    *([0.0] * 14),
    1.0,
    1.0,
)

ROBOTWIN_LATENT_HEIGHT = (ROBOTWIN_HEIGHT // 16 * 3) // 2
ROBOTWIN_LATENT_WIDTH = ROBOTWIN_WIDTH // 16
ROBOTWIN_LATENT_CHANNELS = 48
ROBOTWIN_LATENT_TOKEN_PER_CHUNK = (
    ROBOTWIN_FRAME_CHUNK_SIZE
    * ROBOTWIN_LATENT_HEIGHT
    * ROBOTWIN_LATENT_WIDTH
    // (ROBOTWIN_PATCH_SIZE[0] * ROBOTWIN_PATCH_SIZE[1] * ROBOTWIN_PATCH_SIZE[2])
)
ROBOTWIN_ACTION_TOKEN_PER_CHUNK = ROBOTWIN_FRAME_CHUNK_SIZE * ROBOTWIN_ACTION_PER_FRAME
