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

"""FlashVSR recipe constants.

Numbers that pin the FlashVSR-v1.1 streaming contract: the chunk-target
table that the encoder accepts, the TC decoder's latent-channel split
(noise + bicubic-conditioning packing), and the noise/conditioning patch
size for the decoder's pixel-shuffle. Mirrors ``wan/constants.py`` and
``alpadreams/constants.py``.
"""

# Mirrors the legacy ``_CHUNK_TARGET = {5: 8, 13: 16, 8: 8, 16: 16}``: the
# encoder accepts these (raw_T -> padded_T) pairs at every AR step.
# Cold-start sizes (5 / 13) are pad-left replicated to 8 / 16 so the
# projector's 4-frame causal stride aligns; steady chunks (8 / 16) pass
# through unchanged.
FLASHVSR_CHUNK_FRAME_TARGETS: dict[int, int] = {5: 8, 13: 16, 8: 8, 16: 16}

# Number of raw frames per internal DiT iteration (``len_t * 4`` after the
# projector's 4x temporal compression). Used by ``FlashVSREncoder`` to
# derive ``cache.last_n_iters = T_padded // FLASHVSR_FRAMES_PER_DIT_ITER``.
FLASHVSR_FRAMES_PER_DIT_ITER: int = 8

# TC decoder latent layout.
#
# - ``FLASHVSR_DECODER_NOISE_CHANNELS``: clean latent channels coming out
#   of the DiT (post flow-match denoise).
# - ``FLASHVSR_DECODER_COND_CHANNELS``: bicubic-upres channels packed by
#   ``PixelShuffle3d(4, 8, 8)`` (3 RGB * 4 * 8 * 8 = 768) and concatenated
#   onto the noise as the TAEHV input.
# - ``FLASHVSR_DECODER_LATENT_CHANNELS``: their sum, which is what
#   ``TAEHV.latent_channels`` is sized for.
FLASHVSR_DECODER_NOISE_CHANNELS: int = 16
FLASHVSR_DECODER_COND_CHANNELS: int = 768
FLASHVSR_DECODER_LATENT_CHANNELS: int = (
    FLASHVSR_DECODER_NOISE_CHANNELS + FLASHVSR_DECODER_COND_CHANNELS
)

# ``PixelShuffle3d`` factors used by the TC decoder to pack the bicubic
# conditioning ``(B, 3, T_raw, H, W)`` into ``(B, 768, T_raw // 4,
# H // 8, W // 8)`` before concatenating onto the latent.
FLASHVSR_DECODER_CONDITION_PATCH: tuple[int, int, int] = (4, 8, 8)
