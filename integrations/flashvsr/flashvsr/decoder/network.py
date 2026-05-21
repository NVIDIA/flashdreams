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

"""FlashDreams-backed FlashVSR TCDecoder candidate.

This class is the live implementation; the parity reference is upstream
FlashVSR's ``examples/WanVSR/utils/TCDecoder.py``, loaded directly out
of the parity-check sibling tree by
``integrations/flashvsr/tests/parity_check/test_tcdecoder_parity.py``.
The candidate preserves the same FlashVSR-facing API (``decode_video``
+ a ``forward`` adapter that accepts the legacy ``parallel`` /
``show_progress_bar`` kwargs as a sink).
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init

from flashdreams.recipes.taehv.checkpoint import StateDictTransform
from flashdreams.recipes.taehv.impl import (
    TAEHV as _FlashDreamsTAEHV,
)
from flashdreams.recipes.taehv.impl import (
    MemBlock,
)
from flashdreams.recipes.taehv.impl import (
    TAEHVCache as FlashVSR_TAEHV_Cache,
)
from flashvsr.encoder.network import PixelShuffle3d

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


class IdentityConv2d(nn.Conv2d):
    """Same-shape Conv2d initialized to identity for FlashVSR deepening.

    Structurally identical to upstream FlashVSR's
    ``examples/WanVSR/utils/TCDecoder.py:IdentityConv2d`` so the parity
    test under ``integrations/flashvsr/tests/parity_check/`` agrees
    byte-for-byte.
    """

    def __init__(self, channels: int, kernel_size: int = 3, bias: bool = False):
        pad = kernel_size // 2
        super().__init__(channels, channels, kernel_size, padding=pad, bias=bias)
        with torch.no_grad():
            init.dirac_(self.weight)
            if self.bias is not None:
                self.bias.zero_()


def _apply_identity_deepen(
    blocks: nn.Sequential, how_many_each: int = 1
) -> nn.Sequential:
    """Insert ``how_many_each`` ``IdentityConv2d → ReLU`` pairs after every
    ``nn.ReLU`` in ``blocks``, returning a new ``nn.Sequential``.

    FlashVSR's ``TCDecoder.ckpt`` was trained on a deeper decoder than the
    canonical FlashDreams ``StreamingDecoder``: the FlashVSR authors grew the body by
    appending ``Conv→ReLU`` pairs after each existing ReLU, and the checkpoint
    stores weights for those extra convs. This helper reproduces that
    structural change in-process so checkpoint keys line up at load time.
    Mirrors upstream FlashVSR's ``examples/WanVSR/utils/TCDecoder.py``,
    which hard-codes the same deepening; the parity test under
    ``integrations/flashvsr/tests/parity_check/`` loads upstream's file
    directly and asserts chunk-by-chunk numerical agreement.

    The new convs are same-channels-in/out 3x3 and Dirac-initialized, so the
    deepened block is an identity until weights are loaded - i.e., a freshly
    constructed model produces the same outputs as the un-deepened one.

    Channel width is inferred from the layer that produced the activations
    feeding into each ReLU (the entry just before the ReLU in the new
    sequence): ``out_channels`` for a plain ``Conv2d``, or the trailing conv's
    ``out_channels`` for a ``MemBlock``. ReLUs preceded by anything else
    (e.g., an ``Upsample``) are left alone.

    Args:
        blocks: The original decoder body. Not mutated.
        how_many_each: How many ``IdentityConv2d → ReLU`` pairs to insert
            after each ReLU. ``1`` matches the FlashVSR checkpoint layout.

    Returns:
        A new ``nn.Sequential`` containing the original layers interleaved
        with the inserted identity pairs.
    """
    new_layers: list[nn.Module] = []
    for block in blocks:
        new_layers.append(block)
        if not isinstance(block, nn.ReLU):
            continue
        # ``out_channels`` is documented as ``int`` on ``nn.Conv2d`` but ty
        # types it as ``int | Tensor | Module`` because ``nn.Module.__getattr__``
        # erases the descriptor. Cast to int so ``IdentityConv2d(channels)``
        # resolves cleanly.
        channels: int | None = None
        prev = new_layers[-2] if len(new_layers) >= 2 else None
        if isinstance(prev, nn.Conv2d):
            channels = int(prev.out_channels)
        elif isinstance(prev, MemBlock):
            last_conv = prev.conv[-1]
            assert isinstance(last_conv, nn.Conv2d)
            channels = int(last_conv.out_channels)
        if channels is None:
            continue
        for _ in range(how_many_each):
            new_layers.append(IdentityConv2d(channels, kernel_size=3, bias=False))
            new_layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*new_layers)


class FlashVSR_TAEHV(_FlashDreamsTAEHV):
    """FlashVSR-configured TAEHV decoder using the shared FlashDreams backend.

    Subclass of :class:`flashdreams.recipes.taehv.impl.TAEHV` (imported
    here as ``_FlashDreamsTAEHV``) that bakes in the FlashVSR-tiny
    structural tweaks: the bicubic-conditioning ``PixelShuffle3d`` on the
    input side and the identity-deepening of the decoder body so the
    FlashVSR ``TCDecoder.ckpt`` keys line up at load time. The
    ``FlashVSR_`` prefix disambiguates from the canonical FlashDreams
    ``TAEHV`` parent class.

    The init sequence is deliberately explicit -- ``super().__init__`` is
    asked to stop after the meta-built decoder is in place
    (``checkpoint_path=None``), this class then mutates
    ``decoder.blocks``, and finally :meth:`load_from_checkpoint` does the
    actual weight load + ``torch.compile`` / CUDA-graph wiring on top of
    the already-deepened tree.
    """

    image_channels = 3

    def __init__(
        self,
        checkpoint_path: str,
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
        channels: tuple[int, int, int, int] = (256, 128, 64, 64),
        latent_channels: int = 16,
        use_cuda_graph: bool = False,
        use_compile: bool = False,
        warmup_iters: int = 2,
        state_dict_transform: StateDictTransform | None = None,
    ) -> None:
        # Pass ``checkpoint_path=None`` so the parent stops after meta
        # construction; we need to deepen ``decoder.blocks`` (so the
        # FlashVSR checkpoint keys line up) and ``compile_module`` wraps
        # ``self.decoder`` -- both of which have to happen on the final
        # module tree, before the actual load.
        super().__init__(
            checkpoint_path=None,
            decoder_time_upscale=decoder_time_upscale,
            decoder_space_upscale=decoder_space_upscale,
            channels=channels,
            latent_channels=latent_channels,
            clamp_output=False,
            use_cuda_graph=use_cuda_graph,
            use_compile=use_compile,
            warmup_iters=warmup_iters,
        )

        # Identity-deepen the decoder body to match the FlashVSR
        # checkpoint layout. Dirac-initialized so the deepened tree is an
        # identity until weights are loaded.
        self.decoder.blocks = _apply_identity_deepen(
            self.decoder.blocks, how_many_each=1
        )

        self.load_from_checkpoint(
            checkpoint_path, state_dict_transform=state_dict_transform
        )

        # TC decoder packs the bicubic conditioning into per-frame latent
        # channels: the cond's frame count may not be a clean multiple of 4
        # on cold-start chunks, so autopad frame 0; the downstream concat
        # along the channel axis expects the ``frame_first`` layout.
        self.pixel_shuffle = PixelShuffle3d(
            4, 8, 8, out_layout="frame_first", autopad_first_frame=True
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        cache: FlashVSR_TAEHV_Cache | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Concat the ``cond`` pixel-shuffle channels onto ``x`` and run :meth:`decode`.

        ``**kwargs`` is accepted as a sink for trailing keyword arguments
        forwarded by the legacy FlashVSR caller (``parallel``,
        ``show_progress_bar``) so call sites stay drop-in.
        """
        if cond is not None:
            x = torch.cat([self.pixel_shuffle(cond), x], dim=2)
        return self.decode(x, cache=cache, **kwargs)
