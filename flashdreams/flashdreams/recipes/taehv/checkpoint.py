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

"""State-dict transforms for TAEHV checkpoint variants.

Each known checkpoint (``lighttae``, FlashVSR ``TCDecoder.ckpt``, ...)
declares its own remap next to its URL in the consuming config and
threads it into :class:`~flashdreams.recipes.taehv.impl.TAEHV` via the
``state_dict_transform`` kwarg. The reusable building blocks live here;
:func:`compose` glues them together.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

StateDictTransform = Callable[[Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]
"""Pure remap from one TAEHV state dict to another; chain via :func:`compose`."""


def legacy_to_blocks_keys(
    sd: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Re-key legacy ``decoder.<i>.*`` weights to ``decoder.blocks.<i>.*``.

    The current :class:`~flashdreams.recipes.taehv.impl.Decoder` wraps
    its ``Sequential`` in a ``blocks`` attribute, so older checkpoints
    whose keys flatten to ``decoder.<idx>.*`` need rewriting to line up.
    Keys already under ``decoder.blocks.`` (and keys outside the
    ``decoder.`` subtree) pass through unchanged.
    """
    return {
        (
            k.replace("decoder.", "decoder.blocks.", 1)
            if k.startswith("decoder.") and not k.startswith("decoder.blocks.")
            else k
        ): v
        for k, v in sd.items()
    }


def truncate_oversize_tgrow_weights(
    *,
    channels: tuple[int, int, int, int],
    decoder_time_upscale: tuple[bool, bool] = (True, True),
) -> StateDictTransform:
    """Build a transform that clips oversize ``TGrow`` weights to model strides.

    Some shipped TAEHV checkpoints store ``TGrow`` ``conv.weight`` for
    stride=2 even when the target model is built with stride=1 at that
    position; the returned transform keeps only the last-timestep slice
    (matching the model's expected ``conv.weight.shape[0]``).

    The Sequential indices encoded here (``7``, ``13``, ``19``) match
    the layout in :class:`~flashdreams.recipes.taehv.impl.Decoder`'s
    body; subclasses that mutate ``decoder.blocks`` between
    meta-construction and weight load (e.g. FlashVSR's identity-deepening)
    shift these indices and must use
    :func:`truncate_oversize_tgrow_weights_from_blocks` instead.

    Args:
        channels: ``Decoder`` block widths (``n_f``); used to compute
            per-``TGrow`` expected output widths without an instantiated
            model.
        decoder_time_upscale: Per-stage temporal-upsample flags; selects
            the expected stride for the second and third ``TGrow``.
    """
    expected_channels: dict[int, int] = {
        7: channels[0] * 1,
        13: channels[1] * (2 if decoder_time_upscale[0] else 1),
        19: channels[2] * (2 if decoder_time_upscale[1] else 1),
    }

    def transform(
        sd: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out = dict(sd)
        for idx, expected in expected_channels.items():
            key = f"decoder.blocks.{idx}.conv.weight"
            if key in out and out[key].shape[0] > expected:
                out[key] = out[key][-expected:]
        return out

    return transform


def truncate_oversize_tgrow_weights_from_blocks(
    decoder_blocks: nn.Sequential,
) -> StateDictTransform:
    """Model-aware variant of :func:`truncate_oversize_tgrow_weights`.

    Walks ``decoder_blocks`` once at build time, snapshotting the index
    and expected ``conv.weight.shape[0]`` of every ``TGrow`` layer, then
    returns a transform that clips any oversize entries. Handles
    subclasses that mutate ``decoder.blocks`` between meta-construction
    and weight load (e.g. FlashVSR's identity-deepening shifts every
    ``TGrow`` to a new index) without the caller having to spell the new
    layout out. Used as the generic fallback in
    :meth:`TAEHV.load_from_checkpoint`.

    Args:
        decoder_blocks: Live ``Decoder.blocks`` sequence to walk.
    """
    # Lazy import: ``TGrow`` lives in ``impl`` which imports this module
    # for ``StateDictTransform``, so a top-level import would deadlock.
    from flashdreams.recipes.taehv.impl import TGrow

    expected_channels: dict[int, int] = {}
    for i, layer in enumerate(decoder_blocks):
        if isinstance(layer, TGrow):
            expected_channels[i] = int(layer.conv.weight.shape[0])

    def transform(
        sd: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out = dict(sd)
        for idx, expected in expected_channels.items():
            key = f"decoder.blocks.{idx}.conv.weight"
            if key in out and out[key].shape[0] > expected:
                out[key] = out[key][-expected:]
        return out

    return transform


def compose(*transforms: StateDictTransform) -> StateDictTransform:
    """Compose :data:`StateDictTransform` callables left-to-right.

    ``compose(f, g)(sd)`` is ``g(f(sd))``. Returns ``sd`` unchanged when
    given no transforms.
    """

    def composed(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = dict(sd)
        for t in transforms:
            out = t(out)
        return out

    return composed
