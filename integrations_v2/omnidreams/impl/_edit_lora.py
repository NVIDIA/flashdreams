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

"""Pre-merged text-edit LoRA deploy hook for mid-stream prompt swaps.

Deploys a ``guidance_distill/train_guidance.py`` checkpoint — a LoRA
distilled from the two-prompt edit guidance — so a plain prompt swap
responds at guided strength without the guidance's extra forward per
denoise step. Both weight sets (base and base-plus-delta) are cached at
load; toggling an edit window ``copy_``s the right set into the live
projection weights, so storage addresses survive and captured CUDA graphs
stay valid (the drift corrector's pointer-rebinding swap is not
graph-safe). Toggles happen only at edit-window boundaries — a few chunks
apart — so the copy cost (~1.6 GiB, sub-millisecond) is off the hot path.

Window semantics live in :class:`~omnidreams.transformer.TextEditGuidance`:
``CosmosTransformer.replace_text_embeddings`` builds a ``use_lora`` window
when a hook is attached, ``predict_flow`` activates the merged weights for
the window's chunks (including the KV-commit context forwards — the
checkpoint was trained to match the guided context forward too), and the
first forward after the countdown expires restores the base weights.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omnidreams.impl._module_utils import unwrap_compiled_module
from torch import Tensor

_LORA_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.k_proj",
    "cross_attn.v_proj",
    "cross_attn.output_proj",
)
"""Projections the guidance-distillation checkpoints were trained on.

Must match ``guidance_distill/train_guidance.py``'s ``LORA_TARGETS`` (same
substring rule, same ``named_modules`` walk) so the checkpoint's
load-order indices line up. ``cross_attn.`` does not match the multi-view
``cross_view_attn.`` modules.
"""


def _target_linears(network: nn.Module) -> list[nn.Linear]:
    """Target linears in checkpoint load order (the training-side walk)."""
    linears: list[nn.Linear] = []
    for mname, module in network.named_modules():
        for cname, child in module.named_children():
            full = f"{mname}.{cname}" if mname else cname
            if isinstance(child, nn.Linear) and any(t in full for t in _LORA_TARGETS):
                linears.append(child)
    return linears


class TextEditLoRA:
    """Two cached weight sets (base / edit) toggled per edit window.

    Args:
        network: The unwrapped ``CosmosDiTNetwork`` whose projection
            weights are toggled in place.
        checkpoint: ``train_guidance.py`` checkpoint (a dict whose
            ``"lora"`` entry maps load-order indices to A/B tensors;
            ``A_i`` at ``2i``, ``B_i`` at ``2i + 1``).
        scale: Gain on the LoRA delta. The checkpoint distills a fixed
            teacher strength, so ``1.0`` reproduces the evaluated deploy.
    """

    def __init__(
        self,
        network: nn.Module,
        checkpoint: Path | str,
        *,
        scale: float = 1.0,
    ) -> None:
        network = unwrap_compiled_module(network)
        linears = _target_linears(network)
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["lora"]
        assert len(sd) == 2 * len(linears), (
            f"edit-LoRA checkpoint has {len(sd)} tensors but the network "
            f"exposes {2 * len(linears)} ({len(linears)} target projections); "
            "target-list mismatch with the training recipe."
        )

        self._linears = linears
        self._base: list[Tensor] = []
        self._edit: list[Tensor] = []
        added_bytes = 0
        for i, lin in enumerate(linears):
            a = sd[2 * i].to(lin.weight.device, torch.float32)
            b = sd[2 * i + 1].to(lin.weight.device, torch.float32)
            base = lin.weight.detach().clone()
            w32 = base.to(torch.float32, copy=True)
            edit = w32.addmm_(b, a, alpha=scale).to(base.dtype)
            self._base.append(base)
            self._edit.append(edit)
            added_bytes += 2 * base.numel() * base.element_size()
        self.rank = int(sd[0].shape[0])
        self.added_bytes = added_bytes
        self.active = False

    def set_active(self, active: bool) -> None:
        """Copy the requested weight set into the live buffers (idempotent).

        In-place ``copy_`` so the weight storage addresses never change —
        captured CUDA graphs keep reading the same buffers and only the
        contents differ.
        """
        if active == self.active:
            return
        source = self._edit if active else self._base
        for lin, w in zip(self._linears, source):
            lin.weight.data.copy_(w)
        self.active = active

    def release_targets(self, linears: list[nn.Linear]) -> list[Tensor]:
        """Stop toggling the given live linears; return their fp32 deltas.

        Composition seam for the fused drift-corrector dispatch
        (:class:`omnidreams._drift_corrector.DriftCorrectorDispatch`): the
        dispatch becomes the sole writer of the released projections and
        folds the returned ``edit - base`` deltas into its per-state
        pre-merged weight sets, while this hook keeps toggling only the
        remaining (cross-attention) projections. Deltas are returned in
        the order of ``linears``.
        """
        assert not self.active, "release targets while the base weights are live"
        index = {id(lin): i for i, lin in enumerate(self._linears)}
        drop: set[int] = set()
        deltas: list[Tensor] = []
        for lin in linears:
            assert id(lin) in index, "linear is not one of this LoRA's targets"
            i = index[id(lin)]
            deltas.append(
                self._edit[i].to(torch.float32) - self._base[i].to(torch.float32)
            )
            drop.add(i)
        keep = [i for i in range(len(self._linears)) if i not in drop]
        self._linears = [self._linears[i] for i in keep]
        self._base = [self._base[i] for i in keep]
        self._edit = [self._edit[i] for i in keep]
        return deltas

    def describe(self) -> str:
        """One-line deploy description for startup logs."""
        return (
            f"text-edit LoRA r{self.rank} pre-merged on "
            f"{len(self._linears)} projections "
            f"(+{self.added_bytes / 2**20:.0f} MiB weight sets)"
        )
