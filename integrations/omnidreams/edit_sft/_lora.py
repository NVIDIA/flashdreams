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

"""LoRA corrector r_phi as low-rank deltas on the frozen Cosmos DiT.

Vendored from the Clean Forcing training infra (PR #398) so this pipeline
is self-contained; consolidate when that lands.

Mirror of the HY port's ``_lora.py`` with the Cosmos module names: the
blocks' self-attention projections are
``blocks.{i}.self_attn.{q,k,v,output}_proj``.
"""

import torch
import torch.nn as nn
from torch import Tensor

DEFAULT_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
)
"""Module-name suffixes wrapped by :func:`apply_lora`."""


class LoRALinear(nn.Module):
    """Frozen base linear plus a runtime-gated low-rank delta.

    ``B`` is zero-initialized so the wrapped module starts as an exact
    identity over the base (any ``scale`` gives the base output until
    training moves ``B``). ``scale`` is the runtime gain: ``0`` recovers the
    frozen base (the clean-history teacher pass), ``1`` the corrected pass.
    The A/B path runs in fp32 regardless of the base dtype.
    """

    def __init__(self, base: nn.Linear, rank: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / rank)
        nn.init.zeros_(self.B.weight)
        self.scale = 1.0

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.scale != 0:
            delta = self.B(self.A(x.to(self.A.weight.dtype)))
            out = out + self.scale * delta.to(out.dtype)
        return out


def unwrap_compiled(network: object) -> nn.Module:
    """Return the eager module behind a ``torch.compile`` wrapper, if any.

    Accepts ``object``: the DiT lives on the transformer as a plain
    attribute, which the type checker resolves through ``nn.Module``'s
    ``__getattr__`` union.
    """
    inner = getattr(network, "_orig_mod", network)
    assert isinstance(inner, nn.Module), type(inner)
    return inner


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> list[str]:
    """Wrap every target linear in ``model`` with a :class:`LoRALinear`.

    Args:
        model: The (unwrapped) DiT network.
        rank: LoRA rank.
        targets: Module-name suffixes to wrap.

    Returns:
        Fully qualified names of the wrapped modules.
    """
    wrapped = []
    for mname, module in list(model.named_modules()):
        for cname, child in list(module.named_children()):
            full = f"{mname}.{cname}" if mname else cname
            if isinstance(child, nn.Linear) and any(t in full for t in targets):
                setattr(module, cname, LoRALinear(child, rank).to(child.weight.device))
                wrapped.append(full)
    return wrapped


def set_lora_scale(model: nn.Module, scale: float) -> None:
    """Set the runtime gain on every :class:`LoRALinear` in ``model``."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.scale = scale


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect the trainable A/B parameters in deterministic module order."""
    ps: list[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            ps += list(m.A.parameters()) + list(m.B.parameters())
    return ps


def save_lora(model: nn.Module, path) -> None:
    """Save the LoRA parameters (index-keyed, CPU) to ``path``."""
    torch.save(
        {"lora": {i: p.detach().cpu() for i, p in enumerate(lora_parameters(model))}},
        path,
    )


def load_lora(model: nn.Module, path) -> None:
    """Load :func:`save_lora` output into an already-wrapped ``model``."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["lora"]
    params = lora_parameters(model)
    assert len(sd) == len(params), (
        f"checkpoint has {len(sd)} LoRA tensors but the model exposes "
        f"{len(params)}; rank or target mismatch."
    )
    for i, p in enumerate(params):
        p.data.copy_(sd[i].to(p.device, p.dtype))
