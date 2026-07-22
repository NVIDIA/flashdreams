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

"""Content-keyed Clean Forcing drift corrector for the HY-WorldPlay runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

## Deploy policy

GATE_ALPHA = {1000.0: 0.81, 960.0: 0.53, 888.8889: 0.53, 727.2728: 0.58}
"""Unbiased alpha*(t) from the step-0 systematicity gate (drift_correction's
``outputs/gate/gate_faithful.json``): the systematic fraction of the
drift-induced error at each of the distilled solver's timesteps. The
shipped config deploys the LoRA at ``alpha*(t) * gain`` per denoise step."""

STATIC_ACTION_CLASS = 0
"""Action label for a no-translation / no-rotation step; a trajectory of
only this class is a locked-off camera."""

_LORA_TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o")
"""Self-attention projections the corrector checkpoint was trained on."""

_LORA_RANK = 16
"""Rank of the shipped v2 corrector checkpoint."""


class _LoRALinear(nn.Module):
    """Frozen base linear plus a runtime-gated low-rank delta.

    Mirrors the training-side module in
    ``integrations/hy_worldplay/drift_correction/_lora.py``: ``scale`` is
    the runtime gain (``0`` = exact base output), and the A/B path runs in
    fp32 regardless of the base dtype.
    """

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.B.weight)
        self.scale = 0.0

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.scale != 0:
            delta = self.B(self.A(x.to(self.A.weight.dtype)))
            out = out + self.scale * delta.to(out.dtype)
        return out


def _apply_lora(network: nn.Module) -> list[nn.Parameter]:
    """Wrap the target linears and return the LoRA parameters in load order."""
    for mname, module in list(network.named_modules()):
        for cname, child in list(module.named_children()):
            full = f"{mname}.{cname}" if mname else cname
            if isinstance(child, nn.Linear) and any(t in full for t in _LORA_TARGETS):
                setattr(
                    module,
                    cname,
                    _LoRALinear(child, _LORA_RANK).to(child.weight.device),
                )
    params: list[nn.Parameter] = []
    for m in network.modules():
        if isinstance(m, _LoRALinear):
            params += list(m.A.parameters()) + list(m.B.parameters())
    return params


def _set_scale(network: nn.Module, scale: float) -> None:
    """Set the runtime gain on every wrapped linear."""
    for m in network.modules():
        if isinstance(m, _LoRALinear):
            m.scale = scale


def is_static_trajectory(pose: str | Path, n_latents: int) -> bool:
    """Whether the job's trajectory is a locked-off camera.

    Args:
        pose: Pose-string or trajectory-JSON path (upstream grammar).
        n_latents: Rollout latent budget (``num_chunk * 4``).

    Returns:
        ``True`` when every per-latent action label is
        :data:`STATIC_ACTION_CLASS`.
    """
    from hy_worldplay._pose import parse_pose_action_labels

    labels = parse_pose_action_labels(pose, n_latents)
    return bool((labels == STATIC_ACTION_CLASS).all())


def maybe_apply_drift_corrector(runner: Any, checkpoint: Path, gain: float) -> str:
    """Deploy the corrector on ``runner`` keyed on the job's trajectory content.

    Ship rule (owner decision 2026-07-21): static scenes measure negative
    drift on this host, so correction there is pure artifact cost — static
    jobs run the untouched base weights. Commanded-motion jobs get the
    corrector LoRA at ``alpha*(t) * gain`` per denoise step (the
    ``corrgate050`` config at the default ``gain=0.5``).

    Args:
        runner: A built ``HyWorldPlayWanI2VRunner``.
        checkpoint: Corrector LoRA checkpoint (``save_lora`` format).
        gain: Global gain multiplied into the alpha*(t) profile.

    Returns:
        ``"base (static trajectory)"`` or ``"corrected (alpha*(t) x gain)"``,
        for the runner's log line.
    """
    cfg = runner.config
    if is_static_trajectory(cfg.pose, cfg.num_chunk * 4):
        return "base (static trajectory)"

    network = runner.pipeline.diffusion_model.transformer.network
    if hasattr(network, "_orig_mod"):  # unwrap torch.compile
        network = network._orig_mod
    params = _apply_lora(network)

    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["lora"]
    assert len(sd) == len(params), (
        f"corrector checkpoint has {len(sd)} LoRA tensors but the network "
        f"exposes {len(params)}; rank or target mismatch."
    )
    for i, p in enumerate(params):
        p.data.copy_(sd[i].to(p.device, p.dtype))

    # Per-step gate: rescale the LoRA to alpha*(t) x gain before every
    # denoise step (nearest-t lookup; per-token AR0 timesteps include the
    # first-frame stabilization value, the max is always the scheduler step).
    transformer = runner.pipeline.diffusion_model.transformer
    orig_pf = transformer.predict_flow

    def gated_pf(*args, **kwargs):
        t = float(kwargs["timestep"].reshape(-1).max())
        alpha = min(GATE_ALPHA.items(), key=lambda kv: abs(kv[0] - t))[1]
        _set_scale(network, alpha * gain)
        return orig_pf(*args, **kwargs)

    transformer.predict_flow = gated_pf
    return "corrected (alpha*(t) x gain)"
