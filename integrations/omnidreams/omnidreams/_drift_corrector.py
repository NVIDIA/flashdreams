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

"""Clean Forcing drift corrector for the Omnidreams runner.

Deploys the trained corrector LoRA (``drift_correction/train_v2.py``
checkpoints) on a built :class:`~omnidreams.runner.OmnidreamsRunner`'s
pipeline at ``alpha*(t) * gain`` per denoise step. SHIPPED config (owner
decision 2026-07-24): ``lora_v2_v3_valpeak.pt`` at gain 0.25
(``corrgate025`` — best trees/foliage detail and consistency, drift
Delta +0.99 vs base +2.44). Mirrors the HY-WorldPlay deploy module
(``hy_worldplay/_drift_corrector.py``); self-contained so the production
runner does not import the research directory.

By default the LoRA is **pre-merged**: at load time each discrete
``alpha*(t) * gain`` value gets its own cached copy of the target
projection weights with the scaled delta folded in, and the per-step gate
just swaps the cached set in — zero extra work in the hot path. The gate
is driven CPU-side from the load-time solver schedule (one
``predict_flow`` call per solver step), so the corrected forward issues
the same kernels as base with no GPU timestep readback. Set
``DRIFT_CORRECTOR_UNFUSED=1`` to fall back to the runtime A/B-matmul path
(the pre-2026-07-25 behavior).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

## Deploy policy


def _gate_alpha() -> dict[float, float]:
    """Resolve the gate profile: ``GATE_ALPHA_JSON`` override or the default.

    The override file holds either a flat ``{timestep: alpha}`` mapping or
    an object with a ``"gate_alpha"`` entry (the ``edit_sft/gate_style.py``
    output format). Read once at import time, so set the variable before
    importing this module.
    """
    path = os.environ.get("GATE_ALPHA_JSON", "")
    if not path:
        return {1000.0: 0.96, 803.0: 0.667}
    table = json.loads(Path(path).read_text())
    table = table.get("gate_alpha", table)
    profile = {float(t): float(a) for t, a in table.items()}
    assert profile and all(0.0 < a <= 1.0 for a in profile.values()), (
        f"GATE_ALPHA_JSON={path!r} must map timesteps to alphas in (0, 1]"
    )
    return profile


GATE_ALPHA = _gate_alpha()
"""Unbiased alpha*(t) from the step-0 systematicity gate. Default: the
photoreal drift-pair profile (drift_correction's
``outputs/gate/gate_faithful_v2.json``) — the systematic fraction of the
drift-induced error at each of the two distilled solver timesteps. The
corrector LoRA is rescaled to ``alpha*(t) * gain`` before every denoise
step (nearest-t lookup); the ``finalize_kv_cache`` context forward (t=128)
resolves to the nearest entry (t=803 in the default profile), matching the
evaluated deploy configs. ``GATE_ALPHA_JSON`` swaps in a measured profile
(e.g. ``edit_sft/outputs/gate_style.json`` for styled worlds), which may
add its own low-t entry for the context forward."""

_LORA_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
)
"""Self-attention projections the corrector checkpoints were trained on."""

_LORA_RANK = 16
"""Rank of the shipped corrector checkpoints."""


class _LoRALinear(nn.Module):
    """Frozen base linear plus a runtime-gated low-rank delta.

    Mirrors the training-side module in
    ``integrations/omnidreams/drift_correction/_lora.py``: ``scale`` is the
    runtime gain (``0`` = exact base output), and the A/B path runs in fp32
    regardless of the base dtype.
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
            # Substring match, exactly as the training-side apply_lora, so
            # the wrap set and load order match the checkpoint indices.
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


def _nearest_alpha(t: float) -> float:
    """Return the :data:`GATE_ALPHA` entry with the nearest timestep."""
    return min(GATE_ALPHA.items(), key=lambda kv: abs(kv[0] - t))[1]


def _target_linears(network: nn.Module) -> list[nn.Linear]:
    """Target linears in checkpoint load order (same walk as ``_apply_lora``)."""
    linears: list[nn.Linear] = []
    for mname, module in network.named_modules():
        for cname, child in module.named_children():
            full = f"{mname}.{cname}" if mname else cname
            if isinstance(child, nn.Linear) and any(t in full for t in _LORA_TARGETS):
                linears.append(child)
    return linears


def _premerge_weight_sets(
    linears: list[nn.Linear], sd: dict, gain: float
) -> tuple[dict[float, list[Tensor]], int]:
    """Cache ``W + gain*alpha*(B @ A)`` per distinct gate value.

    ``sd`` holds the checkpoint tensors in load order (``A_i`` at ``2i``,
    ``B_i`` at ``2i + 1``). The merge runs in fp32 (matching the unfused
    path's fp32 delta) and is cast back to the base weight dtype.

    Returns:
        The per-alpha weight sets and the total cached bytes.
    """
    sets: dict[float, list[Tensor]] = {}
    added_bytes = 0
    for alpha in sorted(set(GATE_ALPHA.values())):
        merged: list[Tensor] = []
        for i, lin in enumerate(linears):
            a = sd[2 * i].to(lin.weight.device, torch.float32)
            b = sd[2 * i + 1].to(lin.weight.device, torch.float32)
            w32 = lin.weight.detach().to(torch.float32, copy=True)
            w = w32.addmm_(b, a, alpha=gain * alpha).to(lin.weight.dtype)
            merged.append(w)
            added_bytes += w.numel() * w.element_size()
        sets[alpha] = merged
    return sets, added_bytes


def apply_drift_corrector(
    runner: Any, checkpoint: Path, gain: float, *, unfused: bool | None = None
) -> str:
    """Deploy the corrector LoRA on ``runner`` with the alpha*(t) gate.

    Args:
        runner: A built ``OmnidreamsRunner``.
        checkpoint: Corrector LoRA checkpoint (``train_v1``/``train_v2``
            format: a dict whose ``"lora"`` entry maps load-order indices
            to tensors).
        gain: Global gain composed with the alpha*(t) profile; the
            shipped configuration (``corrgate025``) is 0.25.
        unfused: Force the runtime A/B-matmul path instead of the default
            per-step pre-merged weights. ``None`` reads the
            ``DRIFT_CORRECTOR_UNFUSED`` environment variable.

    Returns:
        A log-line string describing the deployed configuration.
    """
    if unfused is None:
        unfused = os.environ.get("DRIFT_CORRECTOR_UNFUSED", "0") == "1"
    network = runner.pipeline.diffusion_model.transformer.network
    if hasattr(network, "_orig_mod"):  # unwrap torch.compile
        network = network._orig_mod
    transformer = runner.pipeline.diffusion_model.transformer
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["lora"]

    if unfused:
        params = _apply_lora(network)
        assert len(sd) == len(params), (
            f"corrector checkpoint has {len(sd)} LoRA tensors but the network "
            f"exposes {len(params)}; rank or target mismatch."
        )
        for i, p in enumerate(params):
            p.data.copy_(sd[i].to(p.device, p.dtype))
        orig_pf = transformer.predict_flow

        # Per-step gate: rescale the LoRA to alpha*(t) x gain before every
        # denoise step (nearest-t lookup; finalize_kv_cache calls positionally).
        def gated_pf(*args, **kwargs):
            ts = kwargs.get("timestep", args[1] if len(args) > 1 else None)
            t = float(ts.reshape(-1).max())
            _set_scale(network, _nearest_alpha(t) * gain)
            return orig_pf(*args, **kwargs)

        transformer.predict_flow = gated_pf
        return f"corrected (alpha*(t) x {gain}, unfused)"

    # Pre-merged path (default): one cached weight set per distinct
    # alpha*(t) value; the per-step gate just re-points the projection
    # weights at the cached set — no LoRA matmuls in the hot path.
    linears = _target_linears(network)
    assert len(sd) == 2 * len(linears), (
        f"corrector checkpoint has {len(sd)} LoRA tensors but the network "
        f"exposes {2 * len(linears)}; rank or target mismatch."
    )
    weight_sets, added_bytes = _premerge_weight_sets(linears, sd, gain)
    current: list[float | None] = [None]

    def _swap(alpha: float) -> None:
        if alpha != current[0]:
            for lin, w in zip(linears, weight_sets[alpha]):
                lin.weight.data = w
            current[0] = alpha

    # Drive the gate CPU-side. The scheduler makes exactly one
    # ``predict_flow`` call per solver step in a Python loop, so each
    # step's alpha resolves from the load-time schedule by call index —
    # reading the timestep tensor back per step (the unfused path's
    # ``float(timestep.max())``) would stall the CPU launch queue every
    # solver step.
    scheduler = runner.pipeline.diffusion_model.scheduler
    step_alphas = [_nearest_alpha(t) for t in scheduler.denoising_step_list.tolist()]
    ctx_alpha = _nearest_alpha(
        float(runner.pipeline.diffusion_model.config.context_noise)
    )
    orig_sample = scheduler.sample

    def gated_sample(initial_noise, predict_flow, rng=None):
        calls = [0]

        def pf(noisy, timestep):
            assert calls[0] < len(step_alphas), "predict_flow calls > solver steps"
            _swap(step_alphas[calls[0]])
            calls[0] += 1
            return predict_flow(noisy, timestep)

        return orig_sample(initial_noise=initial_noise, predict_flow=pf, rng=rng)

    scheduler.sample = gated_sample
    orig_finalize = transformer.finalize_kv_cache

    def gated_finalize(*args, **kwargs):
        _swap(ctx_alpha)
        return orig_finalize(*args, **kwargs)

    transformer.finalize_kv_cache = gated_finalize
    return (
        f"corrected (alpha*(t) x {gain}, pre-merged {len(weight_sets)} weight "
        f"sets, +{added_bytes / 2**20:.0f} MiB)"
    )
