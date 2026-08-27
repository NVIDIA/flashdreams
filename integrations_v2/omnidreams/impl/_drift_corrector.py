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

CUDA-graph-safe ``fused`` mode (``DRIFT_CORRECTOR_MODE=fused``)
---------------------------------------------------------------

Neither default works under the accelerated serving stack
(``compile_network=True`` + ``use_cuda_graph=True``): the unfused path
adds live gated matmuls (new ops the captured graph never saw), and the
pre-merged path rebinds ``lin.weight.data`` to a cached tensor — captured
kernels reference the *original* storage address, so after capture the
gate silently stops changing what the graph computes. Serving therefore
had to disable acceleration to run the corrector (~6 fps vs 30 fps).

The fused mode keeps the same pre-merged weight sets and the same
CPU-side call-index gate, but swaps by ``copy_``-ing the cached set into
the original parameter storages (batched ``torch._foreach_copy_``) — the
:class:`~omnidreams._edit_lora.TextEditLoRA` mechanism, extended from a
per-window toggle to a per-denoise-step one. This is graph-safe by
construction: the transformer's ``CUDAGraphWrapper`` captures ONE network
forward, and each solver step plus the ``finalize_kv_cache`` context
forward is a separate replay of that graph (``timestep`` is a staged
input), so the exact per-step ``alpha*(t) * gain`` profile survives — no
gate collapse to a constant is needed, and the graph sees only fixed
parameter addresses whose values change between replays.

Tradeoff, stated honestly: each within-chunk alpha change is a
device-to-device copy of the four attention projections' weights (one
copy per distinct consecutive alpha; the default two-entry profile costs
two swaps per chunk, a profile with its own context-noise entry costs
three). That is HBM bandwidth the pointer-rebind mode does not spend —
sub-millisecond per swap on datacenter parts, and orders of magnitude
cheaper than the acceleration the corrector previously forfeited by
forcing the eager stack (real-time ~30 fps down to ~6 fps). An in-place ``addmm_`` of the rank-16 delta
difference would avoid the cached sets' VRAM, but repeated bf16
accumulation drifts over long live-game rollouts; copying from pristine
fp32-merged sets restores exact values on every swap. Fused mode does
not compose with a concurrently attached ``TextEditLoRA`` (both ``copy_``
into the same self-attention projections; the last writer wins) — same
restriction as the pre-merged mode; use ``unfused`` for stacked deploys,
or the per-state :class:`DriftCorrectorDispatch` below, whose composed
weight sets carry the style-LoRA delta and the corrector delta in one
``copy_`` source (resolving the last-writer-wins conflict for the
self-attention projections; the LoRA hook must then stop toggling them).

Mode selection: ``DRIFT_CORRECTOR_MODE`` = ``premerged`` (default) |
``fused`` | ``unfused``; the legacy ``DRIFT_CORRECTOR_UNFUSED=1`` still
forces ``unfused``.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omnidreams.impl._module_utils import unwrap_compiled_module
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
    return _load_gate_json(path)


def _load_gate_json(path: Path | str) -> dict[float, float]:
    """Load a ``{timestep: alpha}`` profile from a gate JSON file.

    Accepts either a flat mapping or an object with a ``"gate_alpha"``
    entry (the ``edit_sft/gate_style.py`` output format).
    """
    table = json.loads(Path(path).read_text())
    table = table.get("gate_alpha", table)
    profile = {float(t): float(a) for t, a in table.items()}
    assert profile and all(0.0 < a <= 1.0 for a in profile.values()), (
        f"gate profile {str(path)!r} must map timesteps to alphas in (0, 1]"
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


def _nearest_alpha(t: float, profile: dict[float, float] | None = None) -> float:
    """Return the profile entry (default :data:`GATE_ALPHA`) nearest in t."""
    profile = GATE_ALPHA if profile is None else profile
    return min(profile.items(), key=lambda kv: abs(kv[0] - t))[1]


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


_MODES = ("premerged", "fused", "unfused")


def _resolve_mode(mode: str | None, unfused: bool | None) -> str:
    """Resolve the deploy mode from explicit args, then the environment.

    Precedence: explicit ``mode`` > explicit ``unfused`` bool >
    ``DRIFT_CORRECTOR_MODE`` > legacy ``DRIFT_CORRECTOR_UNFUSED=1`` >
    ``premerged``.
    """
    if mode is None and unfused is not None:
        mode = "unfused" if unfused else "premerged"
    if mode is None:
        mode = os.environ.get("DRIFT_CORRECTOR_MODE", "")
    if not mode:
        legacy = os.environ.get("DRIFT_CORRECTOR_UNFUSED", "0") == "1"
        mode = "unfused" if legacy else "premerged"
    assert mode in _MODES, f"drift-corrector mode {mode!r} not in {_MODES}"
    return mode


def apply_drift_corrector(
    runner: Any,
    checkpoint: Path,
    gain: float,
    *,
    unfused: bool | None = None,
    mode: str | None = None,
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
            per-step pre-merged weights. ``None`` reads the environment
            (see :func:`_resolve_mode`).
        mode: Explicit deploy mode (``premerged`` | ``fused`` |
            ``unfused``); overrides ``unfused`` and the environment.
            ``fused`` is the CUDA-graph-safe in-place-``copy_`` variant —
            required whenever the serving stack runs with
            ``compile_network`` / ``use_cuda_graph`` enabled.

    Returns:
        A log-line string describing the deployed configuration.
    """
    mode = _resolve_mode(mode, unfused)
    unfused = mode == "unfused"
    network = unwrap_compiled_module(
        runner.pipeline.diffusion_model.transformer.network
    )
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

    # Pre-merged paths: one cached weight set per distinct alpha*(t)
    # value; the per-step gate installs the cached set — no LoRA matmuls
    # in the hot path. "premerged" re-points ``lin.weight.data`` (zero
    # copy, NOT CUDA-graph-safe); "fused" ``copy_``s into the original
    # parameter storages (graph-safe: captured kernels keep reading the
    # same addresses and only the values change between replays).
    linears = _target_linears(network)
    assert len(sd) == 2 * len(linears), (
        f"corrector checkpoint has {len(sd)} LoRA tensors but the network "
        f"exposes {2 * len(linears)}; rank or target mismatch."
    )
    if mode == "fused":
        assert getattr(transformer, "_optimized_dit_executor", None) is None, (
            "fused drift corrector merges into the PyTorch network's "
            "weights, which the native optimized-DiT executor bypasses; "
            "run with native_dit_acceleration='disabled'."
        )
    weight_sets, added_bytes = _premerge_weight_sets(linears, sd, gain)
    current: list[float | None] = [None]

    if mode == "fused":
        live = [lin.weight.data for lin in linears]

        def _swap(alpha: float) -> None:
            if alpha != current[0]:
                torch._foreach_copy_(live, weight_sets[alpha])
                current[0] = alpha
    else:

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
    kind = "graph-safe fused" if mode == "fused" else "pre-merged"
    return (
        f"corrected (alpha*(t) x {gain}, {kind} {len(weight_sets)} weight "
        f"sets, +{added_bytes / 2**20:.0f} MiB)"
    )


## Per-state dispatch (fused mode, multiple correctors)

_VRAM_WARN_GIB = 8.0
"""Warn when the dispatch's cached weight sets exceed this budget."""


@dataclass
class _CorrectorState:
    """Pre-merged weight sets and the per-step gate schedule for one state."""

    sets: dict[float, list[Tensor]]
    step_alphas: list[float]
    ctx_alpha: float
    added_bytes: int


class DriftCorrectorDispatch:
    """Multiple pre-merged corrector states behind one graph-safe selector.

    Extends the ``fused`` mode of :func:`apply_drift_corrector` from one
    corrector to a registry of named states, each pre-merged from a
    pristine base-weight snapshot taken at construction:

        ``sets[alpha] = pristine (+ lora_delta) + alpha * gain * (B @ A)``

    The optional ``lora_delta`` (e.g. a style ``TextEditLoRA``'s
    self-attention deltas) rides the same ``copy_`` source as the
    corrector delta, which resolves the fused mode's last-writer-wins
    conflict: this dispatch becomes the SOLE writer of the self-attention
    projection weights, so a concurrently attached edit LoRA must be
    restricted to projections outside :data:`_LORA_TARGETS` (e.g.
    cross-attention only) while the dispatch is installed.

    A ``"base"`` state (pristine weights, corrector off) is registered at
    construction and is the initial selection; re-register it to give the
    base world its own corrector (e.g. the shipped photoreal checkpoint).

    :meth:`set_active_corrector` is safe between rollouts and at chunk
    boundaries (between ``scheduler.sample`` calls) — the recommended call
    sites. A mid-rollout call is still graph-safe (weights only change
    between graph replays) but takes effect at the next denoise step, so
    one chunk mixes two states; keep swaps at chunk boundaries for clean
    visuals.
    """

    def __init__(self, runner: Any) -> None:
        diffusion_model = runner.pipeline.diffusion_model
        transformer = diffusion_model.transformer
        assert getattr(transformer, "_optimized_dit_executor", None) is None, (
            "the fused drift-corrector dispatch merges into the PyTorch "
            "network's weights, which the native optimized-DiT executor "
            "bypasses; run with native_dit_acceleration='disabled'."
        )
        network = unwrap_compiled_module(transformer.network)
        self._linears = _target_linears(network)
        self._live = [lin.weight.data for lin in self._linears]
        self._pristine32 = [
            lin.weight.detach().to(torch.float32, copy=True) for lin in self._linears
        ]
        self._step_ts = [
            float(t) for t in diffusion_model.scheduler.denoising_step_list
        ]
        self._ctx_t = float(diffusion_model.config.context_noise)
        self._states: dict[str, _CorrectorState] = {}
        self._active = "base"
        self._current: tuple[str, float] | None = None
        self.register_state("base")
        self._install_gate_driver(diffusion_model.scheduler, transformer)

    def register_state(
        self,
        name: str,
        *,
        checkpoint: Path | str | None = None,
        gain: float = 0.0,
        gate_alpha: dict[float, float] | Path | str | None = None,
        lora_delta: list[Tensor] | None = None,
    ) -> str:
        """Pre-merge and cache the weight sets for one named state.

        Args:
            name: State name for :meth:`set_active_corrector`.
            checkpoint: Corrector LoRA checkpoint (``train_v2`` format);
                ``None`` (or ``gain == 0``) makes this an off state.
            gain: Global gain composed with the alpha*(t) profile.
            gate_alpha: Per-state gate profile — a ``{timestep: alpha}``
                dict, a gate-JSON path, or ``None`` for :data:`GATE_ALPHA`.
            lora_delta: Optional fp32 weight deltas (one per target linear,
                checkpoint load order) folded into EVERY set of this state,
                including its within-state off set — the state-aware base.

        Returns:
            A log-line string describing the registered state.
        """
        if isinstance(gate_alpha, (str, Path)):
            gate_alpha = _load_gate_json(gate_alpha)
        profile = GATE_ALPHA if gate_alpha is None else gate_alpha
        base32 = [w.clone() for w in self._pristine32]
        if lora_delta is not None:
            assert len(lora_delta) == len(self._linears), (
                f"lora_delta has {len(lora_delta)} tensors for "
                f"{len(self._linears)} target projections"
            )
            for w, d in zip(base32, lora_delta):
                w.add_(d.to(w.device, w.dtype))

        added_bytes = 0
        if checkpoint is None or gain == 0.0:
            merged = [w.to(lin.weight.dtype) for w, lin in zip(base32, self._linears)]
            added_bytes = sum(w.numel() * w.element_size() for w in merged)
            state = _CorrectorState(
                sets={0.0: merged},
                step_alphas=[0.0] * len(self._step_ts),
                ctx_alpha=0.0,
                added_bytes=added_bytes,
            )
        else:
            sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["lora"]
            assert len(sd) == 2 * len(self._linears), (
                f"corrector checkpoint has {len(sd)} LoRA tensors but the "
                f"network exposes {2 * len(self._linears)}; rank or target "
                "mismatch."
            )
            sets: dict[float, list[Tensor]] = {}
            for alpha in sorted(set(profile.values())):
                merged = []
                for i, (lin, w32) in enumerate(zip(self._linears, base32)):
                    a = sd[2 * i].to(w32.device, torch.float32)
                    b = sd[2 * i + 1].to(w32.device, torch.float32)
                    w = w32.clone().addmm_(b, a, alpha=gain * alpha)
                    merged.append(w.to(lin.weight.dtype))
                    added_bytes += merged[-1].numel() * merged[-1].element_size()
                sets[alpha] = merged
            state = _CorrectorState(
                sets=sets,
                step_alphas=[_nearest_alpha(t, profile) for t in self._step_ts],
                ctx_alpha=_nearest_alpha(self._ctx_t, profile),
                added_bytes=added_bytes,
            )

        self._states[name] = state
        if name == self._active:
            self._current = None  # re-registration invalidates live weights
        total = sum(s.added_bytes for s in self._states.values())
        if total > _VRAM_WARN_GIB * 2**30:
            warnings.warn(
                f"drift-corrector dispatch caches {total / 2**30:.1f} GiB of "
                f"weight sets across {len(self._states)} states, over the "
                f"~{_VRAM_WARN_GIB:.0f} GiB budget; drop states or gate "
                "entries.",
                ResourceWarning,
                stacklevel=2,
            )
        return (
            f"corrector state {name!r}: {len(state.sets)} weight sets "
            f"(gain {gain}, +{state.added_bytes / 2**20:.0f} MiB, "
            f"total {total / 2**20:.0f} MiB)"
        )

    def set_active_corrector(self, name: str) -> None:
        """Select the state whose weight sets the gate driver installs.

        Takes effect at the next gated forward (next denoise step /
        context forward); call at chunk or rollout boundaries. Forces a
        copy on that forward even if the alpha value matches, so a
        re-selected state always restores exact pre-merged values.
        """
        assert name in self._states, (
            f"unknown corrector state {name!r}; registered: {sorted(self._states)}"
        )
        self._active = name
        self._current = None

    @property
    def active_state(self) -> str:
        """Name of the currently selected state."""
        return self._active

    def _swap(self, alpha: float) -> None:
        key = (self._active, alpha)
        if key != self._current:
            torch._foreach_copy_(self._live, self._states[self._active].sets[alpha])
            self._current = key

    def _install_gate_driver(self, scheduler: Any, transformer: Any) -> None:
        """Drive the gate CPU-side by call index (see the fused-mode notes)."""
        orig_sample = scheduler.sample

        def gated_sample(initial_noise, predict_flow, rng=None):
            calls = [0]

            def pf(noisy, timestep):
                # Resolve the state per call so a mid-rollout selector swap
                # stays consistent between step_alphas and the weight sets.
                step_alphas = self._states[self._active].step_alphas
                assert calls[0] < len(step_alphas), "predict_flow calls > solver steps"
                self._swap(step_alphas[calls[0]])
                calls[0] += 1
                return predict_flow(noisy, timestep)

            return orig_sample(initial_noise=initial_noise, predict_flow=pf, rng=rng)

        scheduler.sample = gated_sample
        orig_finalize = transformer.finalize_kv_cache

        def gated_finalize(*args, **kwargs):
            self._swap(self._states[self._active].ctx_alpha)
            return orig_finalize(*args, **kwargs)

        transformer.finalize_kv_cache = gated_finalize
