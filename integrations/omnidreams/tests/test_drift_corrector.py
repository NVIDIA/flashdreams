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

"""CPU-only unit tests for the Clean Forcing drift-corrector deploy hook.

Covers the behaviours a deployment depends on:

* ``_LoRALinear`` is a strict identity at ``scale == 0`` and at the
  zero-initialized ``B`` (so wrapping the network never changes base
  outputs until a trained checkpoint is loaded and gated on).
* ``_apply_lora`` wraps exactly the self-attention projections the
  training-side module wraps (same match rule -> same checkpoint order),
  and ``_set_scale`` reaches every wrapped linear.
* The ``alpha*(t)`` gate profile resolves by nearest-t lookup, including
  the context-noise forward.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from omnidreams._drift_corrector import (
    _LORA_RANK,
    GATE_ALPHA,
    DriftCorrectorDispatch,
    _apply_lora,
    _gate_alpha,
    _LoRALinear,
    _nearest_alpha,
    _premerge_weight_sets,
    _resolve_mode,
    _set_scale,
    _target_linears,
    apply_drift_corrector,
)

pytestmark = pytest.mark.ci_cpu


def test_lora_linear_is_identity_at_zero_scale():
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=False)
    lora = _LoRALinear(base, rank=2)
    nn.init.normal_(lora.A.weight)
    nn.init.normal_(lora.B.weight)  # non-zero delta path
    x = torch.randn(3, 8)
    lora.scale = 0.0
    assert torch.equal(lora(x), base(x))


def test_lora_linear_is_identity_at_zero_init_b():
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=False)
    lora = _LoRALinear(base, rank=2)  # B is zero-initialized
    lora.scale = 1.0
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x))


def test_lora_linear_applies_scaled_delta():
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=False)
    lora = _LoRALinear(base, rank=2)
    nn.init.normal_(lora.A.weight)
    nn.init.normal_(lora.B.weight)
    x = torch.randn(3, 8)
    lora.scale = 0.5
    delta = lora(x) - base(x)
    lora.scale = 1.0
    assert torch.allclose(2.0 * delta, lora(x) - base(x), atol=1e-5)


def test_apply_lora_wraps_only_attention_targets_and_set_scale_reaches_all():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict(
                {
                    n: nn.Linear(4, 4, bias=False)
                    for n in ("q_proj", "k_proj", "v_proj", "output_proj")
                }
            )
            self.mlp = nn.Linear(4, 4, bias=False)

    toy = Toy()
    params = _apply_lora(toy)
    wrapped = [m for m in toy.modules() if isinstance(m, _LoRALinear)]
    assert len(wrapped) == 4  # q/k/v/output projections but not the mlp
    assert len(params) == 8  # A + B per wrapped linear
    assert not isinstance(toy.mlp, _LoRALinear)
    _set_scale(toy, 0.25)
    assert all(m.scale == 0.25 for m in wrapped)


def test_gate_profile_nearest_t_lookup():
    assert _nearest_alpha(1000.0) == GATE_ALPHA[1000.0]
    assert _nearest_alpha(803.0) == GATE_ALPHA[803.0]
    # The context-noise forward (t=128) resolves to the low-t entry,
    # matching the evaluated deploy configs.
    assert _nearest_alpha(128.0) == GATE_ALPHA[803.0]
    # The deployed 2-step solver schedule (warped [1000, 350] -> ~[1000,
    # 803]) resolves to the two gate entries in order.
    assert [_nearest_alpha(t) for t in (1000.0, 802.9)] == [
        GATE_ALPHA[1000.0],
        GATE_ALPHA[803.0],
    ]


def test_gate_profile_is_a_strict_attenuation():
    assert all(0.0 < a <= 1.0 for a in GATE_ALPHA.values())


## GATE_ALPHA_JSON override


def test_gate_alpha_defaults_to_photoreal_profile(monkeypatch):
    monkeypatch.delenv("GATE_ALPHA_JSON", raising=False)
    assert _gate_alpha() == {1000.0: 0.96, 803.0: 0.667}


def test_gate_alpha_json_accepts_a_flat_mapping(tmp_path, monkeypatch):
    path = tmp_path / "gate.json"
    path.write_text('{"1000": 0.9, "803": 0.5, "128": 0.3}')
    monkeypatch.setenv("GATE_ALPHA_JSON", str(path))
    assert _gate_alpha() == {1000.0: 0.9, 803.0: 0.5, 128.0: 0.3}


def test_gate_alpha_json_accepts_the_gate_style_output_format(tmp_path, monkeypatch):
    path = tmp_path / "gate_style.json"
    path.write_text(
        '{"per_timestep": {"1000": {"alpha_star_unbiased": 0.9}},'
        ' "gate_alpha": {"1000": 0.9, "803": 0.5}}'
    )
    monkeypatch.setenv("GATE_ALPHA_JSON", str(path))
    assert _gate_alpha() == {1000.0: 0.9, 803.0: 0.5}


def test_gate_alpha_json_rejects_out_of_range_alphas(tmp_path, monkeypatch):
    path = tmp_path / "gate.json"
    path.write_text('{"1000": 1.5}')
    monkeypatch.setenv("GATE_ALPHA_JSON", str(path))
    with pytest.raises(AssertionError):
        _gate_alpha()


def test_gate_alpha_json_missing_file_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_ALPHA_JSON", str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError):
        _gate_alpha()


## Pre-merged weight path


def _toy_network() -> nn.Module:
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict(
                {
                    n: nn.Linear(4, 4, bias=False)
                    for n in ("q_proj", "k_proj", "v_proj", "output_proj")
                }
            )
            self.mlp = nn.Linear(4, 4, bias=False)

    return nn.Sequential(Block(), Block())


def _random_checkpoint(linears) -> dict[int, torch.Tensor]:
    sd = {}
    for i, lin in enumerate(linears):
        sd[2 * i] = 0.1 * torch.randn(_LORA_RANK, lin.in_features)
        sd[2 * i + 1] = 0.1 * torch.randn(lin.out_features, _LORA_RANK)
    return sd


def test_target_linears_match_apply_lora_load_order():
    torch.manual_seed(0)
    net = _toy_network()
    linears = _target_linears(net)
    wrapped_net = copy.deepcopy(net)
    params = _apply_lora(wrapped_net)
    wrapped = [m for m in wrapped_net.modules() if isinstance(m, _LoRALinear)]
    assert len(params) == 2 * len(linears) == 2 * len(wrapped)
    for lin, w in zip(linears, wrapped):
        assert torch.equal(lin.weight, w.base.weight)  # same walk order


def test_premerged_weights_match_the_unfused_delta():
    torch.manual_seed(0)
    net = _toy_network()
    linears = _target_linears(net)
    sd = _random_checkpoint(linears)
    gain = 0.25
    sets, added_bytes = _premerge_weight_sets(linears, sd, gain)
    assert set(sets) == set(GATE_ALPHA.values())
    x = torch.randn(3, 4)
    for alpha, merged in sets.items():
        for i, lin in enumerate(linears):
            unfused = lin(x) + gain * alpha * (x @ sd[2 * i].T @ sd[2 * i + 1].T)
            premerged = torch.nn.functional.linear(x, merged[i])
            assert torch.allclose(premerged, unfused, atol=1e-5)
    n_weights = sum(lin.weight.numel() for lin in linears)
    assert added_bytes == len(sets) * n_weights * 4  # fp32 toy weights


def test_premerge_does_not_mutate_base_weights():
    torch.manual_seed(0)
    net = _toy_network()
    linears = _target_linears(net)
    before = [lin.weight.detach().clone() for lin in linears]
    _premerge_weight_sets(linears, _random_checkpoint(linears), gain=1.0)
    for lin, w in zip(linears, before):
        assert torch.equal(lin.weight, w)


## Mode resolution


def test_resolve_mode_defaults_to_premerged(monkeypatch):
    monkeypatch.delenv("DRIFT_CORRECTOR_MODE", raising=False)
    monkeypatch.delenv("DRIFT_CORRECTOR_UNFUSED", raising=False)
    assert _resolve_mode(None, None) == "premerged"


def test_resolve_mode_env_and_legacy_env(monkeypatch):
    monkeypatch.setenv("DRIFT_CORRECTOR_MODE", "fused")
    assert _resolve_mode(None, None) == "fused"
    monkeypatch.delenv("DRIFT_CORRECTOR_MODE", raising=False)
    monkeypatch.setenv("DRIFT_CORRECTOR_UNFUSED", "1")
    assert _resolve_mode(None, None) == "unfused"


def test_resolve_mode_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("DRIFT_CORRECTOR_MODE", "fused")
    assert _resolve_mode("unfused", None) == "unfused"
    assert _resolve_mode(None, unfused=True) == "unfused"
    assert _resolve_mode(None, unfused=False) == "premerged"
    assert _resolve_mode("fused", unfused=True) == "fused"  # mode wins


def test_resolve_mode_rejects_unknown_modes():
    with pytest.raises(AssertionError):
        _resolve_mode("turbo", None)


## End-to-end deploy-hook harness (toy runner)


class _ToyTransformer(nn.Module):
    """Minimal stand-in for CosmosTransformer's deploy surface."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.network = _toy_network()

    def predict_flow(self, noisy, timestep):
        x = noisy
        for block in self.network:
            attn = block.self_attn
            x = attn["output_proj"](
                attn["q_proj"](x) + attn["k_proj"](x) + attn["v_proj"](x)
            ) + block.mlp(x)
        return x * (1.0 + timestep / 1000.0)

    def finalize_kv_cache(self, noisy, timestep):
        # Mirrors the base transformer: the context forward is one
        # predict_flow call whose flow is discarded; return it here so
        # tests can compare the corrected context forward across modes.
        return self.predict_flow(noisy, timestep)


class _ToyScheduler:
    denoising_step_list = torch.tensor([1000.0, 803.0])

    def sample(self, initial_noise, predict_flow, rng=None):
        flows = []
        for t in self.denoising_step_list:
            flows.append(predict_flow(initial_noise, t))
        return torch.stack(flows)


def _toy_runner(tmp_path):
    """Fake runner + checkpoint exposing what apply_drift_corrector uses."""
    from types import SimpleNamespace

    transformer = _ToyTransformer()
    sd = _random_checkpoint(_target_linears(transformer.network))
    ckpt = tmp_path / "corrector.pt"
    torch.save({"lora": sd}, ckpt)
    diffusion_model = SimpleNamespace(
        transformer=transformer,
        scheduler=_ToyScheduler(),
        config=SimpleNamespace(context_noise=128),
    )
    runner = SimpleNamespace(pipeline=SimpleNamespace(diffusion_model=diffusion_model))
    return runner, ckpt


def test_fused_matches_unfused_per_step_and_context(tmp_path):
    """The graph-safe fused mode reproduces the unfused gated outputs."""
    runner_u, ckpt = _toy_runner(tmp_path)
    runner_f, _ = _toy_runner(tmp_path)  # same seed -> identical weights

    apply_drift_corrector(runner_u, ckpt, gain=0.25, mode="unfused")
    tr_u = runner_u.pipeline.diffusion_model.transformer
    apply_drift_corrector(runner_f, ckpt, gain=0.25, mode="fused")

    torch.manual_seed(1)
    noise = torch.randn(3, 4)
    sched_f = runner_f.pipeline.diffusion_model.scheduler
    flows_f = sched_f.sample(
        initial_noise=noise,
        predict_flow=runner_f.pipeline.diffusion_model.transformer.predict_flow,
    )
    flows_u = torch.stack(
        [tr_u.predict_flow(noise, t) for t in _ToyScheduler.denoising_step_list]
    )
    assert torch.allclose(flows_f, flows_u, atol=1e-5)

    ctx_t = torch.tensor(128.0)
    ctx_f = runner_f.pipeline.diffusion_model.transformer.finalize_kv_cache(
        noise, ctx_t
    )
    ctx_u = tr_u.finalize_kv_cache(noise, ctx_t)
    assert torch.allclose(ctx_f, ctx_u, atol=1e-5)


def test_fused_swaps_change_outputs_across_steps(tmp_path):
    """The two solver steps see different alphas (gate actually swaps)."""
    runner, ckpt = _toy_runner(tmp_path)
    dm = runner.pipeline.diffusion_model
    apply_drift_corrector(runner, ckpt, gain=1.0, mode="fused")
    torch.manual_seed(1)
    noise = torch.randn(3, 4)
    flows = dm.scheduler.sample(
        initial_noise=noise, predict_flow=dm.transformer.predict_flow
    )
    # Undo the deterministic timestep modulation, leaving only the
    # weight-set difference between the two steps.
    step0 = flows[0] / (1.0 + 1000.0 / 1000.0)
    step1 = flows[1] / (1.0 + 803.0 / 1000.0)
    assert not torch.allclose(step0, step1, atol=1e-6)


def test_fused_preserves_weight_storage_premerged_does_not(tmp_path):
    """Graph-safety property: fused copies in place, premerged rebinds.

    Captured CUDA graphs reference parameter storage addresses; only the
    fused mode keeps them stable across gate swaps.
    """

    def drive_and_collect_ptrs(mode):
        runner, ckpt = _toy_runner(tmp_path)
        dm = runner.pipeline.diffusion_model
        linears = _target_linears(dm.transformer.network)
        before = [lin.weight.data_ptr() for lin in linears]
        apply_drift_corrector(runner, ckpt, gain=0.25, mode=mode)
        torch.manual_seed(1)
        noise = torch.randn(3, 4)
        dm.scheduler.sample(
            initial_noise=noise, predict_flow=dm.transformer.predict_flow
        )
        dm.transformer.finalize_kv_cache(noise, torch.tensor(128.0))
        after = [lin.weight.data_ptr() for lin in linears]
        return before, after

    before, after = drive_and_collect_ptrs("fused")
    assert before == after
    before, after = drive_and_collect_ptrs("premerged")
    assert before != after


def test_fused_refuses_native_dit_executor(tmp_path):
    runner, ckpt = _toy_runner(tmp_path)
    runner.pipeline.diffusion_model.transformer._optimized_dit_executor = object()
    with pytest.raises(AssertionError, match="optimized-DiT"):
        apply_drift_corrector(runner, ckpt, gain=0.25, mode="fused")


## Per-state dispatch


def _sample_flows(runner, noise):
    dm = runner.pipeline.diffusion_model
    return dm.scheduler.sample(
        initial_noise=noise, predict_flow=dm.transformer.predict_flow
    )


def _lora_delta(linears, scale=0.05):
    gen = torch.Generator().manual_seed(7)
    return [scale * torch.randn(lin.weight.shape, generator=gen) for lin in linears]


def test_dispatch_registers_base_and_rejects_unknown_states(tmp_path):
    runner, ckpt = _toy_runner(tmp_path)
    dispatch = DriftCorrectorDispatch(runner)
    assert dispatch.active_state == "base"
    summary = dispatch.register_state("skin", checkpoint=ckpt, gain=0.25)
    assert "skin" in summary and "weight sets" in summary
    dispatch.set_active_corrector("skin")
    assert dispatch.active_state == "skin"
    with pytest.raises(AssertionError, match="unknown corrector state"):
        dispatch.set_active_corrector("nope")


def test_dispatch_base_state_reproduces_pristine_outputs(tmp_path):
    runner, ckpt = _toy_runner(tmp_path)
    pristine, _ = _toy_runner(tmp_path)  # same seed -> identical weights
    dispatch = DriftCorrectorDispatch(runner)
    dispatch.register_state("skin", checkpoint=ckpt, gain=1.0)
    torch.manual_seed(1)
    noise = torch.randn(3, 4)
    assert torch.allclose(_sample_flows(runner, noise), _sample_flows(pristine, noise))


def test_dispatch_composed_state_matches_sequential_lora_then_corrector(tmp_path):
    """base + lora_delta + alpha*gain*corrector == applying them in sequence."""
    runner, ckpt = _toy_runner(tmp_path)
    reference, _ = _toy_runner(tmp_path)  # same seed -> identical weights
    gain = 0.25

    ref_linears = _target_linears(
        reference.pipeline.diffusion_model.transformer.network
    )
    delta = _lora_delta(ref_linears)
    with torch.no_grad():
        for lin, d in zip(ref_linears, delta):  # sequential: merge LoRA first
            lin.weight.add_(d)
    apply_drift_corrector(reference, ckpt, gain=gain, mode="unfused")
    tr_ref = reference.pipeline.diffusion_model.transformer

    dispatch = DriftCorrectorDispatch(runner)
    dispatch.register_state("skin", checkpoint=ckpt, gain=gain, lora_delta=delta)
    dispatch.set_active_corrector("skin")

    torch.manual_seed(1)
    noise = torch.randn(3, 4)
    flows = _sample_flows(runner, noise)
    flows_ref = torch.stack(
        [tr_ref.predict_flow(noise, t) for t in _ToyScheduler.denoising_step_list]
    )
    assert torch.allclose(flows, flows_ref, atol=1e-5)
    ctx_t = torch.tensor(128.0)
    ctx = runner.pipeline.diffusion_model.transformer.finalize_kv_cache(noise, ctx_t)
    assert torch.allclose(ctx, tr_ref.finalize_kv_cache(noise, ctx_t), atol=1e-5)


def test_dispatch_selector_switching_restores_exact_state_weights(tmp_path):
    runner, ckpt = _toy_runner(tmp_path)
    pristine, _ = _toy_runner(tmp_path)
    dispatch = DriftCorrectorDispatch(runner)
    dispatch.register_state("skin", checkpoint=ckpt, gain=1.0)
    torch.manual_seed(1)
    noise = torch.randn(3, 4)
    base_flows = _sample_flows(pristine, noise)

    dispatch.set_active_corrector("skin")
    skin_flows = _sample_flows(runner, noise)
    assert not torch.allclose(skin_flows, base_flows, atol=1e-6)
    dispatch.set_active_corrector("base")
    assert torch.allclose(_sample_flows(runner, noise), base_flows)
    dispatch.set_active_corrector("skin")
    assert torch.allclose(_sample_flows(runner, noise), skin_flows)


def test_dispatch_keeps_weight_storage_addresses_across_states(tmp_path):
    runner, ckpt = _toy_runner(tmp_path)
    linears = _target_linears(runner.pipeline.diffusion_model.transformer.network)
    before = [lin.weight.data_ptr() for lin in linears]
    dispatch = DriftCorrectorDispatch(runner)
    dispatch.register_state("skin", checkpoint=ckpt, gain=0.5)
    torch.manual_seed(1)
    noise = torch.randn(3, 4)
    for state in ("skin", "base", "skin"):
        dispatch.set_active_corrector(state)
        _sample_flows(runner, noise)
    assert [lin.weight.data_ptr() for lin in linears] == before


def test_dispatch_gate_profiles_are_per_state(tmp_path):
    runner, ckpt = _toy_runner(tmp_path)
    dispatch = DriftCorrectorDispatch(runner)
    dispatch.register_state(
        "skin", checkpoint=ckpt, gain=1.0, gate_alpha={1000.0: 0.9, 803.0: 0.4}
    )
    dispatch.register_state(
        "weather", checkpoint=ckpt, gain=1.0, gate_alpha={1000.0: 0.3}
    )
    assert dispatch._states["skin"].step_alphas == [0.9, 0.4]
    assert dispatch._states["skin"].ctx_alpha == 0.4  # t=128 -> nearest 803
    assert dispatch._states["weather"].step_alphas == [0.3, 0.3]
    assert dispatch._states["weather"].ctx_alpha == 0.3
    assert dispatch._states["base"].step_alphas == [0.0, 0.0]


def test_dispatch_gate_profile_accepts_a_json_path(tmp_path):
    runner, ckpt = _toy_runner(tmp_path)
    gate = tmp_path / "gate.json"
    gate.write_text('{"1000": 0.8, "803": 0.2}')
    dispatch = DriftCorrectorDispatch(runner)
    dispatch.register_state("skin", checkpoint=ckpt, gain=1.0, gate_alpha=gate)
    assert dispatch._states["skin"].step_alphas == [0.8, 0.2]


def test_dispatch_warns_over_the_vram_budget(tmp_path, monkeypatch):
    import omnidreams._drift_corrector as dc

    runner, ckpt = _toy_runner(tmp_path)
    dispatch = DriftCorrectorDispatch(runner)
    monkeypatch.setattr(dc, "_VRAM_WARN_GIB", 1e-9)
    with pytest.warns(ResourceWarning, match="over the"):
        dispatch.register_state("skin", checkpoint=ckpt, gain=0.25)


def test_dispatch_refuses_native_dit_executor(tmp_path):
    runner, _ = _toy_runner(tmp_path)
    runner.pipeline.diffusion_model.transformer._optimized_dit_executor = object()
    with pytest.raises(AssertionError, match="optimized-DiT"):
        DriftCorrectorDispatch(runner)
