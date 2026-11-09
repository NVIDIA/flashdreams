# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Positions, noise seeding and the bounded-cache rollout on a tiny network.

The cache test checks the stream cache against an independent statement of
cosmos-framework's history selection: keep every control and RGB entry, then
let a control read ``2 * sinks`` pinned and ``2 * (window - sinks - 1)`` recent
entries, and a target one more (its own control).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from cosmos3_nano_sim_transfer.config import (
    EXPORT_METADATA_KEY,
    HistoryProfile,
    TransferModelConfig,
)
from cosmos3_nano_sim_transfer.impl.rollout import (
    TransferRollout,
    initial_noise,
    sde_noise,
    text_positions,
    vision_positions,
)
from cosmos_interactive._network import CosmosMoTNetwork, CosmosMoTNetworkConfig
from cosmos_interactive._patchify import patchify_latent, unpatchify_latent
from cosmos_interactive._sampler import fixed_step_denoise
from cosmos_interactive._spec import Cosmos3InteractiveSpec, LLMSpec, VFMSpec

pytestmark = pytest.mark.ci_cpu

MODEL = TransferModelConfig.from_metadata(
    json.loads(
        (Path(__file__).parent / "data" / "checkpoint_metadata.json").read_text()
    )[EXPORT_METADATA_KEY]
)
LATENT_HW = (4, 6)  # a 2x3 token grid
CHANNELS = 4
SEED = 20261012


@pytest.fixture(scope="module")
def network() -> CosmosMoTNetwork:
    torch.manual_seed(0)
    spec = Cosmos3InteractiveSpec(
        llm=LLMSpec(
            hidden_size=32,
            num_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            intermediate_size=64,
            vocab_size=50,
            mrope_section=(4, 2, 2),
        ),
        vfm=VFMSpec(vae_latent_channels=CHANNELS),
    )
    net = CosmosMoTNetworkConfig(
        spec=spec, dtype=torch.float32, include_vision_tower=False
    ).setup()
    assert isinstance(net, CosmosMoTNetwork)
    return net.eval()


def _controls(count: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return [
        torch.randn(CHANNELS, 1, *LATENT_HW, generator=generator) for _ in range(count)
    ]


TEXT = torch.tensor([3, 14, 15, 9, 26, 5, 35])


def _rollout(
    network: CosmosMoTNetwork,
    history: HistoryProfile,
    model: TransferModelConfig = MODEL,
) -> TransferRollout:
    return TransferRollout(
        network,
        model,
        history=history,
        latent_hw=LATENT_HW,
        noise_dtype=torch.float32,
    )


@torch.no_grad()
def _reference(
    network: CosmosMoTNetwork,
    history: HistoryProfile,
    controls: list[torch.Tensor],
    t_list: tuple[float, ...] = MODEL.t_list,
) -> list[torch.Tensor]:
    """Keep every entry; select each pass's keys the way cosmos-framework counts them."""
    layers = list(network.layers)
    grid = (LATENT_HW[0] // 2, LATENT_HW[1] // 2)
    text = network.init_stream_cache(chunk_grid=(1, 1, 1), window_frames=1)
    network.prefill_text_stream(
        TEXT, text_positions(int(TEXT.numel())), text, native_gqa_attention=True
    )
    entries: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

    def select(for_target: bool) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        pinned = 2 * history.sink_frames
        recent = 2 * (history.window_frames - history.sink_frames - 1) + int(for_target)
        if len(entries) > pinned + recent:
            return entries[:pinned] + entries[len(entries) - recent :]
        return list(entries)

    def memory(chosen, index):  # noqa: ANN001, ANN202
        if not chosen:
            return None, None
        return (
            torch.cat([entry[index][0] for entry in chosen], dim=2),
            torch.cat([entry[index][1] for entry in chosen], dim=2),
        )

    def clean(latent, cos, sin, chosen):  # noqa: ANN001, ANN202
        h = network.vae2llm(patchify_latent(latent, 2))[None]
        kv = []
        for index, layer in enumerate(layers):
            hk, hv = memory(chosen, index)
            h, k, v = layer.gen_compute_with_kv(
                h,
                cos,
                sin,
                text.text_k[index],
                text.text_v[index],
                hk,
                hv,
                None,
                native_gqa_attention=True,
            )
            kv.append((k, v))
        return kv

    outputs = []
    shape = (CHANNELS, 1, *LATENT_HW)
    for k, control in enumerate(controls):
        positions = vision_positions(
            k, num_text=int(TEXT.numel()), grid_hw=grid, fps=30.0, model=MODEL
        )
        cos, sin = network._mrope(positions)
        entries.append(clean(control, cos, sin, select(False)))
        chosen = select(True)

        def velocity(x, t, cos=cos, sin=sin, chosen=chosen):  # noqa: ANN001, ANN202
            h = network.vae2llm(patchify_latent(x.reshape(shape), 2))
            h = (h + network.embed_uniform_timestep(t, h.shape[0]))[None]
            for index, layer in enumerate(layers):
                hk, hv = memory(chosen, index)
                h = layer.gen_compute(
                    h,
                    cos,
                    sin,
                    text.text_k[index],
                    text.text_v[index],
                    hk,
                    hv,
                    None,
                    native_gqa_attention=True,
                )
            flow = network.llm2vae(network.norm_moe_gen(h))[0]
            return unpatchify_latent(
                flow,
                channels=CHANNELS,
                grid_t=1,
                grid_h=grid[0],
                grid_w=grid[1],
                patch_spatial=2,
            ).reshape(-1)

        start = initial_noise(SEED, k, shape, device="cpu", dtype=torch.float32)
        numel = start.numel()
        denoised = fixed_step_denoise(
            velocity,
            start.reshape(-1),
            t_list=list(t_list),
            sde_noise_fn=lambda step, k=k, numel=numel: sde_noise(
                SEED, k, step, numel, device="cpu"
            ),
            fp32_math=True,
        ).reshape(shape)
        entries.append(clean(denoised, cos, sin, chosen))
        outputs.append(denoised)
    return outputs


def test_text_positions_count_from_zero_on_all_axes() -> None:
    ids = text_positions(458)
    assert ids.shape == (3, 458)
    assert torch.equal(ids[0], torch.arange(458, dtype=torch.float32))
    assert torch.equal(ids[0], ids[1]) and torch.equal(ids[0], ids[2])


@pytest.mark.parametrize("latent", [0, 1, 50, 150])
def test_vision_positions_scale_time_to_base_fps(latent: int) -> None:
    ids = vision_positions(
        latent, num_text=458, grid_hw=(15, 26), fps=30.0, model=MODEL
    )
    assert ids.shape == (3, 390)
    # Vision starts at text + 15000; at 30 fps each latent advances 24/30 = 0.8.
    assert torch.allclose(ids[0], torch.full((390,), 15458 + 0.8 * latent))
    # Spatial ids restart at zero, row-major over the token grid.
    assert torch.equal(ids[1], torch.arange(15).repeat_interleave(26).float())
    assert torch.equal(ids[2], torch.arange(26).repeat(15).float())


def test_vision_positions_without_fps_modulation_step_by_one() -> None:
    model = replace(MODEL, enable_fps_modulation=False)
    ids = vision_positions(3, num_text=10, grid_hw=(2, 3), fps=30.0, model=model)
    assert torch.equal(ids[0], torch.full((6,), 15013.0))


def test_initial_noise_is_seed_plus_latent_in_latent_layout() -> None:
    shape = (48, 1, 30, 52)
    noise = initial_noise(SEED, 5, shape, device="cpu")
    expected = torch.empty(shape, dtype=torch.bfloat16).normal_(
        generator=torch.Generator().manual_seed(SEED + 5)
    )
    assert noise.dtype == torch.bfloat16 and torch.equal(noise, expected)
    assert not torch.equal(noise, initial_noise(SEED, 6, shape, device="cpu"))


def test_renoise_seed_strides() -> None:
    numel = 48 * 30 * 52
    for latent, step in ((0, 0), (3, 2)):
        seed = SEED + latent * 1_000_003 + (step + 1) * 9_176
        expected = torch.empty(numel, dtype=torch.float32).normal_(
            generator=torch.Generator().manual_seed(seed)
        )
        assert torch.equal(sde_noise(SEED, latent, step, numel, device="cpu"), expected)


@pytest.mark.parametrize(
    "history",
    [
        HistoryProfile(5, 2),
        HistoryProfile(4, 0),
        HistoryProfile(3, 1),
        HistoryProfile(20, 1),
    ],
    ids=["w5s2", "w4s0", "w3s1", "w20s1-never-full"],
)
def test_stream_cache_selects_what_cosmos_framework_selects(
    network: CosmosMoTNetwork, history: HistoryProfile
) -> None:
    controls = _controls(11)
    rollout = _rollout(network, history)
    rollout.start(TEXT, seed=SEED)
    ours = [rollout.step(control) for control in controls]
    reference = _reference(network, history, controls)
    for k, (got, want) in enumerate(zip(ours, reference, strict=True)):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5, msg=f"latent {k}")


@pytest.mark.parametrize("steps", [2, 3])
def test_a_shorter_schedule_runs_one_pass_per_step(
    network: CosmosMoTNetwork, steps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fewer sampler steps: the same rollout, one denoise pass per sigma of the new schedule."""
    history = HistoryProfile(5, 2)
    controls = _controls(9)
    model = MODEL.with_schedule(steps=steps)
    rollout = _rollout(network, history, model)
    passes = []
    flow = rollout._flow
    monkeypatch.setattr(
        rollout, "_flow", lambda *a, **kw: passes.append(1) or flow(*a, **kw)
    )
    rollout.start(TEXT, seed=SEED)
    ours = [rollout.step(control) for control in controls]
    assert len(passes) == steps * len(controls)
    reference = _reference(network, history, controls, model.t_list)
    for k, (got, want) in enumerate(zip(ours, reference, strict=True)):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5, msg=f"latent {k}")
    full = _reference(network, history, controls[:1])
    assert not torch.allclose(ours[0], full[0], rtol=1e-3, atol=1e-3)


def test_memory_is_bounded_by_the_window(network: CosmosMoTNetwork) -> None:
    history = HistoryProfile(4, 1)
    rollout = _rollout(network, history)
    block = 2 * rollout.tokens_per_latent
    buffers = [gen._k.data_ptr() for gen in rollout.cache.gen]
    assert rollout.cache.gen[0]._k.shape[2] == history.window_frames * block
    rollout.start(TEXT, seed=SEED)
    for control in _controls(9):
        rollout.step(control)
        assert rollout.cached_tokens <= history.window_frames * block
    assert rollout.cached_tokens == history.window_frames * block
    assert [gen._k.data_ptr() for gen in rollout.cache.gen] == buffers


def test_reset_and_reseed(network: CosmosMoTNetwork) -> None:
    controls = _controls(4)
    rollout = _rollout(network, HistoryProfile(3, 1))

    def run(seed: int) -> list[torch.Tensor]:
        rollout.start(TEXT, seed=seed)
        return [rollout.step(control) for control in controls]

    first = run(SEED)
    text_k = rollout.cache.text_k[0]
    again = run(SEED)
    assert rollout.cache.text_k[0] is text_k  # same prompt: not prefilled again
    for a, b in zip(first, again, strict=True):
        assert torch.equal(a, b)
    other = run(SEED + 1)
    assert not torch.equal(first[0], other[0])

    rollout.start(TEXT[:-1], seed=SEED)
    assert rollout.cache.text_k[0] is not text_k
    new_text_k = rollout.cache.text_k[0]
    assert new_text_k is not None and new_text_k.shape[2] == TEXT.numel() - 1


def test_step_uses_the_seeded_noise(network: CosmosMoTNetwork) -> None:
    control = _controls(1)[0]
    rollout = _rollout(network, HistoryProfile(3, 1))
    rollout.start(TEXT, seed=SEED)
    seeded = rollout.step(control)
    rollout.start(TEXT, seed=SEED)
    explicit = rollout.step(control, noise=rollout.initial_noise(0))
    assert torch.equal(seeded, explicit)
    assert torch.equal(
        rollout.initial_noise(2),
        initial_noise(
            SEED, 2, (CHANNELS, 1, *LATENT_HW), device="cpu", dtype=torch.float32
        ),
    )


def test_replay_commits_the_given_latent(network: CosmosMoTNetwork) -> None:
    """Teacher forcing: the next step reads the replayed latent, not the generated one."""
    controls = _controls(3)
    rollout = _rollout(network, HistoryProfile(3, 1))
    rollout.start(TEXT, seed=SEED)
    free = [rollout.step(control) for control in controls]

    rollout.start(TEXT, seed=SEED)
    same = [rollout.step(c, replay=x) for c, x in zip(controls, free, strict=True)]
    assert all(torch.equal(a, b) for a, b in zip(free, same, strict=True))

    rollout.start(TEXT, seed=SEED)
    first = rollout.step(controls[0], replay=free[0] + 0.5)
    assert torch.equal(first, free[0])  # the generated latent is still returned
    assert not torch.allclose(rollout.step(controls[1]), free[1])

    with pytest.raises(ValueError):
        rollout.step(controls[2], replay=free[2][:, :, :2])


def test_prompt_longer_than_the_static_cache_length_runs(
    network: CosmosMoTNetwork,
) -> None:
    """``text_cache_max_len`` pads a static-shape cache; it does not limit eager prompts."""
    long_text = 3 + torch.arange(MODEL.text_cache_max_len + 1) % 40
    rollout = _rollout(network, HistoryProfile(3, 1))
    rollout.start(long_text, seed=SEED)
    assert rollout.num_text == MODEL.text_cache_max_len + 1
    assert torch.isfinite(rollout.step(_controls(1)[0], last=True)).all()


def test_session_guards(network: CosmosMoTNetwork) -> None:
    rollout = _rollout(network, HistoryProfile(3, 1))
    control = _controls(1)[0]
    with pytest.raises(RuntimeError):
        rollout.step(control)
    rollout.start(TEXT, seed=SEED)
    rollout.step(control, last=True)
    with pytest.raises(RuntimeError):
        rollout.step(control)
    with pytest.raises(ValueError):
        rollout.start(torch.zeros(0, dtype=torch.long), seed=SEED)
    with pytest.raises(ValueError):
        rollout.start(TEXT, seed=SEED)
        rollout.step(control[:, :, :2])
