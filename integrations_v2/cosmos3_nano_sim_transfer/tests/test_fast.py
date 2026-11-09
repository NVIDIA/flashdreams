# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fast rollout on fixed K/V buffers equals the eager rollout (tiny network, CPU).

CUDA graphs and compile need a GPU; on the CPU the fast rollout runs its
passes eagerly, which is exactly what a graph captures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from cosmos3_nano_sim_transfer.config import (
    EXPORT_METADATA_KEY,
    HistoryProfile,
    TransferModelConfig,
)
from cosmos3_nano_sim_transfer.impl.fast import (
    ACCELERATIONS,
    FastTransferRollout,
    make_rollout,
)
from cosmos3_nano_sim_transfer.impl.rollout import TransferRollout
from cosmos_interactive._network import CosmosMoTNetwork, CosmosMoTNetworkConfig
from cosmos_interactive._spec import Cosmos3InteractiveSpec, LLMSpec, VFMSpec

pytestmark = pytest.mark.ci_cpu

MODEL = TransferModelConfig.from_metadata(
    json.loads(
        (Path(__file__).parent / "data" / "checkpoint_metadata.json").read_text()
    )[EXPORT_METADATA_KEY]
)
LATENT_HW = (4, 6)
CHANNELS = 4
SEED = 20261012
TEXT = torch.tensor([3, 14, 15, 9, 26, 5, 35])


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


def _controls(count: int, seed: int = 7) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(CHANNELS, 1, *LATENT_HW, generator=generator) for _ in range(count)
    ]


def _eager(
    network: CosmosMoTNetwork,
    history: HistoryProfile,
    model: TransferModelConfig = MODEL,
) -> TransferRollout:
    return TransferRollout(
        network, model, history=history, latent_hw=LATENT_HW, noise_dtype=torch.float32
    )


def _fast(
    network: CosmosMoTNetwork,
    history: HistoryProfile,
    *,
    model: TransferModelConfig = MODEL,
    native_gqa_attention: bool = True,
    text_slots: int | None = None,
) -> FastTransferRollout:
    return FastTransferRollout(
        network,
        model,
        history=history,
        latent_hw=LATENT_HW,
        noise_dtype=torch.float32,
        cuda_graphs=False,
        native_gqa_attention=native_gqa_attention,
        text_slots=text_slots,
    )


@pytest.mark.parametrize(
    "history",
    [HistoryProfile(5, 2), HistoryProfile(4, 0), HistoryProfile(3, 1)],
    ids=["w5s2", "w4s0", "w3s1"],
)
def test_fast_rollout_equals_the_eager_rollout(
    network: CosmosMoTNetwork, history: HistoryProfile
) -> None:
    """Through the fill and past it (the window rolls in place in the buffer)."""
    controls = _controls(11)
    eager = _eager(network, history)
    eager.start(TEXT, seed=SEED)
    want = [eager.step(c) for c in controls]
    fast = _fast(network, history)
    fast.start(TEXT, seed=SEED)
    got = [fast.step(c) for c in controls]
    for k, (a, b) in enumerate(zip(got, want, strict=True)):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6, msg=f"latent {k}")
    assert fast.cached_tokens == eager.cached_tokens


def test_fast_rollout_on_fewer_steps_equals_the_eager_rollout(
    network: CosmosMoTNetwork,
) -> None:
    """The 2-step skip schedule: one denoise pass per step on the fast path too."""
    history = HistoryProfile(5, 2)
    model = MODEL.with_schedule(steps=2)
    controls = _controls(9)
    eager = _eager(network, history, model)
    eager.start(TEXT, seed=SEED)
    want = [eager.step(c) for c in controls]
    fast = _fast(network, history, model=model)
    passes = []
    flow = fast._flow
    fast._flow = lambda *a, **kw: passes.append(1) or flow(*a, **kw)  # ty: ignore[invalid-assignment]
    fast.start(TEXT, seed=SEED)
    got = [fast.step(c) for c in controls]
    assert len(passes) == 2 * len(controls)
    for k, (a, b) in enumerate(zip(got, want, strict=True)):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6, msg=f"latent {k}")


def test_fast_rollout_replays_and_restarts_like_the_eager_rollout(
    network: CosmosMoTNetwork,
) -> None:
    history = HistoryProfile(3, 1)
    controls = _controls(6)
    replays = _controls(6, seed=11)
    eager = _eager(network, history)
    fast = _fast(network, history)
    for rollout in (eager, fast):
        rollout.start(TEXT, seed=SEED)
        for c in controls:
            rollout.step(c)
    # A new session reuses the buffers; teacher forcing commits the given latents.
    results = []
    for rollout in (eager, fast):
        rollout.start(TEXT, seed=SEED + 1)
        results.append(
            [rollout.step(c, replay=r) for c, r in zip(controls, replays, strict=True)]
        )
    for k, (a, b) in enumerate(zip(*results, strict=True)):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6, msg=f"latent {k}")


def test_a_new_prompt_moves_into_the_text_slots(network: CosmosMoTNetwork) -> None:
    history = HistoryProfile(3, 1)
    controls = _controls(4)
    fast = _fast(network, history)
    buffers = [k.data_ptr() for k in fast.k_buffers]
    for text in (TEXT, TEXT[:-2], TEXT):
        eager = _eager(network, history)
        eager.start(text, seed=SEED)
        fast.start(text, seed=SEED)
        assert fast.text_start == fast.text_slots - text.numel()
        for c in controls:
            torch.testing.assert_close(fast.step(c), eager.step(c), rtol=0, atol=1e-6)
    assert [k.data_ptr() for k in fast.k_buffers] == buffers


def test_the_cache_lives_in_the_buffers(network: CosmosMoTNetwork) -> None:
    history = HistoryProfile(4, 1)
    fast = _fast(network, history)
    block = 2 * fast.tokens_per_latent
    for gen, k in zip(fast.cache.gen, fast.k_buffers, strict=True):
        assert gen._k.data_ptr() == k[:, :, fast.text_slots :].data_ptr()
        assert gen._k.shape[2] == history.window_frames * block
    assert fast.steady_control_start == (history.window_frames - 1) * block
    assert fast.steady_rgb_start == fast.steady_control_start + block // 2


def test_prompt_longer_than_the_text_slots_is_refused(
    network: CosmosMoTNetwork,
) -> None:
    fast = _fast(network, HistoryProfile(3, 1), text_slots=6)
    with pytest.raises(ValueError, match="7 tokens"):
        fast.start(TEXT, seed=SEED)
    fast.start(TEXT[:6], seed=SEED)
    assert fast.text_start == 0


def test_graphs_need_cuda_and_native_gqa(network: CosmosMoTNetwork) -> None:
    with pytest.raises(ValueError, match="CUDA"):
        FastTransferRollout(
            network, MODEL, history=HistoryProfile(3, 1), latent_hw=LATENT_HW
        )
    with pytest.raises(ValueError, match="grouped-query"):
        _fast(network, HistoryProfile(3, 1), native_gqa_attention=False)


def test_make_rollout(network: CosmosMoTNetwork) -> None:
    assert ACCELERATIONS == ("eager", "graphs", "compile")
    kwargs = {"history": HistoryProfile(3, 1), "latent_hw": LATENT_HW}
    rollout = make_rollout("eager", network, MODEL, **kwargs)
    assert type(rollout) is TransferRollout
    with pytest.raises(ValueError, match="CUDA"):
        make_rollout("graphs", network, MODEL, **kwargs)
    with pytest.raises(ValueError, match="unknown acceleration"):
        make_rollout("fast", network, MODEL, **kwargs)
