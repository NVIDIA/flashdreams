# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the embedding adapters.

A tiny randomly-initialised ``CosmosMoTNetwork`` stands in for the 15.2B one, as
in ``test_forward.py``. The arithmetic inside ``embed_tokens``, ``vae2llm`` and
the timestep MLP is not what these check -- it is upstream's and already tested.
What they check is that the split keeps the pieces in the same relationship the
dense path puts them in.

The anchor is ``denoise_forward`` itself. It embeds, runs the layers over one
packed square-masked sequence and projects back, which is the whole path we have
taken apart. Configure it so its packed sequence is text followed by one chunk
with an empty memory -- the prefill shape -- and the two must agree to
floating-point noise. That one equality covers the embedding split, the layer
split, the final norm and the output projection at once, and it is the only test
here that would notice if a later change to any of them drifted.
"""

from __future__ import annotations

import pytest
import torch
from cosmos3_nano_sim_auto.impl.transformer.embedding import (
    flow_prediction,
    text_embeddings,
    vision_embeddings,
)
from cosmos3_nano_sim_auto.impl.transformer.forward import (
    run_chunk_layers,
    run_text_layers,
)
from cosmos_interactive._network import (
    CosmosMoTNetwork,
    CosmosMoTNetworkConfig,
)
from cosmos_interactive._spec import Cosmos3InteractiveSpec, LLMSpec, VFMSpec

pytestmark = pytest.mark.ci_cpu

SPEC = Cosmos3InteractiveSpec(
    llm=LLMSpec(
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=16,
        vocab_size=100,
        mrope_section=(2, 1, 1),
    ),
    # patch_latent_dim = 2*2*4 = 16.
    vfm=VFMSpec(vae_latent_channels=4, latent_patch_size=2, time_embed_freq_dim=8),
)
TEXT, VIS = 3, 6
PATCH_DIM = SPEC.vfm.patch_latent_dim
SIGMA = 0.75


def build(seed: int = 0) -> CosmosMoTNetwork:
    torch.manual_seed(seed)
    return (
        CosmosMoTNetworkConfig(
            spec=SPEC, dtype=torch.float32, include_vision_tower=False
        )
        .setup()
        .eval()
    )


def patches(count: int = VIS, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, PATCH_DIM, generator=generator)


def text_ids(seed: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, SPEC.llm.vocab_size, (TEXT,), generator=generator)


# --- parity with the dense path ---------------------------------------------


@torch.no_grad()
def test_prompt_embedding_matches_the_packed_one() -> None:
    network = build()
    ids = text_ids()

    packed = network._assemble_input(
        ids,
        patches(),
        torch.full((VIS,), SIGMA),
        torch.cat(
            [torch.zeros(TEXT, dtype=torch.bool), torch.ones(VIS, dtype=torch.bool)]
        ),
    )

    assert torch.equal(text_embeddings(network, ids), packed[:, :TEXT])


@torch.no_grad()
def test_vision_embedding_matches_the_packed_one() -> None:
    network = build()
    vision = patches()
    timestep = torch.full((VIS,), SIGMA)

    packed = network._assemble_input(
        text_ids(),
        vision,
        timestep,
        torch.cat(
            [torch.zeros(TEXT, dtype=torch.bool), torch.ones(VIS, dtype=torch.bool)]
        ),
    )

    ours = vision_embeddings(network, vision, timestep=timestep)
    assert torch.equal(ours, packed[:, TEXT:])


@torch.no_grad()
def test_the_split_path_reproduces_denoise_forward() -> None:
    """Embed, run, project -- apart -- against the same thing done together."""
    network = build()
    ids, vision = text_ids(), patches()
    timestep = torch.full((VIS,), SIGMA)

    generator = torch.Generator().manual_seed(3)
    text_positions = torch.rand(3, TEXT, generator=generator) * 5.0
    vision_positions = torch.rand(3, VIS, generator=generator) * 5.0 + 100.0

    # A chunk sees the prompt and part of itself. Random rather than the real
    # rule, so the test pins the plumbing and not the mask's contents -- but with
    # the diagonal forced, since a query row that attends to nothing is a NaN
    # rather than an interesting case.
    chunk_mask = torch.rand(VIS, TEXT + VIS, generator=generator) > 0.3
    chunk_mask[:, TEXT:] |= torch.eye(VIS, dtype=torch.bool)

    order = torch.arange(TEXT)
    dense_mask = torch.zeros(TEXT + VIS, TEXT + VIS, dtype=torch.bool)
    dense_mask[:TEXT, :TEXT] = order[None, :] <= order[:, None]
    dense_mask[TEXT:] = chunk_mask
    is_gen = torch.cat(
        [torch.zeros(TEXT, dtype=torch.bool), torch.ones(VIS, dtype=torch.bool)]
    )

    theirs = network.denoise_forward(
        text_ids=ids,
        vision_patches=vision,
        vision_timestep=timestep,
        is_gen=is_gen,
        position_ids=torch.cat([text_positions, vision_positions], dim=1),
        mask=dense_mask,
    )

    _text_out, text_kv = run_text_layers(
        network.layers, SPEC.llm, text_embeddings(network, ids), text_positions
    )
    empty = torch.zeros(1, SPEC.llm.num_key_value_heads, 0, SPEC.llm.head_dim)
    hidden, _chunk_kv = run_chunk_layers(
        network.layers,
        SPEC.llm,
        vision_embeddings(network, vision, timestep=timestep),
        vision_positions,
        text_kv=text_kv,
        memory_kv=[(empty, empty)] * SPEC.llm.num_layers,
        mask=chunk_mask,
        final_norm=network.norm_moe_gen,
    )
    ours = flow_prediction(network, hidden)

    assert ours.shape == theirs.shape == (1, VIS, PATCH_DIM)
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_skipping_the_final_norm_would_not_reproduce_it() -> None:
    """The pairing in the docstring, made a failure rather than a convention."""
    network = build()
    hidden = torch.randn(
        1, VIS, SPEC.llm.hidden_size, generator=torch.Generator().manual_seed(4)
    )

    assert not torch.allclose(
        flow_prediction(network, hidden),
        flow_prediction(network, network.norm_moe_gen(hidden)),
    )


# --- the timestep -----------------------------------------------------------


@torch.no_grad()
def test_a_clean_pack_is_the_patch_projection_alone() -> None:
    network = build()
    vision = patches()

    clean = vision_embeddings(
        network, vision, is_noisy=torch.zeros(VIS, dtype=torch.bool)
    )

    assert torch.equal(clean, network.vae2llm(vision)[None])


@torch.no_grad()
def test_clean_is_not_the_timestep_of_zero() -> None:
    # The whole reason ``timestep=None`` exists. sinusoidal(0) is zeros and ones,
    # not zeros, and the MLP after it carries a bias, so this vector is large.
    network = build()
    at_zero = network.time_embedder(torch.zeros(1))

    assert at_zero.abs().max() > 1e-3
    assert not torch.allclose(
        vision_embeddings(
            network, patches(), is_noisy=torch.zeros(VIS, dtype=torch.bool)
        ),
        vision_embeddings(network, patches(), timestep=0.0),
    )


@torch.no_grad()
def test_only_the_noisy_tokens_move() -> None:
    network = build()
    vision = patches()
    is_noisy = torch.zeros(VIS, dtype=torch.bool)
    is_noisy[2:4] = True

    embedded = vision_embeddings(network, vision, timestep=SIGMA, is_noisy=is_noisy)
    projected = network.vae2llm(vision)[None]

    assert torch.equal(embedded[:, ~is_noisy], projected[:, ~is_noisy])
    for token in is_noisy.nonzero().squeeze(-1).tolist():
        assert not torch.allclose(embedded[:, token], projected[:, token])


@torch.no_grad()
def test_an_all_clean_mask_needs_no_timestep() -> None:
    # The prefill pass: every token clean, so the term drops rather than being
    # computed and multiplied by zero.
    network = build()
    vision = patches()
    clean = torch.zeros(VIS, dtype=torch.bool)

    assert torch.equal(
        vision_embeddings(network, vision, is_noisy=clean),
        vision_embeddings(network, vision, timestep=SIGMA, is_noisy=clean),
    )


@torch.no_grad()
def test_one_sigma_expanded_matches_one_per_token() -> None:
    network = build()
    vision = patches()

    uniform = vision_embeddings(network, vision, timestep=SIGMA)
    per_token = vision_embeddings(network, vision, timestep=torch.full((VIS,), SIGMA))

    torch.testing.assert_close(uniform, per_token, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_different_sigmas_give_different_embeddings() -> None:
    network = build()
    vision = patches()

    assert not torch.allclose(
        vision_embeddings(network, vision, timestep=0.25),
        vision_embeddings(network, vision, timestep=0.9),
    )


# --- what is refused --------------------------------------------------------


def test_a_noisy_token_without_a_timestep_is_refused() -> None:
    network = build()
    with pytest.raises(ValueError, match="6 of 6 tokens are noisy"):
        vision_embeddings(network, patches())


def test_a_partly_noisy_pack_without_a_timestep_is_refused() -> None:
    network = build()
    is_noisy = torch.zeros(VIS, dtype=torch.bool)
    is_noisy[1] = True

    with pytest.raises(ValueError, match="1 of 6 tokens are noisy"):
        vision_embeddings(network, patches(), is_noisy=is_noisy)


def test_patches_of_the_wrong_width_are_refused() -> None:
    network = build()
    with pytest.raises(ValueError, match="are 15 wide but vae2llm takes 16"):
        vision_embeddings(network, torch.zeros(VIS, 15), timestep=SIGMA)


def test_a_mask_of_the_wrong_length_is_refused() -> None:
    network = build()
    with pytest.raises(ValueError, match=r"is_noisy is \(5,\) for 6 tokens"):
        vision_embeddings(
            network,
            patches(),
            timestep=SIGMA,
            is_noisy=torch.zeros(5, dtype=torch.bool),
        )


def test_a_timestep_per_token_of_the_wrong_length_is_refused() -> None:
    network = build()
    with pytest.raises(ValueError, match="timestep has 4 values for 6 tokens"):
        vision_embeddings(network, patches(), timestep=torch.full((4,), SIGMA))


def test_two_dimensional_patches_are_required() -> None:
    network = build()
    with pytest.raises(ValueError, match=r"got \(2, 6, 16\)"):
        vision_embeddings(network, torch.zeros(2, VIS, PATCH_DIM), timestep=SIGMA)


def test_token_ids_must_be_one_dimensional() -> None:
    network = build()
    with pytest.raises(ValueError, match=r"got \(1, 3\)"):
        text_embeddings(network, text_ids()[None])


def test_hidden_states_of_the_wrong_width_are_refused() -> None:
    network = build()
    with pytest.raises(ValueError, match="are 31 wide, but the model's are 32"):
        flow_prediction(network, torch.zeros(1, VIS, 31))


@pytest.mark.parametrize("noisy", [False, True])
@torch.no_grad()
def test_rig_embeddings_follow_physical_camera_ids_for_clean_and_noisy_tokens(noisy):
    network = build()
    network.rig_view_embed = torch.nn.Embedding(12, SPEC.llm.hidden_size)
    network.rig_view_embed.weight.copy_(
        torch.arange(12)[:, None].expand_as(network.rig_view_embed.weight)
    )
    vision = patches()
    # A reordered subset must use its physical rows, not local IDs 0 and 1.
    ids = torch.tensor([6, 6, 6, 4, 4, 4])
    is_noisy = torch.full((VIS,), noisy)
    with_rig = vision_embeddings(
        network, vision, timestep=SIGMA, is_noisy=is_noisy, view_ids=ids
    )
    del network.rig_view_embed
    without_rig = vision_embeddings(network, vision, timestep=SIGMA, is_noisy=is_noisy)
    torch.testing.assert_close(
        with_rig - without_rig, ids[None, :, None].expand_as(with_rig).float()
    )


def test_rig_checkpoint_requires_camera_identity_instead_of_silently_ignoring_it():
    network = build()
    network.rig_view_embed = torch.nn.Embedding(12, SPEC.llm.hidden_size)
    with pytest.raises(ValueError, match="canonical camera ID"):
        vision_embeddings(network, patches(), timestep=SIGMA)
