# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokens into hidden states, and hidden states back into flow.

The adapters at either end of the layer stack, called directly rather than
through ``CosmosMoTNetwork._assemble_input``: that scatters both embeddings into
one buffer covering a whole packed sequence, and a rollout has no such sequence.
"""

from __future__ import annotations

import torch
from cosmos_interactive._network import CosmosMoTNetwork
from torch import Tensor


def text_embeddings(network: CosmosMoTNetwork, text_ids: Tensor) -> Tensor:
    """The prompt's hidden states, ``[1, num_und, hidden]``.

    Ready for :func:`~cosmos3_nano_sim_auto.impl.transformer.forward.run_text_layers`. Done once per
    rollout: the prompt does not change, and neither do the keys and values it
    produces.

    Args:
        text_ids: ``[num_und]`` token ids, from
            :func:`~cosmos3_nano_sim_auto.inputs.prompt_token_ids`.
    """
    if text_ids.ndim != 1:
        raise ValueError(f"expected [num_und] token ids, got {tuple(text_ids.shape)}.")
    device = network.embed_tokens.weight.device
    return network.embed_tokens(text_ids.to(device)).to(network.config.dtype)[None]


def vision_embeddings(
    network: CosmosMoTNetwork,
    patches: Tensor,
    *,
    timestep: float | Tensor | None = None,
    is_noisy: Tensor | None = None,
    view_ids: Tensor | None = None,
) -> Tensor:
    """Patchified latents plus their timestep, ``[1, N, hidden]``.

    Ready for :func:`~cosmos3_nano_sim_auto.impl.transformer.forward.run_chunk_layers`, whether the
    tokens are a prefill pack or one chunk.

    Args:
        patches: ``[N, patch_latent_dim]``, from
            :func:`~cosmos3_nano_sim_auto.inputs.control_tokens` or from the rollout's
            own noisy latent. Rows must be in the order the pack stamps metadata
            in -- view-outer, frame-inner, spatial-innermost.
        timestep: The raw sigma, scaled by ``vfm.timestep_scale`` here rather
            than by the caller. A scalar embeds once and expands; an ``[N]``
            tensor embeds per token. Omit it for wholly clean tokens, which is
            not the same as passing 0: ``time_embedder(0)`` is a large nonzero
            vector that would place the token at the end of the schedule rather
            than off it.
        is_noisy: ``[N]`` bool, ordinarily ``fields.is_noisy`` off the pack the
            metadata came from. Clean tokens skip the timestep embedding.
            ``None`` treats every token as noisy.
        view_ids: Canonical camera IDs, one per token, for checkpoints carrying
            a learned rig embedding. Applied to clean and noisy RGB tokens.

    Raises:
        ValueError: the patch width, the mask length or the timestep length
            disagrees with the tokens, or a noisy token has no timestep to embed.
    """
    vfm = network.config.spec.vfm
    device = network.embed_tokens.weight.device
    dtype = network.config.dtype

    if patches.ndim != 2:
        raise ValueError(
            f"expected [N, patch_latent_dim] patches, got {tuple(patches.shape)}."
        )
    count, width = patches.shape
    if width != vfm.patch_latent_dim:
        raise ValueError(
            f"patches are {width} wide but vae2llm takes {vfm.patch_latent_dim} "
            f"({vfm.latent_patch_size}x{vfm.latent_patch_size} patches of "
            f"{vfm.vae_latent_channels} channels)."
        )

    h = network.vae2llm(patches.to(device=device, dtype=dtype))
    if hasattr(network, "rig_view_embed"):
        if view_ids is None or view_ids.shape != (count,):
            raise ValueError(
                "This checkpoint requires a canonical camera ID for every RGB token"
            )
        h = h + network.rig_view_embed(view_ids.to(device=device)).to(h.dtype)

    noisy = None
    if is_noisy is not None:
        if is_noisy.shape != (count,):
            raise ValueError(
                f"is_noisy is {tuple(is_noisy.shape)} for {count} tokens; it must be [N]."
            )
        noisy = is_noisy.to(device)
        # Nothing noisy means the whole term is multiplied by zero, so skip it.
        # The prefill pass takes this branch for every one of its tokens.
        if not bool(noisy.any()):
            return h[None]

    if timestep is None:
        raise ValueError(
            f"{count if noisy is None else int(noisy.sum())} of {count} tokens are noisy "
            "and need a timestep. Pass timestep=..., or mark them clean with is_noisy."
        )

    embedded = _timestep_embedding(network, timestep, count)
    if noisy is not None:
        embedded = embedded * noisy.to(dtype)[:, None]
    return (h + embedded)[None]


def flow_prediction(network: CosmosMoTNetwork, h: Tensor) -> Tensor:
    """Generation-half hidden states -> flow, ``[B, N, patch_latent_dim]``.

    Applied whole, since a chunk's hidden states are all vision and there is
    nothing to select out.

    ``h`` must be normed already: the generation half's final norm belongs one
    call earlier, as ``run_chunk_layers(final_norm=network.norm_moe_gen)``. Skip
    it and the prediction comes out finite, plausibly scaled, and wrong.
    """
    hidden = network.config.spec.llm.hidden_size
    if h.ndim != 3:
        raise ValueError(
            f"expected [B, N, hidden] hidden states, got {tuple(h.shape)}."
        )
    if h.shape[-1] != hidden:
        raise ValueError(
            f"hidden states are {h.shape[-1]} wide, but the model's are {hidden}."
        )
    device = network.embed_tokens.weight.device
    return network.llm2vae(h.to(device=device, dtype=network.config.dtype))


def _timestep_embedding(
    network: CosmosMoTNetwork, timestep: float | Tensor, count: int
) -> Tensor:
    """``[count, hidden]`` for a scalar sigma or one per token."""
    device = network.embed_tokens.weight.device
    dtype = network.config.dtype
    # float32 through the embedder either way: ``embed_uniform_timestep`` casts,
    # and the two paths agreeing matters more than matching a bf16 caller.
    scaled = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)

    if scaled.numel() == 1:
        return network.embed_uniform_timestep(scaled, count).to(dtype)
    if scaled.numel() != count:
        raise ValueError(
            f"timestep has {scaled.numel()} values for {count} tokens; "
            "pass one scalar for the whole pack or one value per token."
        )
    return network.time_embedder(scaled * network.config.spec.vfm.timestep_scale).to(
        dtype
    )
