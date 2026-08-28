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

"""Omnidreams host adapter for the Clean Forcing drift corrector.

Implements the three per-model functions from the library design
(``LIBRARY_DESIGN.md`` in the HY port): rollout capture, and the
counterfactual x0 probe (history substitution + matched-state predict).

Host specifics vs HY-WorldPlay: history lives ONLY in the rolling
self-attention KV window (``window_size_t`` latent frames; no explicit
latent-history tensor and no long-range memory reads), so substituting a
history means resetting the per-block KV caches and replaying the prefix
chunks through ``finalize_kv_cache`` forwards at the model's own
``context_noise`` timestep. RoPE is absolute (``shift_t(ar_idx)``), so a
replay at the original chunk indices reproduces positions exactly.
"""

from __future__ import annotations

import dataclasses
import gc
import os
from dataclasses import dataclass
from pathlib import Path

# Must land before the first CUDA allocation: long captures fragment the
# allocator; expandable segments keep the probe phase inside the VRAM share
# left over by co-tenant jobs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from omnidreams.config import SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
from omnidreams.pipeline import OmnidreamsPipeline, OmnidreamsPipelineCache
from omnidreams.transformer import CosmosTransformer, CosmosTransformerCache
from torch import Tensor

from flashdreams.infra.config import derive_config

## Pipeline construction

CONTEXT_NOISE_SEED = 77_000
"""Base seed for the per-chunk context-noise draws used by history replays.
Both branches of a counterfactual probe re-noise their history chunks with
the SAME eps (seeded per chunk index), so the prediction gap isolates the
history *content* difference."""


def build_pipeline(
    *,
    with_oneshot_encoders: bool,
    seed: int | None = 42,
) -> OmnidreamsPipeline:
    """Build the distilled chunk2 Omnidreams pipeline for capture / probes.

    Compile and CUDA graphs are disabled everywhere: probes reset and replay
    the KV caches out of the normal AR order, which a captured graph (whose
    slot pointers bake in one cache object) cannot express — and eager keeps
    capture and probe numerics identical.

    Args:
        with_oneshot_encoders: Load the Cosmos-Reason1 text encoder and the
            first-frame VAE encoder. Pair-building needs them; the gate
            rebuilds caches from embeddings stored in the clips and skips
            the ~14 GB text encoder.
        seed: Diffusion-model RNG seed (callers typically re-seed per
            rollout via ``diffusion_model._rng``).

    Returns:
        Pipeline with the checkpoint loaded, on CUDA.
    """
    cfg = derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        enable_sync_and_profile=False,
        diffusion_model=dict(
            seed=seed,
            transformer=dict(compile_network=False, use_cuda_graph=False),
        ),
    )
    if not with_oneshot_encoders:
        cfg = dataclasses.replace(cfg, text_encoder=None, image_encoder=None)
    pipe = cfg.setup()
    assert isinstance(pipe, OmnidreamsPipeline)
    return pipe.to("cuda")


## Rollout capture


@dataclass
class ChunkSnapshot:
    """Per-chunk state captured from a rollout, sufficient to replay probes."""

    clean_latent: Tensor
    """Patchified x0 the sampler produced for this chunk (image latent
    injected at chunk 0, as the AR cache consumed it)."""

    hdmap: Tensor
    """Patchified per-AR-step HDMap conditioning exactly as
    ``predict_flow`` consumed it (post streaming-encoder, post patchify)."""

    def to(self, device: torch.device | str) -> "ChunkSnapshot":
        """Return a copy with both tensors on ``device``."""
        return ChunkSnapshot(
            clean_latent=self.clean_latent.to(device), hdmap=self.hdmap.to(device)
        )


def capture_rollout(
    pipe: OmnidreamsPipeline,
    cache: OmnidreamsPipelineCache,
    *,
    hdmap_video: Tensor,
    num_chunk: int,
    noise_seed: int,
) -> tuple[list[ChunkSnapshot], Tensor]:
    """Roll out ``num_chunk`` chunks and snapshot the per-chunk probe inputs.

    Args:
        pipe: Pipeline from :func:`build_pipeline`.
        cache: Fresh per-rollout cache (``initialize_cache`` /
            ``initialize_cache_from_embeddings``); consumed by the rollout.
        hdmap_video: ``[B=1, V=1, T, 3, H, W]`` HDMap pixels in ``[-1, 1]``;
            must cover ``5 + (num_chunk - 1) * 8`` frames on the chunk2 host.
        noise_seed: Seed for the diffusion model's noise generator (initial
            noise + context-noise draws), making captures reproducible.

    Returns:
        ``(snaps, video)`` — one :class:`ChunkSnapshot` per chunk (tensors
        stay on the compute device) and the decoded rollout
        ``[B, V, T, 3, H, W]`` on CPU for eyeballing.
    """
    device = pipe.device
    pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(noise_seed)

    snaps: list[ChunkSnapshot] = []
    chunks: list[Tensor] = []
    start = 0
    for ar_idx in range(num_chunk):
        num_frames = pipe.get_num_frames(ar_idx)
        end = start + num_frames
        assert end <= hdmap_video.shape[2], (
            f"hdmap video too short: chunk {ar_idx} needs frames "
            f"[{start}, {end}) but only {hdmap_video.shape[2]} available."
        )
        chunk = pipe.generate(ar_idx, cache, hdmap=hdmap_video[:, :, start:end])
        fs = cache.final_state
        assert fs is not None and isinstance(fs.input, Tensor)
        snaps.append(
            ChunkSnapshot(
                clean_latent=fs.clean_latent.detach().clone(),
                hdmap=fs.input.detach().clone(),
            )
        )
        pipe.finalize(ar_idx, cache)
        chunks.append(chunk.cpu())
        start = end

    video = torch.cat(chunks, dim=2)
    del cache, chunks
    gc.collect()
    torch.cuda.empty_cache()
    return snaps, video


## Counterfactual x0 probes


@torch.no_grad()
def swap_text_kv(network, tc: CosmosTransformerCache, text_embeddings: Tensor) -> None:
    """Swap the per-block cross-attention (text) KV for another clip's prompt.

    Mirrors the cross-attn half of ``CosmosDiTNetwork.initialize_cache`` so
    one long-lived transformer cache (rope, masks, rolling self-attn
    buffers) can serve clips with different prompts.
    """
    w = network.blocks[0].cross_attn.k_proj.weight
    ctx = text_embeddings.to(device=w.device, dtype=w.dtype)
    if network.config.use_crossattn_projection:
        ctx = network.crossattn_proj(ctx)
    for block, bc in zip(network.blocks, tc.network_cache.block_caches):
        bc.cross_attn = block.cross_attn.initialize_cache(ctx)


def reset_history(tc: CosmosTransformerCache) -> None:
    """Reset the rolling self-attention KV caches to the empty state.

    The cross-attention (text) KV and the image/mask fields are per-rollout
    constants and stay untouched. After a reset the next replay must start
    at chunk 0 (``BlockKVCache`` enforces contiguous chunk indices).
    """
    for bc in tc.network_cache.block_caches:
        bc.self_attn.reset()
    if tc.network_cache_uncond is not None:
        for bc in tc.network_cache_uncond.block_caches:
            bc.self_attn.reset()


def replay_history(
    transformer: CosmosTransformer,
    tc: CosmosTransformerCache,
    *,
    latents: list[Tensor],
    hdmaps: list[Tensor],
    context_timestep: Tensor,
    add_noise,
) -> None:
    """Rebuild the KV history by replaying chunks ``0 .. len(latents) - 1``.

    Each chunk runs one ``finalize_kv_cache``-style forward on its latent
    re-noised at the host's ``context_noise`` timestep — the exact cache
    write path of a real rollout. The eps draw is seeded per chunk index
    (:data:`CONTEXT_NOISE_SEED`), so gen and clean replays of the same
    positions consume identical noise and differ only in content.

    Args:
        transformer: The pipeline's Cosmos transformer.
        tc: AR cache, freshly :func:`reset_history`-ed.
        latents: Patchified per-chunk x0 history (branch-specific content).
        hdmaps: Patchified per-chunk HDMap conditioning (shared).
        context_timestep: 0-d context-noise timestep tensor (host config).
        add_noise: ``scheduler.add_noise`` bound from the pipeline.
    """
    assert len(latents) == len(hdmaps)
    device = latents[0].device
    for j, (lat, hd) in enumerate(zip(latents, hdmaps)):
        rng = torch.Generator(device=device).manual_seed(CONTEXT_NOISE_SEED + j)
        noisy = add_noise(lat, context_timestep, rng=rng)
        tc.start(j)
        transformer.finalize_kv_cache(
            noisy_latent=noisy, timestep=context_timestep, cache=tc, input=hd
        )
        tc.finalize(j)


def start_probe_chunk(tc: CosmosTransformerCache, *, ar_idx: int) -> None:
    """Open the probed chunk after a history replay up to ``ar_idx - 1``."""
    tc.start(ar_idx)


def finish_probe_chunk(tc: CosmosTransformerCache, *, ar_idx: int) -> None:
    """Close the ``start`` / ``finalize`` bracket after a probe sweep."""
    tc.finalize(ar_idx)


def predict_v(
    transformer: CosmosTransformer,
    tc: CosmosTransformerCache,
    *,
    hdmap: Tensor,
    z_t: Tensor,
    timestep: Tensor,
) -> Tensor:
    """Predict the velocity at a matched noisy state under the current history.

    This host is velocity/rectified-flow (``v = eps - x0``); the corrector's
    target space follows the solver, so probes and training targets live in
    v-space. ``x0_hat = z_t - sigma * v`` recovers the x0 view when needed
    (per fixed ``(z_t, t)`` the two spaces differ by the factor ``-sigma``).

    Args:
        transformer: The pipeline's Cosmos transformer.
        tc: AR cache positioned via :func:`start_probe_chunk`.
        hdmap: The probed chunk's captured (patchified) HDMap conditioning.
        z_t: Patchified noisy latent ``(1 - sigma) * x0 + sigma * eps``.
        timestep: 0-d timestep tensor in the network dtype.

    Returns:
        fp32 predicted flow, same shape as ``z_t``.
    """
    flow = transformer.predict_flow(
        noisy_latent=z_t, timestep=timestep, cache=tc, input=hdmap
    )
    return flow.float()


## Clip I/O (shared by build_pairs / gate)


def save_clip(
    path: Path,
    *,
    snaps: list[ChunkSnapshot],
    embeddings: dict,
    meta: dict,
) -> None:
    """Save a captured pair clip (CPU tensors) for the gate / training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": [s.clean_latent.cpu() for s in snaps],
            "hdmaps": [s.hdmap.cpu() for s in snaps],
            "embeddings": {
                k: (v.cpu() if isinstance(v, Tensor) else v)
                for k, v in embeddings.items()
            },
            **meta,
        },
        path,
    )


def load_clip(path: Path, device: torch.device | str, dtype: torch.dtype) -> dict:
    """Load a ``build_pairs.py`` clip; latents/hdmaps land on ``device``."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    d["latents"] = [x.to(device, dtype) for x in d["latents"]]
    d["hdmaps"] = [x.to(device, dtype) for x in d["hdmaps"]]
    return d
