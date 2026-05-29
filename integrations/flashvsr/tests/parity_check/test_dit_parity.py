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

"""DiT parity between upstream FlashVSR and FlashDreams.

The legacy reference is upstream's ``diffsynth.models.wan_video_dit.WanModel``
combined with the streaming forward wrapper
``diffsynth.pipelines.flashvsr_tiny_long.model_fn_wan_video`` (both inside the
parity-check sibling tree ``./FlashVSR/...``, staged by ``run.sh``). The
candidate is the live :class:`flashvsr.transformer.FlashVSRTransformer`.
Both load the same ``diffusion_pytorch_model_streaming_dmd.safetensors``
checkpoint staged from Hugging Face, and we compare chunk-by-chunk outputs
under the streaming KV-cache protocol.

``WanModel.forward`` itself is upstream's training-only path (it pre-checks
a ``_parameters_updated_after_loading_checkpoint`` flag, takes a different
kwarg list, and rebuilds RoPE freqs lazily). The streaming inference path
the upsampler actually uses is the loose function ``model_fn_wan_video``
sitting alongside the pipeline -- it consumes ``dit.patchify`` /
``dit.freqs`` / ``dit.blocks`` / ``dit.head`` / ``dit.unpatchify`` directly
and is what ``FlashVSRTinyLongPipeline`` drives per chunk. Mirrors the
upstream call site verbatim so a future refactor that lands streaming
into ``WanModel.forward`` will catch us in this test.

The TC decoder uses a different parity strategy (``importlib.util.spec_from_file_location``
on the self-contained ``examples/WanVSR/utils/TCDecoder.py``) because that
file has no relative imports; the DiT references reach back into
``diffsynth.models`` / ``diffsynth.pipelines`` so we route them through
the parity-check venv's editable ``diffsynth`` install instead.

Skipped automatically when the upstream tree (run ``bash run.sh`` next to
this file) or the FlashVSR-v1.1 weight dir is absent. Set
``$FLASHVSR_WEIGHTS_ROOT`` (default
``~/.cache/flashdreams/upsampler/weights``) to override the staging root.
``run.sh`` invokes the test from the parity-check venv where both
``diffsynth`` (upstream editable) and ``flashvsr`` (workspace editable)
are importable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

pytestmark = pytest.mark.manual

# ``diffsynth`` is upstream FlashVSR's package; it's only installed
# inside this directory's parity-check venv (``uv pip install -e ./FlashVSR``
# in ``run.sh``), not in the workspace venv that the repo-wide ty pass
# uses. Suppress the unresolved-import noise here rather than excluding
# the file globally so any *other* type errors in this test still get
# caught by ty.
from diffsynth.models.wan_video_dit import (  # ty: ignore[unresolved-import]
    WanModel,
    WanModelStateDictConverter,
    sinusoidal_embedding_1d,
)
from diffsynth.pipelines.flashvsr_tiny_long import (  # ty: ignore[unresolved-import]
    model_fn_wan_video,
)
from flashvsr.transformer import (
    FlashVSRTransformer,
    FlashVSRTransformerConfig,
)
from flashvsr.transformer.network import (
    FlashVSRDiTNetworkConfig,
)

_HERE = Path(__file__).resolve().parent
_UPSTREAM_DIT = _HERE / "FlashVSR" / "diffsynth" / "models" / "wan_video_dit.py"
_DEFAULT_WEIGHTS_ROOT = "~/.cache/flashdreams/upsampler/weights"
_WEIGHTS_ROOT = Path(
    os.environ.get("FLASHVSR_WEIGHTS_ROOT", _DEFAULT_WEIGHTS_ROOT)
).expanduser()
_MODEL_NAME = "FlashVSR-v1.1"
_DIT_SD = (
    _WEIGHTS_ROOT / _MODEL_NAME / "diffusion_pytorch_model_streaming_dmd.safetensors"
)

_GPU_REASON = "DiT parity requires CUDA"
_DIT_CHUNK_MAX_ATOL = 2.5e-1
_DIT_CHUNK_MEAN_ATOL = 3.5e-2
_UPSTREAM_REASON = (
    f"Upstream FlashVSR tree not found at {_UPSTREAM_DIT}; "
    f"run ``bash run.sh`` next to this test to clone the pinned commit."
)
_WEIGHTS_REASON = (
    f"FlashVSR DiT checkpoint not found at {_DIT_SD}; "
    f"set $FLASHVSR_WEIGHTS_ROOT or stage weights with run.sh."
)


def _load_dit_checkpoint() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load the HF safetensors DiT checkpoint in upstream ``WanModel`` layout."""
    state = load_file(_DIT_SD, device="cpu")
    converted_state, cfg_dict = WanModelStateDictConverter().from_civitai(state)
    assert cfg_dict, (
        f"Could not derive FlashVSR DiT config from {_DIT_SD}; "
        "the checkpoint layout may have changed."
    )
    return converted_state, cfg_dict


def _build_network_config(cfg_dict: dict) -> FlashVSRDiTNetworkConfig:
    """Translate the upstream DiT checkpoint config to ``FlashVSRDiTNetworkConfig``.

    The lone rename is FlashVSR's ``has_image_input`` ->
    flashdreams' ``cross_attn_enable_img``.
    """
    return FlashVSRDiTNetworkConfig(
        dim=cfg_dict["dim"],
        in_dim=cfg_dict["in_dim"],
        ffn_dim=cfg_dict["ffn_dim"],
        out_dim=cfg_dict["out_dim"],
        text_dim=cfg_dict["text_dim"],
        freq_dim=cfg_dict["freq_dim"],
        eps=cfg_dict["eps"],
        patch_size=tuple(cfg_dict["patch_size"]),
        num_heads=cfg_dict["num_heads"],
        num_layers=cfg_dict["num_layers"],
        text_len=512,
        cross_attn_norm=True,
        cross_attn_enable_img=bool(cfg_dict.get("has_image_input", False)),
        patch_embedding_type="conv3d",
    )


def _build_models(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Any, FlashVSRTransformer, dict]:
    """Build the upstream reference and the live candidate side-by-side.

    The upstream ``WanModel`` is typed ``Any`` because every attribute the
    streaming loop touches (``dim``, ``freq_dim``, ``time_embedding``,
    ``time_projection``, ``reinit_cross_kv``, ``patchify``, ``unpatchify``,
    ``blocks``, ``head``, ``freqs``) is read via ``nn.Module.__getattr__``
    and would otherwise force a type-ignore at every call site.

    Per-rollout ``(height, width)`` is not threaded through here because
    the candidate stashes them later via ``initialize_autoregressive_cache``;
    the upstream reference is resolution-agnostic at construction time too.

    Skips ``update_parameters_after_loading_checkpoint``: upstream's
    ``WanModel`` doesn't carry that hook (the candidate handles the
    equivalent rearrange via ``flashvsr.transformer.network.state_dict_transform``,
    applied at checkpoint load).
    """
    state, cfg_dict = _load_dit_checkpoint()

    legacy = WanModel(**cfg_dict).to(device=device, dtype=dtype)
    legacy.load_state_dict(state, strict=True)
    legacy = legacy.eval().requires_grad_(False)

    # ``(height, width)`` is per-rollout state that flows into
    # ``initialize_autoregressive_cache`` (see ``flashvsr/config.py``);
    # it's not a config field. ``cp_size`` is auto-detected from
    # ``torch.distributed.get_world_size()`` inside ``Wan21Transformer``.
    # Compile / cudagraph paths are disabled here to keep the comparison
    # against the eager upstream reference clean and the test fast.
    candidate_cfg = FlashVSRTransformerConfig(
        network=_build_network_config(cfg_dict),
        dtype=dtype,
        checkpoint_path=str(_DIT_SD),
        batch_shape=(1,),
        len_t=2,
        guidance_scale=1.0,
        topk_ratio=2.0,
        kv_ratio=3,
        local_range=11,
        compile_network=False,
        use_cuda_graph=False,
    )
    candidate = candidate_cfg.setup().to(device=device)
    assert isinstance(candidate, FlashVSRTransformer)
    candidate = candidate.eval().requires_grad_(False)

    return legacy, candidate, cfg_dict


def _make_chunk_inputs(
    *,
    chunks: int,
    batch: int,
    latent_h: int,
    latent_w: int,
    num_layers: int,
    dim: int,
    text_dim: int,
    text_len: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[list[tuple[torch.Tensor, list[torch.Tensor]]], torch.Tensor]:
    """Build a fake but FlashVSR-shaped sequence of upstream process chunks.

    Upstream's ``FlashVSRTinyLongPipeline`` drives one cold-start DiT call with
    6 latent frames, then steady-state calls with 2 latent frames. FlashDreams
    exposes the cold start as sequential 2-frame internal steps; the test uses
    process 0 to seed both KV caches and compares the steady-state chunks that
    follow.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    pH = latent_h // 2
    pW = latent_w // 2

    items: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
    for process_idx in range(chunks):
        latent_frames = 6 if process_idx == 0 else 2
        token_len = latent_frames * pH * pW
        z = torch.randn(batch, 16, latent_frames, latent_h, latent_w, generator=gen).to(
            device=device, dtype=dtype
        )
        lq = [
            torch.randn(batch, token_len, dim, generator=gen).to(
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ]
        items.append((z, lq))

    prompt = torch.randn(batch, text_len, text_dim, generator=gen).to(
        device=device, dtype=dtype
    )
    return items, prompt


def _legacy_t_and_t_mod(
    legacy: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute ``(t, t_mod)`` exactly like ``FlashVSRTinyLongPipeline``.

    Upstream's pipeline runs these two projections once per chunk batch
    and threads the results into ``model_fn_wan_video`` as kwargs;
    ``WanModel.forward`` would otherwise compute them internally from
    ``timestep``. We mirror the pipeline so the test stays faithful to
    the streaming path even though the candidate accepts a raw
    ``timestep``.
    """
    timestep = torch.tensor([1000.0], device=device, dtype=dtype)
    t = legacy.time_embedding(
        sinusoidal_embedding_1d(legacy.freq_dim, timestep).to(dtype)
    )
    t_mod = legacy.time_projection(t).unflatten(1, (6, legacy.dim))
    return t, t_mod


def _seed_candidate_self_attn_cache_from_legacy(
    candidate_cache: Any,
    legacy_k: list[torch.Tensor | None],
    legacy_v: list[torch.Tensor | None],
    *,
    num_heads: int,
) -> int:
    """Seed FlashDreams' self-attention KV cache from upstream cold-start K/V.

    Upstream cold start runs as one 6-latent-frame process call and returns
    three 2-frame chunks worth of K/V per block. FlashDreams' cache stores the
    same window-partitioned tensors as ``[B, tokens, heads, head_dim]``. Seeding
    those tensors directly lets the test compare the shared steady-state
    protocol without requiring the two cold-start execution strategies to have
    identical intermediate outputs.
    """
    next_idx = None
    block_caches = candidate_cache.network_cache.block_caches
    for block_idx, block_cache in enumerate(block_caches):
        k = legacy_k[block_idx]
        v = legacy_v[block_idx]
        assert k is not None and v is not None
        # Upstream: [block_n, win_size, heads * head_dim].
        # FlashDreams: [B, block_n * win_size, heads, head_dim].
        block_n, win_size, dim = k.shape
        assert dim % num_heads == 0
        k_cache = k.reshape(1, block_n * win_size, num_heads, dim // num_heads)
        v_cache = v.reshape(1, block_n * win_size, num_heads, dim // num_heads)

        self_attn = block_cache.self_attn
        cached = k_cache.shape[1]
        assert cached <= self_attn._k.shape[self_attn.seq_dim]
        self_attn._k[self_attn._seq_slice(0, cached)] = k_cache
        self_attn._v[self_attn._seq_slice(0, cached)] = v_cache
        self_attn._n_cached = cached
        self_attn._prev_chunk_idx = cached // self_attn.chunk_size - 1
        self_attn._curr_chunk_idx = None
        if next_idx is None:
            next_idx = self_attn._prev_chunk_idx + 1
        else:
            assert next_idx == self_attn._prev_chunk_idx + 1
    assert next_idx is not None
    return next_idx


@pytest.mark.skipif(not _UPSTREAM_DIT.exists(), reason=_UPSTREAM_REASON)
@pytest.mark.skipif(not _DIT_SD.exists(), reason=_WEIGHTS_REASON)
def test_dit_state_dict_shapes_match() -> None:
    """Both the upstream and candidate state dicts agree with the checkpoint shapes."""
    state, cfg_dict = _load_dit_checkpoint()

    legacy = WanModel(**cfg_dict)
    candidate_network = _build_network_config(cfg_dict).setup()

    for label, model_state in (
        ("upstream WanModel", legacy.state_dict()),
        ("FlashVSRDiTNetwork", candidate_network.state_dict()),
    ):
        missing = sorted(k for k in model_state if k not in state)
        unexpected = sorted(k for k in state if k not in model_state)
        mismatched = sorted(
            k
            for k in model_state.keys() & state.keys()
            if tuple(model_state[k].shape) != tuple(state[k].shape)
        )
        assert not missing, f"{label}: missing keys vs checkpoint: {missing[:8]}"
        assert not unexpected, f"{label}: unexpected keys: {unexpected[:8]}"
        assert not mismatched, f"{label}: shape mismatches: {mismatched[:8]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.skipif(not _UPSTREAM_DIT.exists(), reason=_UPSTREAM_REASON)
@pytest.mark.skipif(not _DIT_SD.exists(), reason=_WEIGHTS_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("chunks", [4])
def test_dit_chunk_parity(dtype: torch.dtype, chunks: int) -> None:
    """The upstream and candidate DiTs agree per chunk under the streaming KV protocol."""
    device = torch.device("cuda")
    latent_h, latent_w = 32, 32

    legacy, candidate, cfg_dict = _build_models(dtype=dtype, device=device)

    inputs, prompt = _make_chunk_inputs(
        chunks=chunks,
        batch=1,
        latent_h=latent_h,
        latent_w=latent_w,
        num_layers=cfg_dict["num_layers"],
        dim=cfg_dict["dim"],
        text_dim=cfg_dict["text_dim"],
        text_len=512,
        device=device,
        dtype=dtype,
        seed=1234,
    )

    legacy.reinit_cross_kv(prompt)
    candidate_cache = candidate.initialize_autoregressive_cache(
        height=latent_h,
        width=latent_w,
        text_embeddings=prompt,
    )
    t, t_mod = _legacy_t_and_t_mod(legacy, dtype, device)
    timestep = torch.tensor([1000.0], device=device, dtype=dtype)

    pre_cache_k_l: list[torch.Tensor | None] = [None] * cfg_dict["num_layers"]
    pre_cache_v_l: list[torch.Tensor | None] = [None] * cfg_dict["num_layers"]

    with torch.inference_mode():
        internal_ar_idx = 0
        for process_idx, (z, lq) in enumerate(inputs):
            # ``model_fn_wan_video`` is the loose streaming-forward wrapper
            # ``FlashVSRTinyLongPipeline`` invokes per chunk -- it bypasses
            # ``WanModel.forward`` (training-only upstream) and drives
            # ``patchify`` / ``blocks`` / ``head`` / ``unpatchify`` directly,
            # threading the same ``pre_cache_k`` / ``pre_cache_v`` list-of-
            # tensors KV cache that the candidate maintains in
            # ``Wan21TransformerCache``. The ``timestep`` kwarg here is
            # vestigial in the wrapper body (the function never reads it
            # because ``t`` / ``t_mod`` are pre-computed by the caller),
            # but we pass it to match the upstream signature so a future
            # refactor that wires it through will not silently break this
            # test.
            out_legacy, pre_cache_k_l, pre_cache_v_l = model_fn_wan_video(
                legacy,
                x=z,
                timestep=timestep,
                context=None,
                LQ_latents=lq,
                is_stream=True,
                pre_cache_k=pre_cache_k_l,
                pre_cache_v=pre_cache_v_l,
                topk_ratio=2.0,
                kv_ratio=3.0,
                cur_process_idx=process_idx,
                t_mod=t_mod,
                t=t,
                local_range=11,
            )

            if process_idx == 0:
                internal_ar_idx = _seed_candidate_self_attn_cache_from_legacy(
                    candidate_cache,
                    pre_cache_k_l,
                    pre_cache_v_l,
                    num_heads=cfg_dict["num_heads"],
                )
                continue

            # ``cache.start`` / ``cache.finalize`` bracket each AR step --
            # ``start`` calls ``before_update`` on every BlockKVCache and
            # ``finalize`` calls ``after_update``. ``FlashVSRPipeline.generate``
            # drives the pair via ``cache.transformer_cache.start(...)`` /
            # ``...finalize(...)`` (per-iter for non-final iters) plus the
            # framework's ``DiffusionModel.finalize`` -> ``cache.finalize(...)``
            # for the final iter. Here we drive both ends manually since we
            # bypass the pipeline; without ``finalize`` the next chunk's
            # ``before_update`` finds ``_curr_chunk_idx`` still set from the
            # previous chunk and trips the "Must call after_update() before
            # before_update()" assertion in :class:`BlockKVCache`.
            candidate_parts: list[torch.Tensor] = []
            latent_frames = z.shape[2]
            assert latent_frames % 2 == 0
            for iter_idx in range(latent_frames // 2):
                candidate_cache.start(autoregressive_index=internal_ar_idx)
                z_slice = z[:, :, iter_idx * 2 : (iter_idx + 1) * 2]
                z_patched = candidate.patchify_and_maybe_split_cp(
                    z_slice.transpose(1, 2)
                )
                token_start = iter_idx * 2 * (latent_h // 2) * (latent_w // 2)
                token_end = (iter_idx + 1) * 2 * (latent_h // 2) * (latent_w // 2)
                lq_slice = [layer[:, token_start:token_end, :] for layer in lq]
                flow = candidate.predict_flow(
                    noisy_latent=z_patched,
                    timestep=timestep,
                    cache=candidate_cache,
                    input=lq_slice,
                )
                candidate_parts.append(
                    candidate.unpatchify_and_maybe_gather_cp(flow).transpose(1, 2)
                )
                candidate_cache.finalize(autoregressive_index=internal_ar_idx)
                internal_ar_idx += 1
            out_candidate = torch.cat(candidate_parts, dim=2)

            # The live FlashDreams path intentionally uses optimized numerics
            # that are not bit-close to upstream's reference path:
            # upstream rotates RoPE in fp64 and calls the public
            # ``block_sparse_attn_func`` wrapper, while the candidate uses the
            # fused RoPE kernel and the direct sparse-attention function path.
            # Assert a calibrated bf16 envelope that still catches gross
            # protocol / checkpoint / cache regressions.
            diff = (out_legacy - out_candidate).float().abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            assert (
                max_abs <= _DIT_CHUNK_MAX_ATOL and mean_abs <= _DIT_CHUNK_MEAN_ATOL
            ), (
                f"process chunk {process_idx} parity failed: "
                f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
                f"(limits: max_abs<={_DIT_CHUNK_MAX_ATOL:.6g}, "
                f"mean_abs<={_DIT_CHUNK_MEAN_ATOL:.6g})"
            )
