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
Both load the same ``flashvsr_tiny_long/dit_state_dict.pt`` and we compare
chunk-by-chunk outputs under the streaming KV-cache protocol.

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

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

# ``diffsynth`` is upstream FlashVSR's package; it's only installed
# inside this directory's parity-check venv (``uv pip install -e ./FlashVSR``
# in ``run.sh``), not in the workspace venv that the repo-wide ty pass
# uses. Suppress the unresolved-import noise here rather than excluding
# the file globally so any *other* type errors in this test still get
# caught by ty.
from diffsynth.models.wan_video_dit import (  # ty: ignore[unresolved-import]
    WanModel,
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
_DIT_DIR = _WEIGHTS_ROOT / _MODEL_NAME / "flashvsr_tiny_long"
_DIT_CFG = _DIT_DIR / "dit_config.json"
_DIT_SD = _DIT_DIR / "dit_state_dict.pt"

_GPU_REASON = "DiT parity requires CUDA"
_UPSTREAM_REASON = (
    f"Upstream FlashVSR tree not found at {_UPSTREAM_DIT}; "
    f"run ``bash run.sh`` next to this test to clone the pinned commit."
)
_WEIGHTS_REASON = (
    f"FlashVSR DiT weights not found under {_DIT_DIR}; "
    f"set $FLASHVSR_WEIGHTS_ROOT or stage with download_flashvsr_weights.sh."
)


def _build_network_config(cfg_dict: dict) -> FlashVSRDiTNetworkConfig:
    """Translate ``flashvsr_tiny_long/dit_config.json`` to ``FlashVSRDiTNetworkConfig``.

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
    cfg_dict = json.loads(_DIT_CFG.read_text())
    state = torch.load(_DIT_SD, map_location="cpu")

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
    """Build a fake but FlashVSR-shaped sequence of (latent, LQ_latents) chunks.

    Each chunk is a 2-latent-frame slice (``len_t=2``); ``LQ_latents[i]`` is
    the LR-projector output for block ``i`` and has shape
    ``[B, len_t * pH * pW, dim]``.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    pH = latent_h // 2
    pW = latent_w // 2
    L = 2 * pH * pW

    items: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
    for _ in range(chunks):
        z = torch.randn(batch, 16, 2, latent_h, latent_w, generator=gen).to(
            device=device, dtype=dtype
        )
        lq = [
            torch.randn(batch, L, dim, generator=gen).to(device=device, dtype=dtype)
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


@pytest.mark.skipif(not _UPSTREAM_DIT.exists(), reason=_UPSTREAM_REASON)
@pytest.mark.skipif(not _DIT_SD.exists(), reason=_WEIGHTS_REASON)
def test_dit_state_dict_shapes_match() -> None:
    """Both the upstream and candidate state dicts agree with the checkpoint shapes."""
    cfg_dict = json.loads(_DIT_CFG.read_text())
    state = torch.load(_DIT_SD, map_location="cpu")

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
        for chunk_idx, (z, lq) in enumerate(inputs):
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
                cur_process_idx=chunk_idx,
                t_mod=t_mod,
                t=t,
                local_range=11,
            )

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
            candidate_cache.start(autoregressive_index=chunk_idx)
            z_patched = candidate.patchify_and_maybe_split_cp(z.transpose(1, 2))
            flow = candidate.predict_flow(
                noisy_latent=z_patched,
                timestep=timestep,
                cache=candidate_cache,
                input=lq,
            )
            out_candidate = candidate.unpatchify_and_maybe_gather_cp(flow).transpose(
                1, 2
            )
            candidate_cache.finalize(autoregressive_index=chunk_idx)

            # Tolerance retained from the previous ``_wan_model_dit.py``
            # frozen-reference parity. Upstream rotates RoPE in fp64 then
            # casts back; the candidate stays in compute dtype (bf16
            # here). On a DGX H100 box this nudge sits well inside the
            # ``1e-3`` envelope; if a future hardware / cuDNN combo
            # grazes it, calibrate empirically and document the new
            # bound in this comment (mirroring ``test_tcdecoder_parity``).
            diff = (out_legacy - out_candidate).float().abs()
            assert torch.allclose(
                out_legacy.float(),
                out_candidate.float(),
                atol=1e-3,
                rtol=1e-3,
            ), (
                f"chunk {chunk_idx} parity failed: "
                f"max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}"
            )
