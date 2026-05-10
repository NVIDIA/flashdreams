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

"""Bit-for-bit DiT parity between the legacy ``WanModel`` and FlashVSRTransformer.

Loads the frozen legacy reference at ``_wan_model_dit.py`` (sibling) and
the live :class:`FlashVSRTransformer` candidate side-by-side, loads the
same ``flashvsr_tiny_long/dit_state_dict.pt`` into both, and verifies they
produce identical chunk-by-chunk outputs under the streaming KV-cache
protocol. The legacy module is intentionally not packaged: it stays a
loose file used only as a parity reference, mirroring the
``_tcdecoder.py`` convention.

Marked ``manual`` and skipped automatically when the FlashVSR-v1.1 weight
dir is absent. Set ``$FLASHVSR_WEIGHTS_ROOT`` (default
``~/.cache/flashdreams/upsampler/weights``) to override the staging root.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from flashdreams.recipes.flashvsr.transformer import (
    FlashVSRTransformer,
    FlashVSRTransformerConfig,
)
from flashdreams.recipes.flashvsr.transformer.network import (
    FlashVSRDiTNetworkConfig,
)
from flashdreams.recipes.wan.transformer.impl.modules import (
    sinusoidal_embedding_1d,
)

_LEGACY_REF_PATH = Path(__file__).resolve().parent / "_wan_model_dit.py"
_DEFAULT_WEIGHTS_ROOT = "~/.cache/flashdreams/upsampler/weights"
_WEIGHTS_ROOT = Path(
    os.environ.get("FLASHVSR_WEIGHTS_ROOT", _DEFAULT_WEIGHTS_ROOT)
).expanduser()
_MODEL_NAME = "FlashVSR-v1.1"
_DIT_DIR = _WEIGHTS_ROOT / _MODEL_NAME / "flashvsr_tiny_long"
_DIT_CFG = _DIT_DIR / "dit_config.json"
_DIT_SD = _DIT_DIR / "dit_state_dict.pt"

_GPU_REASON = "DiT parity requires CUDA"
_WEIGHTS_REASON = (
    f"FlashVSR DiT weights not found under {_DIT_DIR}; "
    f"set $FLASHVSR_WEIGHTS_ROOT or stage with download_flashvsr_weights.sh."
)


def _load_legacy_module():
    """Load the frozen ``_wan_model_dit.py`` sibling without packaging it.

    The module imports ``flashdreams.*`` absolute paths, so the parent
    process's already-active ``flashdreams`` install is reused -- no extra
    ``sys.path`` plumbing required.
    """
    spec = importlib.util.spec_from_file_location(
        "flashvsr_legacy_wan_model_dit", _LEGACY_REF_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {_LEGACY_REF_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    target_h: int,
    target_w: int,
) -> tuple[Any, FlashVSRTransformer, dict]:
    """Build the legacy reference and the live candidate side-by-side.

    The first return value is the dynamically-loaded ``WanModel`` from
    ``_wan_model_dit.py``; it's typed ``Any`` because
    ``importlib.spec_from_file_location`` erases the concrete class -- if
    we typed it ``torch.nn.Module``, every attribute access in the body
    of :func:`test_dit_chunk_parity` would resolve through
    ``nn.Module.__getattr__`` (which returns ``Tensor | Module``) and
    require type-ignore noise at every call site.
    """
    legacy_module = _load_legacy_module()
    cfg_dict = json.loads(_DIT_CFG.read_text())
    state = torch.load(_DIT_SD, map_location="cpu")

    legacy = legacy_module.WanModel(**cfg_dict).to(device=device, dtype=dtype)
    legacy.load_state_dict(state, strict=True)
    legacy.update_parameters_after_loading_checkpoint()
    legacy = legacy.eval().requires_grad_(False)

    candidate_cfg = FlashVSRTransformerConfig(
        network=_build_network_config(cfg_dict),
        dtype=dtype,
        checkpoint_path=str(_DIT_SD),
        batch_shape=(1,),
        height=target_h // 8,
        width=target_w // 8,
        len_t=2,
        cp_size=1,
        guidance_scale=1.0,
        topk_ratio=2.0,
        kv_ratio=3,
        local_range=11,
    )
    candidate = candidate_cfg.setup(device=device)
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
    timestep = torch.tensor([1000.0], device=device, dtype=dtype)
    t = legacy.time_embedding(
        sinusoidal_embedding_1d(legacy.freq_dim, timestep).to(dtype)
    )
    t_mod = legacy.time_projection(t).unflatten(1, (6, legacy.dim))
    return t, t_mod


@pytest.mark.skipif(not _DIT_SD.exists(), reason=_WEIGHTS_REASON)
def test_dit_state_dict_shapes_match() -> None:
    """Both the legacy and candidate state dicts agree with the checkpoint shapes."""
    legacy_module = _load_legacy_module()
    cfg_dict = json.loads(_DIT_CFG.read_text())
    state = torch.load(_DIT_SD, map_location="cpu")

    legacy = legacy_module.WanModel(**cfg_dict)
    candidate_network = _build_network_config(cfg_dict).setup()

    for label, model_state in (
        ("legacy WanModel", legacy.state_dict()),
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
@pytest.mark.skipif(not _DIT_SD.exists(), reason=_WEIGHTS_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("chunks", [4])
def test_dit_chunk_parity(dtype: torch.dtype, chunks: int) -> None:
    """The legacy and candidate DiTs agree per chunk under the streaming KV protocol."""
    device = torch.device("cuda")
    latent_h, latent_w = 32, 32
    target_h, target_w = latent_h * 8, latent_w * 8

    legacy, candidate, cfg_dict = _build_models(
        dtype=dtype, device=device, target_h=target_h, target_w=target_w
    )

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
    candidate_cache = candidate.initialize_autoregressive_cache(text_embeddings=prompt)
    t, t_mod = _legacy_t_and_t_mod(legacy, dtype, device)
    timestep = torch.tensor([1000.0], device=device, dtype=dtype)

    pre_cache_k_l: list[torch.Tensor | None] = [None] * cfg_dict["num_layers"]
    pre_cache_v_l: list[torch.Tensor | None] = [None] * cfg_dict["num_layers"]

    with torch.inference_mode():
        for chunk_idx, (z, lq) in enumerate(inputs):
            out_legacy, pre_cache_k_l, pre_cache_v_l = legacy(
                x=z,
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
