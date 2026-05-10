# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alignment checks between the FlashDreams QVG port and official QVG kernels.

These tests are skipped by default. Set ``QVG_OFFICIAL_REPO`` to the cloned
Quant-VideoGen repo and run on CUDA to compare ported pieces against the
official Triton implementation.
"""

from __future__ import annotations

import os
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest
import torch

from flashdreams.core.attention.kv_compress import (
    KVCompressionConfig,
    KVSpan,
    QVGBackend,
    QVGQuantConfig,
    RuntimePhase,
)
from flashdreams.core.attention.kv_compress.qvg import (
    _batched_kmeans,
    _dequantize_residual,
    _gather_centroids,
    _prq_dequantize_tensor,
    _quantize_residual,
    _unpack_signed,
)
from flashdreams.core.attention.kv_compress.qvg_official import (
    official_prq_dequantize_tensor,
    official_prq_quantize_tensor,
)


def _official_repo() -> Path:
    repo = os.getenv("QVG_OFFICIAL_REPO")
    if repo is None:
        pytest.skip("QVG_OFFICIAL_REPO is not set")
    path = Path(repo)
    if not (path / "quant_videogen").is_dir():
        pytest.skip(f"QVG_OFFICIAL_REPO does not look like Quant-VideoGen: {path}")
    _install_official_namespace(path)
    return path


def _install_official_namespace(path: Path) -> None:
    """Import official submodules without executing quant_videogen/__init__.py."""
    for module_name in list(sys.modules):
        if module_name == "quant_videogen" or module_name.startswith("quant_videogen."):
            del sys.modules[module_name]

    root = path / "quant_videogen"
    for name, package_path in {
        "quant_videogen": root,
        "quant_videogen.real": root / "real",
        "quant_videogen.kmeans": root / "kmeans",
    }.items():
        module = types.ModuleType(name)
        module.__path__ = [str(package_path)]
        module.__package__ = name
        module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module


def _requires_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("official QVG alignment checks require CUDA")
    return torch.device("cuda")


def _official_scale_dtype(device: torch.device) -> torch.dtype:
    """Use official FP8 scale path only on GPUs where Triton supports it."""
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 9 and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    return torch.bfloat16


def _scale_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    if dtype == getattr(torch, "float8_e4m3fn", None):
        return "float8_e4m3fn"
    raise AssertionError(f"unexpected dtype: {dtype}")


def _assert_prq_state_close(
    actual: dict[str, object],
    expected: dict[str, object],
) -> None:
    for key in ("residual_quant", "scales"):
        torch.testing.assert_close(actual[key], expected[key])
    for key in ("centroids_list", "cluster_ids_list"):
        actual_list = actual[key]
        expected_list = expected[key]
        assert len(actual_list) == len(expected_list)
        for actual_tensor, expected_tensor in zip(
            actual_list, expected_list, strict=True
        ):
            torch.testing.assert_close(actual_tensor, expected_tensor)


def _assert_packed_residual_aligned(
    packed: torch.Tensor,
    official_packed: torch.Tensor,
    *,
    residual: torch.Tensor,
    block_size: int,
    num_bits: int,
) -> None:
    """Require exact pack match except official Triton exact-half edge cases."""
    if torch.equal(packed, official_packed):
        return

    local_q = _unpack_signed(packed, num_bits=num_bits, D=residual.shape[-1])
    official_q = _unpack_signed(
        official_packed, num_bits=num_bits, D=residual.shape[-1]
    )
    q_mismatch = local_q != official_q
    max_int = 2 ** (num_bits - 1) - 1
    chunks = residual.float().reshape(*residual.shape[:-1], -1, block_size)
    scales = chunks.abs().amax(dim=-1).clamp_min(1e-10) / max_int
    ratios = (chunks / scales.unsqueeze(-1)).reshape_as(residual)
    exact_half = torch.frac(ratios.abs()) == 0.5

    non_half_mismatch = q_mismatch & ~exact_half
    assert not non_half_mismatch.any(), (
        "packed residual differs from official outside exact-half rounding cases"
    )
    assert q_mismatch.sum().item() <= 2, (
        "too many exact-half rounding differences against official QVG"
    )


def test_qvg_residual_pack_matches_official_triton() -> None:
    """Check residual packing, scale layout, and scale dtype."""
    _official_repo()
    device = _requires_cuda()
    from quant_videogen.real.quant_pack import quant_pack

    torch.manual_seed(123)
    scale_dtype = _official_scale_dtype(device)
    residual = torch.randn(1, 2, 17, 128, device=device, dtype=torch.bfloat16)

    official_packed, official_scales = quant_pack(
        residual,
        block_size=64,
        num_bits=2,
        scale_precision=scale_dtype,
        pack_output_int8=True,
    )
    packed, scales = _quantize_residual(
        residual.float(),
        block_size=64,
        num_bits=2,
        scale_dtype=scale_dtype,
    )

    _assert_packed_residual_aligned(
        packed,
        official_packed,
        residual=residual,
        block_size=64,
        num_bits=2,
    )
    torch.testing.assert_close(scales, official_scales)


def test_qvg_dequant_matches_official_accumulate_for_same_state() -> None:
    """Check packed residual unpacking and centroid accumulation order."""
    _official_repo()
    device = _requires_cuda()
    from quant_videogen.real.prq import prq_dequant

    torch.manual_seed(234)
    B, H, S, D = 1, 2, 19, 128
    num_clusters = 16
    centroids = torch.randn(
        B, H, num_clusters, D, device=device, dtype=torch.bfloat16
    )
    cluster_ids = torch.randint(
        0, num_clusters, (B, H, S), device=device, dtype=torch.uint8
    )
    scale_dtype = _official_scale_dtype(device)
    residual = torch.randn(B, H, S, D, device=device, dtype=torch.float32) * 0.2
    residual_quant, scales = _quantize_residual(
        residual,
        block_size=64,
        num_bits=2,
        scale_dtype=scale_dtype,
    )

    official = prq_dequant(
        centroids_list=[centroids],
        cluster_ids_list=[cluster_ids],
        residual_quant=residual_quant,
        scales=scales,
        block_size=64,
        num_bits=2,
        PACK_INPUT_INT8=True,
        CLUSTER_ID_INT8=True,
        output_dtype=torch.bfloat16,
    )
    ported = _prq_dequantize_tensor(
        {
            "shape": (B, H, S, D),
            "centroids_list": [centroids],
            "cluster_ids_list": [cluster_ids],
            "residual_quant": residual_quant,
            "scales": scales,
            "block_size": 64,
            "num_bits": 2,
        },
        output_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(ported, official)


def test_qvg_backend_official_triton_matches_direct_prq_and_preserves_rng() -> None:
    """Check backend official_triton path calls direct QVG kernels deterministically."""
    _official_repo()
    device = _requires_cuda()
    scale_dtype = _official_scale_dtype(device)
    scale_dtype_name = _scale_dtype_name(scale_dtype)

    torch.manual_seed(510)
    k = torch.randn(1, 2, 65, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 65, 128, device=device, dtype=torch.bfloat16)
    config = KVCompressionConfig(
        backend="qvg",
        backend_config={
            "quant_type": "triton-nstages-kmeans-int2",
            "cache_num_k_centroids": 16,
            "cache_num_v_centroids": 16,
            "kmeans_max_iters": 2,
            "quant_block_size": 64,
            "num_prq_stages": 1,
            "scale_dtype": scale_dtype_name,
            "kmeans_seed": 777,
            "kernel_impl": "official_triton",
        },
    )
    backend = QVGBackend(QVGQuantConfig.from_config(config))

    expected_k = official_prq_quantize_tensor(
        k,
        num_stages=1,
        num_clusters=16,
        max_iters=2,
        block_size=64,
        num_bits=2,
        scale_dtype=scale_dtype,
        kmeans_seed=777,
    )
    expected_v = official_prq_quantize_tensor(
        v,
        num_stages=1,
        num_clusters=16,
        max_iters=2,
        block_size=64,
        num_bits=2,
        scale_dtype=scale_dtype,
        kmeans_seed=778,
    )

    torch.manual_seed(999)
    expected_random = torch.rand(4, device=device)
    torch.manual_seed(999)
    payload = backend.compress_span(
        k,
        v,
        span=KVSpan(0, 65),
        phase=RuntimePhase.FINALIZE_CLEAN_KV,
        config=config,
    )
    actual_random = torch.rand(4, device=device)

    _assert_prq_state_close(payload.k, expected_k)
    _assert_prq_state_close(payload.v, expected_v)
    torch.testing.assert_close(actual_random, expected_random)
    out_k, out_v = backend.decompress_span(payload, phase=RuntimePhase.DENOISE)
    direct_k = official_prq_dequantize_tensor(expected_k, output_dtype=k.dtype)
    direct_v = official_prq_dequantize_tensor(expected_v, output_dtype=v.dtype)
    torch.testing.assert_close(out_k, direct_k)
    torch.testing.assert_close(out_v, direct_v)


def test_qvg_kmeans_matches_official_with_same_initial_centroids() -> None:
    """Check k-means assignment/update behavior with controlled init."""
    _official_repo()
    device = _requires_cuda()
    from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid

    torch.manual_seed(345)
    B, H, S, D = 1, 2, 257, 32
    num_clusters = 16
    x = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    x_flat = x.reshape(B * H, S, D).contiguous()
    init_idx = torch.linspace(0, S - 1, num_clusters, device=device).round().long()
    init_idx = init_idx.expand(B * H, num_clusters)
    init_centroids = torch.gather(
        x_flat,
        dim=1,
        index=init_idx.unsqueeze(-1).expand(-1, -1, D),
    ).clone()

    official_ids, official_centroids, _, _ = batch_kmeans_Euclid(
        x_flat,
        n_clusters=num_clusters,
        max_iters=3,
        init_centroids=init_centroids,
    )
    ported_ids, ported_centroids = _batched_kmeans(
        x,
        num_clusters=num_clusters,
        max_iters=3,
        init="linspace",
        seed=None,
    )

    torch.testing.assert_close(ported_ids.reshape(B * H, S), official_ids)
    torch.testing.assert_close(
        ported_centroids.reshape(B * H, num_clusters, D),
        official_centroids,
        atol=1e-5,
        rtol=1e-5,
    )


def test_qvg_kmeans_bfloat16_first_assignment_matches_official() -> None:
    """Check dtype-sensitive BF16 assignment before atomic update drift."""
    _official_repo()
    device = _requires_cuda()
    from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid

    torch.manual_seed(346)
    B, H, S, D = 1, 2, 257, 32
    num_clusters = 16
    x = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    x_flat = x.reshape(B * H, S, D).contiguous()
    init_idx = torch.linspace(0, S - 1, num_clusters, device=device).round().long()
    init_idx = init_idx.expand(B * H, num_clusters)
    init_centroids = torch.gather(
        x_flat,
        dim=1,
        index=init_idx.unsqueeze(-1).expand(-1, -1, D),
    ).clone()

    official_ids, official_centroids, _, _ = batch_kmeans_Euclid(
        x_flat,
        n_clusters=num_clusters,
        max_iters=1,
        init_centroids=init_centroids,
    )
    ported_ids, ported_centroids = _batched_kmeans(
        x,
        num_clusters=num_clusters,
        max_iters=1,
        init="linspace",
        seed=None,
    )

    ported_ids = ported_ids.reshape(B * H, S)
    mismatch_rate = (ported_ids != official_ids).float().mean().item()
    assert mismatch_rate <= 0.01
    assert ported_centroids.dtype == official_centroids.dtype


def test_qvg_residual_reconstruction_matches_direct_math() -> None:
    """Cheap non-official check that dequant and gather compose correctly."""
    torch.manual_seed(456)
    B, H, S, D = 1, 2, 13, 16
    num_clusters = 4
    residual = torch.randn(B, H, S, D, dtype=torch.float32)
    centroids = torch.randn(B, H, num_clusters, D, dtype=torch.float32)
    cluster_ids = torch.randint(0, num_clusters, (B, H, S), dtype=torch.uint8)
    residual_quant, scales = _quantize_residual(
        residual,
        block_size=4,
        num_bits=4,
        scale_dtype=torch.float32,
    )

    expected = _dequantize_residual(
        residual_quant,
        scales,
        block_size=4,
        num_bits=4,
        D=D,
    ) + _gather_centroids(cluster_ids.long(), centroids)
    ported = _prq_dequantize_tensor(
        {
            "shape": (B, H, S, D),
            "centroids_list": [centroids],
            "cluster_ids_list": [cluster_ids],
            "residual_quant": residual_quant,
            "scales": scales,
            "block_size": 4,
            "num_bits": 4,
        },
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(ported, expected)
