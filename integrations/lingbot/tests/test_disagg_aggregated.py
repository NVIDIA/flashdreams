# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the aggregated LingBot benchmark report."""

from __future__ import annotations

import pytest
from lingbot.disagg.benchmark_aggregated import _summarize, _token_layout

pytestmark = pytest.mark.ci_cpu


def test_cp1_layout_accepts_tracked_464_height() -> None:
    layout = _token_layout(
        pixel_height=464,
        pixel_width=832,
        len_t=3,
        patch_size=(1, 2, 2),
        cp_size=1,
    )

    assert layout == {
        "latent_height": 58,
        "latent_width": 104,
        "tokens_per_chunk": 4524,
        "tokens_per_rank": 4524,
    }


def test_cp8_layout_uses_nearest_valid_height() -> None:
    layout = _token_layout(
        pixel_height=448,
        pixel_width=832,
        len_t=3,
        patch_size=(1, 2, 2),
        cp_size=8,
    )

    assert layout == {
        "latent_height": 56,
        "latent_width": 104,
        "tokens_per_chunk": 4368,
        "tokens_per_rank": 546,
    }


def test_cp8_layout_rejects_tracked_464_height() -> None:
    with pytest.raises(ValueError, match="4524 tokens"):
        _token_layout(
            pixel_height=464,
            pixel_width=832,
            len_t=3,
            patch_size=(1, 2, 2),
            cp_size=8,
        )


def test_aggregated_summary_uses_critical_rank_and_node_memory() -> None:
    records = [
        {
            "warmup": False,
            "output_frames": 12,
            "end_to_end_ms": 600.0,
            "critical_rank": {
                "encode_ms": 2.0,
                "diffuse_ms": 500.0,
                "decode_ms": 8.0,
                "finalize_ms": 60.0,
            },
        }
    ]
    sample = {
        "payload_bytes": float(256 * 2**20),
        "transfer_ms": 1.0,
        "bandwidth_gbps": 100.0,
    }

    summary = _summarize(
        records=records,
        tokens_per_chunk=4368,
        cp_probe={"broadcast": [sample], "all_gather": [sample]},
        peak_memory_gib_by_rank=[50.0] * 8,
        steady_memory_gib_by_rank=[49.0] * 8,
        initialization_peak_gib_by_rank=[55.0] * 8,
        comparison={
            "summary": {
                "fps": 10.0,
                "latency_ms": {"median": 1200.0},
            },
            "environment": {
                "resolution": [464, 832],
                "latent_frames_per_chunk": 3,
                "peak_memory_gib_by_rank": [25.0] * 8,
                "allocation": {"cp_size": 6},
            },
        },
    )

    assert summary["fps"] == pytest.approx(20.0)
    assert summary["token_throughput_per_second"] == pytest.approx(7280.0)
    assert summary["dit_ms"]["median"] == 500.0
    assert summary["memory"]["node_peak_gib"] == 400.0
    assert summary["memory"]["node_steady_allocated_gib"] == 392.0
    assert summary["comparison"]["topology"] == "1 encoder : CP6 DiT : 1 decoder"
    assert summary["comparison"]["latency_speedup"] == pytest.approx(2.0)
