# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the two-stage LingBot DiT pipeline benchmark."""

from __future__ import annotations

import pytest
import torch
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from lingbot.disagg.benchmark_pipeline import (
    _summarize,
    build_pipeline_topology,
    stack_conditioning,
    stack_encoder_outputs,
)
from lingbot.disagg.stages import LingbotConditioning
from lingbot.encoder.camctrl import I2VCamCtrlEmbeddings
from lingbot.transformer.impl.network import pipeline_partition_bounds

pytestmark = pytest.mark.ci_cpu


def test_pipeline_topology_assigns_three_pairs_and_one_spare() -> None:
    topology = build_pipeline_topology(world_size=8, sessions_per_group=2)

    assert topology.io_rank == 0
    assert topology.dit_groups == ((1, 2), (3, 4), (5, 6))
    assert topology.dit_ranks == (1, 2, 3, 4, 5, 6)
    assert topology.spare_ranks == (7,)
    assert topology.session_count == 6


@pytest.mark.parametrize(
    ("stage_index", "expected"),
    [(0, (0, 20)), (1, (20, 40))],
)
def test_pipeline_partition_splits_lingbot_layers_evenly(
    stage_index: int,
    expected: tuple[int, int],
) -> None:
    assert pipeline_partition_bounds(
        40,
        stage_index=stage_index,
        stage_count=2,
    ) == expected


def test_stack_conditioning_preserves_session_batch() -> None:
    items = [
        LingbotConditioning(
            height=58,
            width=104,
            text_embeddings=torch.full((1, 2, 3), float(index)),
        )
        for index in range(2)
    ]

    result = stack_conditioning(items)

    assert result.text_embeddings.shape == (2, 2, 3)
    assert result.text_embeddings[:, 0, 0].tolist() == [0.0, 1.0]


def test_stack_encoder_outputs_preserves_session_batch() -> None:
    items = [
        I2VCamCtrlEmbeddings(
            i2v=I2VCtrl(
                latent=torch.full((1, 3, 2, 2, 2), float(index)),
                mask=torch.ones(1, 3, 1, 2, 2),
                _is_patchified=False,
            ),
            plucker=torch.zeros(1, 3, 6, 2, 2),
            _is_patchified=False,
        )
        for index in range(2)
    ]

    result = stack_encoder_outputs(items)

    assert result.i2v.latent.shape == (2, 1, 3, 2, 2, 2)
    assert result.i2v.latent[:, 0, 0, 0, 0, 0].tolist() == [0.0, 1.0]


def test_summary_reports_required_capacity_and_pair_bandwidth() -> None:
    topology = build_pipeline_topology(world_size=8, sessions_per_group=2)
    records = [
        {
            "warmup": False,
            "output_frames": 72,
            "wave_latency_ms": 2000.0,
            "encoder_wave_ms": 20.0,
            "decoder_wave_ms": 30.0,
            "pair_fanout_ms": [1.0, 1.1, 1.2],
            "dit_group_leaders": [
                {"dit_ms": 1800.0, "finalize_ms": 100.0}
            ]
            * 3,
        }
    ]
    memory = {
        "required_capacity_gib_by_rank": [20.0, 32.0, 34.0, 32.0, 34.0, 32.0, 34.0, 0.0]
    }
    baseline = {
        "topology": {"sessions_per_wave": 7},
        "performance": {"aggregate_fps": 35.15, "per_session_fps": 5.02},
        "peak_allocated_gib_by_rank": [20.0] + [56.3] * 7,
    }

    summary = _summarize(
        records=records,
        topology=topology,
        p2p_probe={"1->2": [{"bandwidth_gbps": 300.0}]},
        memory=memory,
        baseline=baseline,
    )

    assert summary["aggregate_fps"] == pytest.approx(36.0)
    assert summary["per_session_fps"] == pytest.approx(6.0)
    assert summary["p2p_probe_gbps"]["all_pairs"]["median"] == 300.0
    assert summary["baseline"]["max_dit_peak_gib"] == 56.3
