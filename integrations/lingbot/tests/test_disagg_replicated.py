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

"""CPU tests for replicated-stage allocation and benchmark summaries."""

from __future__ import annotations

from typing import Any

import pytest
from flashdreams.infra.transfer import TransferStats
from lingbot.disagg.benchmark_replicated import (
    StageAllocation,
    _summarize,
    allocate_stage_replicas,
    allocation_from_baseline,
)

pytestmark = pytest.mark.ci_cpu


def _baseline() -> dict[str, Any]:
    return {
        "summary": {
            "fps": 6.0,
            "latency_ms": {"median": 2000.0},
            "encoder_ms": {"median": 1.0},
            "encoder_to_dit": {"handoff_ms": {"median": 25.0}},
            "dit_ms": {"median": 1700.0},
            "finalize_ms": {"median": 450.0},
            "decoder_ms": {"median": 7.0},
            "dit_to_decoder": {"handoff_ms": {"median": 12.0}},
        }
    }


def test_dit_dominated_eight_gpu_allocation_is_one_six_one() -> None:
    allocation = allocation_from_baseline(_baseline(), total_gpus=8)
    assert allocation == StageAllocation(
        encoder_replicas=1,
        dit_replicas=6,
        decoder_replicas=1,
    )
    assert allocation.total_gpus == 8


def test_allocation_requires_one_positive_service_time_per_stage() -> None:
    with pytest.raises(ValueError, match="At least three GPUs"):
        allocate_stage_replicas(
            total_gpus=2,
            encoder_service_ms=1.0,
            dit_service_ms=1.0,
            decoder_service_ms=1.0,
        )
    with pytest.raises(ValueError, match="must be positive"):
        allocate_stage_replicas(
            total_gpus=3,
            encoder_service_ms=1.0,
            dit_service_ms=0.0,
            decoder_service_ms=1.0,
        )


def test_replicated_summary_reports_aggregate_and_gpu_normalized_scaling() -> None:
    transfer = {
        "payload_bytes": 1024,
        "transfer_ms": 1.0,
        "handoff_ms": 2.0,
    }
    records = [
        {
            "warmup": False,
            "output_frames": 72,
            "wave_latency_ms": 2400.0,
            "encoder_wave_ms": 10.0,
            "decoder_wave_ms": 20.0,
            "dit_workers": [{"dit_ms": 1900.0, "finalize_ms": 200.0} for _ in range(6)],
            "encoder_to_dit": [transfer.copy() for _ in range(6)],
            "dit_to_decoder": [transfer.copy() for _ in range(6)],
        }
    ]
    probe = TransferStats(
        backend="mooncake-rdma",
        payload_bytes=256 * 2**20,
        registration_ms=0.0,
        transfer_ms=6.4,
        bandwidth_gbps=41.0,
    )

    summary = _summarize(
        records=records,
        probes={"encoder_to_dit_1": [probe]},
        baseline=_baseline(),
        dit_replicas=6,
    )

    assert summary["aggregate_fps"] == pytest.approx(30.0)
    assert summary["per_session_fps"] == pytest.approx(5.0)
    assert summary["throughput_speedup"] == pytest.approx(5.0)
    assert summary["gpu_normalized_speedup"] == pytest.approx(1.875)
    assert summary["dit_critical_path_ms"]["median"] == pytest.approx(2100.0)
