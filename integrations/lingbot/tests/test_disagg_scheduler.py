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

"""CPU tests for LingBot session-affine scheduling."""

from __future__ import annotations

import pytest
from lingbot.disagg.scheduler import (
    ServiceClass,
    SessionAwareScheduler,
    SessionRequest,
    WorkerSnapshot,
    build_microbatches,
)

pytestmark = pytest.mark.ci_cpu


def _worker(
    worker_id: str,
    *,
    pool: str = "io-plus-7-dit",
    queue_depth: int = 0,
    predicted_chunk_ms: float = 400.0,
    rack: str = "rack-a",
    nic: str = "mlx5_0",
    rdma_capable: bool = True,
) -> WorkerSnapshot:
    return WorkerSnapshot(
        worker_id=worker_id,
        pool=pool,
        queue_depth=queue_depth,
        predicted_chunk_ms=predicted_chunk_ms,
        free_hbm_gib=20.0,
        supported_shapes=frozenset({(12, 52, 104)}),
        supported_cp_sizes=frozenset({1}),
        rack=rack,
        nic=nic,
        rdma_capable=rdma_capable,
    )


def test_scheduler_uses_predicted_wait_and_preserves_session_affinity() -> None:
    scheduler = SessionAwareScheduler(min_free_hbm_gib=4.0)
    request = SessionRequest(
        session_id="session-a",
        shape=(12, 52, 104),
        cp_size=1,
    )
    workers = (
        _worker("slow", queue_depth=0, predicted_chunk_ms=800.0),
        _worker("busy-fast", queue_depth=1, predicted_chunk_ms=300.0),
    )

    assert scheduler.assign(request, workers) == "busy-fast"
    changed = (
        _worker("slow", queue_depth=0, predicted_chunk_ms=100.0),
        _worker("busy-fast", queue_depth=9, predicted_chunk_ms=300.0),
    )
    assert scheduler.assign(request, changed) == "busy-fast"


def test_scheduler_selects_latency_pool_and_rejects_tcp_fallback() -> None:
    scheduler = SessionAwareScheduler()
    latency = SessionRequest(
        session_id="premium",
        shape=(12, 52, 104),
        cp_size=1,
        service_class=ServiceClass.LATENCY,
    )

    with pytest.raises(RuntimeError, match="No compatible"):
        scheduler.assign(
            latency,
            (
                _worker("throughput"),
                _worker(
                    "latency-tcp",
                    pool="aggregated-cp8",
                    rdma_capable=False,
                ),
            ),
        )


def test_microbatches_only_group_compatible_sessions() -> None:
    assignments = [
        (
            SessionRequest(session_id=f"s{index}", shape=(12, 52, 104), cp_size=1),
            "dit-0",
        )
        for index in range(3)
    ]
    assignments.append(
        (
            SessionRequest(session_id="other", shape=(12, 60, 104), cp_size=1),
            "dit-0",
        )
    )

    batches = build_microbatches(assignments, max_batch_size=2)

    assert [batch.session_ids for batch in batches] == [
        ("s0", "s1"),
        ("s2",),
        ("other",),
    ]
