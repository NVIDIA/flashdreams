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

"""CPU tests for DiT-only context-parallel disaggregation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from lingbot.disagg.benchmark_cp import _summarize
from lingbot.transformer import LingbotWorldTransformer
from torch.distributed import ProcessGroup

from flashdreams.infra.transfer import TransferStats

pytestmark = pytest.mark.ci_cpu


class _FakeGroup:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _FakeNetwork:
    def __init__(self) -> None:
        self.group: ProcessGroup | None = None

    def set_context_parallel_group(self, group: ProcessGroup | None) -> None:
        self.group = group


class _FakeDispatch:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def test_lingbot_transformer_can_bind_dit_only_cp_group_before_cache() -> None:
    group = cast(ProcessGroup, _FakeGroup(6))
    network = _FakeNetwork()
    dispatch = _FakeDispatch()
    transformer = SimpleNamespace(
        _output_height=None,
        _output_width=None,
        _cp_group=None,
        _cp_size=1,
        network=network,
        _use_cuda_graph=True,
        _cuda_graph_dispatch=dispatch,
    )

    LingbotWorldTransformer.set_context_parallel_group(
        cast(LingbotWorldTransformer, transformer),
        group,
    )

    assert transformer._cp_group is group
    assert transformer._cp_size == 6
    assert network.group is group
    assert dispatch.reset_count == 1


def test_lingbot_transformer_rejects_cp_rebind_after_cache_init() -> None:
    transformer = SimpleNamespace(_output_height=58, _output_width=104)
    with pytest.raises(RuntimeError, match="before initializing"):
        LingbotWorldTransformer.set_context_parallel_group(
            cast(LingbotWorldTransformer, transformer),
            cast(ProcessGroup, _FakeGroup(6)),
        )


def _baseline() -> dict[str, Any]:
    return {
        "summary": {
            "fps": 6.0,
            "latency_ms": {"median": 2000.0},
            "dit_ms": {"median": 1600.0},
            "finalize_ms": {"median": 400.0},
        }
    }


def test_cp_summary_reports_single_session_latency_scaling() -> None:
    transfer = {
        "payload_bytes": 1024,
        "transfer_ms": 1.0,
        "handoff_ms": 2.0,
    }
    records = [
        {
            "warmup": False,
            "output_frames": 12,
            "end_to_end_ms": 500.0,
            "encoder_ms": 1.0,
            "encoder_to_cp_leader": transfer,
            "encoder_to_cp_leader_handoff_ms": 2.0,
            "cp_input_fanout_ms": 3.0,
            "cp_workers": [{"dit_ms": 300.0, "finalize_ms": 50.0} for _ in range(6)],
            "cp_leader_to_decoder": transfer,
            "cp_leader_to_decoder_handoff_ms": 2.0,
            "decoder_ms": 7.0,
        }
    ]
    mooncake_sample = TransferStats(
        backend="mooncake-rdma",
        payload_bytes=256 * 2**20,
        registration_ms=0.0,
        transfer_ms=6.4,
        bandwidth_gbps=41.0,
    )
    cp_sample = {
        "payload_bytes": float(256 * 2**20),
        "transfer_ms": 2.0,
        "bandwidth_gbps": 100.0,
    }

    summary = _summarize(
        records=records,
        mooncake_probe={
            "encoder_to_cp_leader": [mooncake_sample],
            "cp_leader_to_decoder": [mooncake_sample],
        },
        cp_probe={"broadcast": [cp_sample], "all_gather": [cp_sample]},
        baseline=_baseline(),
        cp_size=6,
    )

    assert summary["fps"] == pytest.approx(24.0)
    assert summary["latency_speedup"] == pytest.approx(4.0)
    assert summary["fps_speedup"] == pytest.approx(4.0)
    assert summary["dit_speedup"] == pytest.approx(2000.0 / 350.0)
    assert summary["cp_efficiency"] == pytest.approx((2000.0 / 350.0) / 6)
