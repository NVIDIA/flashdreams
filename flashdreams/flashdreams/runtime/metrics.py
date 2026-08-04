# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime metrics boundary for inference sessions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from flashdreams.runtime._utils import freeze_mapping


@dataclass(frozen=True, kw_only=True, slots=True)
class RuntimeMetricSample:
    """One runtime metric sample.

    Timing samples should use seconds as their canonical unit.
    """

    __hash__ = None

    name: str
    value: float | int
    unit: str = "s"
    step_index: int | None = None
    category: str = "runtime"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("RuntimeMetricSample.name must be non-empty.")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("RuntimeMetricSample.value must be numeric.")
        if not math.isfinite(float(self.value)):
            raise ValueError("RuntimeMetricSample.value must be finite.")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("RuntimeMetricSample.step_index must be >= 0.")
        if not self.unit.strip():
            raise ValueError("RuntimeMetricSample.unit must be non-empty.")
        if self.category == "timing" and self.unit != "s":
            raise ValueError("Timing metric samples must use unit='s'.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@runtime_checkable
class MetricsRecorder(Protocol):
    """Collector for runtime metrics."""

    def record(self, sample: RuntimeMetricSample) -> None:
        """Record one metric sample."""
        ...

    def record_timing(
        self,
        name: str,
        duration_s: float,
        *,
        step_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one timing sample in seconds."""
        ...

    def close(self) -> None:
        """Finalize metric collection."""
        ...


@dataclass(slots=True)
class InMemoryMetricsRecorder:
    """Simple metrics recorder useful for tests, smoke runs, and adapters."""

    samples: list[RuntimeMetricSample] = field(default_factory=list)
    closed: bool = False

    def record(self, sample: RuntimeMetricSample) -> None:
        if self.closed:
            raise RuntimeError("Cannot record metrics after close().")
        self.samples.append(sample)

    def record_timing(
        self,
        name: str,
        duration_s: float,
        *,
        step_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.record(
            RuntimeMetricSample(
                name=name,
                value=duration_s,
                unit="s",
                step_index=step_index,
                category="timing",
                metadata={} if metadata is None else metadata,
            )
        )

    def close(self) -> None:
        self.closed = True


class NullMetricsRecorder:
    """Metrics recorder that intentionally drops all samples."""

    def record(self, sample: RuntimeMetricSample) -> None:
        del sample

    def record_timing(
        self,
        name: str,
        duration_s: float,
        *,
        step_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        del name, duration_s, step_index, metadata

    def close(self) -> None:
        return None
