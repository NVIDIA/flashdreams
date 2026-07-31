# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output target boundary for generated inference results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.types import StepResult


@dataclass(frozen=True, kw_only=True, slots=True)
class OutputArtifact:
    """Artifact produced by an output target."""

    __hash__ = None

    kind: str
    uri: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("OutputArtifact.kind must be non-empty.")
        if not self.uri.strip():
            raise ValueError("OutputArtifact.uri must be non-empty.")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@runtime_checkable
class OutputTarget(Protocol):
    """Consumes generated session outputs for presentation or persistence."""

    def open(self) -> None:
        """Prepare the target for a new run."""
        ...

    def write(self, result: StepResult) -> None:
        """Consume one generated step result."""
        ...

    def close(self) -> Sequence[OutputArtifact]:
        """Finalize and return any produced artifacts."""
        ...


@dataclass(slots=True)
class NullOutputTarget:
    """Output target for headless runs and throughput measurements."""

    store_results: bool = False
    output_count: int = field(default=0, init=False)
    results: list[StepResult] = field(default_factory=list, init=False)
    _opened: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return not self._opened

    def open(self) -> None:
        self._opened = True
        self.output_count = 0
        self.results.clear()

    def write(self, result: StepResult) -> None:
        if not self._opened:
            raise RuntimeError("Cannot write to a closed output target.")
        self.output_count += 1
        if self.store_results:
            self.results.append(result)

    def close(self) -> Sequence[OutputArtifact]:
        self._opened = False
        return ()
