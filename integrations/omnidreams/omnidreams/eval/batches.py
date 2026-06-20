# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch planning for large scene datasets."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from omnidreams.eval.manifest import EvalCase

BATCH_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BatchPlan:
    """A bounded set of UUIDs to stage/generate/evaluate together."""

    batch_id: str
    case_uuids: tuple[str, ...]
    total_input_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatchPlan":
        return cls(
            batch_id=str(value["batch_id"]),
            case_uuids=tuple(str(v) for v in value["case_uuids"]),
            total_input_bytes=int(value["total_input_bytes"]),
        )


def plan_batches(
    cases: Sequence[EvalCase],
    *,
    batch_size: int | None = None,
    max_batch_bytes: int | None = None,
    completed_uuids: Iterable[str] = (),
) -> list[BatchPlan]:
    """Group cases into stable size- and byte-capped batches."""

    completed = set(completed_uuids)
    pending = [
        case
        for case in sorted(cases, key=lambda c: c.uuid)
        if case.uuid not in completed
    ]
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when set")
    if max_batch_bytes is not None and max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be positive when set")

    batches: list[BatchPlan] = []
    current: list[EvalCase] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        batches.append(
            BatchPlan(
                batch_id=f"batch-{len(batches):05d}",
                case_uuids=tuple(case.uuid for case in current),
                total_input_bytes=current_bytes,
            )
        )
        current = []
        current_bytes = 0

    for case in pending:
        case_bytes = case.total_input_bytes
        would_exceed_count = batch_size is not None and len(current) >= batch_size
        would_exceed_bytes = (
            max_batch_bytes is not None
            and current
            and current_bytes + case_bytes > max_batch_bytes
        )
        if would_exceed_count or would_exceed_bytes:
            flush()
        current.append(case)
        current_bytes += case_bytes
        if batch_size is not None and len(current) >= batch_size:
            flush()

    flush()
    return batches


def write_batch_plan(path: Path, batches: Sequence[BatchPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": BATCH_PLAN_SCHEMA_VERSION,
        "kind": "omnidreams_eval_batch_plan",
        "batches": [batch.to_dict() for batch in batches],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_batch_plan(path: Path) -> list[BatchPlan]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [BatchPlan.from_dict(value) for value in payload["batches"]]


def parse_byte_size(value: str) -> int:
    """Parse byte sizes such as ``250GB``, ``12GiB`` or ``1024``."""

    raw = value.strip()
    match = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?", raw)
    if match is None:
        raise ValueError(f"invalid byte size: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"invalid byte size: {value!r}")
    return int(number * multipliers[unit])


def cases_for_batch(cases: Sequence[EvalCase], batch: BatchPlan) -> list[EvalCase]:
    by_uuid = {case.uuid: case for case in cases}
    missing = [uuid for uuid in batch.case_uuids if uuid not in by_uuid]
    if missing:
        raise KeyError(f"batch references UUIDs missing from manifest: {missing}")
    return [by_uuid[uuid] for uuid in batch.case_uuids]
