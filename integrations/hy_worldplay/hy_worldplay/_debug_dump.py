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

"""Env-var-gated, CUDA-graph-safe tensor-dump harness for HY-WorldPlay parity diagnosis.

Set ``HY_DEBUG_DUMP=/path/to/file.jsonl`` (or any truthy value to dump
to ``hy_debug_dump.jsonl`` in CWD) to enable. Every :func:`dump` call
appends a single JSON line with tensor stats (shape, dtype, abs_mean,
mean, std, min, max, first-32 flat values). Disabled by default so
production and parity runs pay zero overhead, and silently no-ops
during CUDA graph capture so dump calls embedded in the graph-captured
forward don't invalidate the capture.

The vendor side gets parallel dumps via
``tests/parity_check/dump_patch.py``, which monkey-patches the same
call sites in the vendor source tree.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import Tensor

_DUMP_ENV_VAR = "HY_DEBUG_DUMP"

_lock = threading.Lock()
_context: dict[str, Any] = {}


def enabled() -> bool:
    """Return ``True`` iff ``HY_DEBUG_DUMP`` is set to a non-empty value."""
    return bool(os.environ.get(_DUMP_ENV_VAR, ""))


def _dump_path() -> str:
    val = os.environ.get(_DUMP_ENV_VAR, "")
    if not val:
        return ""
    # Generic truthy values map to a default file in CWD; anything else
    # is taken as the dump path itself.
    if val in {"1", "true", "True", "yes", "on"}:
        return os.path.abspath("hy_debug_dump.jsonl")
    return os.path.abspath(val)


def set_context(**kwargs: Any) -> None:
    """Bind per-call-site context (e.g. ``chunk_idx``, ``step_idx``, ``block_idx``).

    Context is merged into every subsequent :func:`dump` line until
    overridden or :func:`clear_context` is called.
    """
    with _lock:
        _context.update(kwargs)


def clear_context(*keys: str) -> None:
    """Drop one or more keys from the context, or all if no keys are passed."""
    with _lock:
        if not keys:
            _context.clear()
        else:
            for k in keys:
                _context.pop(k, None)


@contextmanager
def context(**kwargs: Any) -> Iterator[None]:
    """Push context for the duration of a ``with`` block."""
    old = {k: _context.get(k) for k in kwargs}
    set_context(**kwargs)
    try:
        yield
    finally:
        with _lock:
            for k, v in old.items():
                if v is None:
                    _context.pop(k, None)
                else:
                    _context[k] = v


def _tensor_stats(t: Tensor) -> dict[str, Any]:
    """Compute scalar stats and a short prefix sample of ``t``.

    Runs in float32 on the tensor's device and captures the resulting
    Python floats eagerly (``.item()``) so the dumped record is
    JSON-serialisable without holding device memory.
    """
    if not isinstance(t, Tensor):
        return {"non_tensor_repr": repr(t)[:200]}
    t32 = t.detach().float() if t.numel() > 0 else t
    flat = t32.reshape(-1)
    n = flat.numel()
    if n == 0:
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "numel": 0,
        }
    sample_n = min(32, n)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": n,
        "abs_mean": float(flat.abs().mean().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item() if n > 1 else 0.0),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "sample": flat[:sample_n].cpu().tolist(),
    }


def dump(name: str, tensor: Tensor | None, **extra: Any) -> None:
    """Append a JSON-line record for ``tensor`` plus the current context.

    No-op when :func:`enabled` is ``False`` and during CUDA graph
    capture (``.item()`` / file I/O inside a graph capture window would
    invalidate the capture). ``tensor=None`` records only the context
    and ``extra`` fields, which is useful for marking control-flow
    events that don't have an obvious tensor.
    """
    if not enabled():
        return
    # CUDA graph capture forbids host-synchronous tensor reads and file
    # I/O; bail before either can invalidate the capture.
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
    path = _dump_path()
    if not path:
        return

    with _lock:
        record: dict[str, Any] = {"name": name, **_context}
        if tensor is not None:
            record["tensor"] = _tensor_stats(tensor)
        if extra:
            record.update(extra)
        try:
            line = json.dumps(record, default=str)
        except (TypeError, ValueError) as e:
            # Best-effort fallback: stringify the non-JSON-clean payload
            # rather than failing the run for a diagnostic write.
            record["__json_error"] = str(e)
            record["__repr"] = repr({k: v for k, v in record.items() if k != "tensor"})[
                :500
            ]
            line = json.dumps({"name": name, "__error": str(e)})

        try:
            with open(path, "a") as f:
                f.write(line + "\n")
        except OSError:
            # Diagnostic-only: never let a dump failure abort the run.
            pass
