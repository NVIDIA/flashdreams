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

"""Per-AR-step ``EventProfiler`` instrumentation for vendor's ``WanPipeline``.

Monkey-patches :meth:`WanPipeline.__call__` and
:meth:`WanPipeline.decode_next_latent` so each chunk's diffuse + decode
times are recorded via :class:`flashdreams.infra.profiler.EventProfiler`.
The dump format matches the native runner's ``stats_<runner>.json``
shape (one dict per AR step with ``diffuse_ms`` / ``decode_ms`` /
``total_ms`` / ``mem_*_gib`` keys), so the same ``bench_summary.py``
post-warmup median logic applies to both sides.

Vendor's chunk loop (``wan/generate.py:213``) calls ``self.pipe(...)``
once per chunk for the diffusion, then ``self.pipe.decode_next_latent()``
``CHUNK_SIZE`` times (4) for the VAE. The patch keys timings on the
``chunk_i`` kwarg passed to ``__call__`` and records ``decode`` after
the ``CHUNK_SIZE``-th ``decode_next_latent`` call completes.

Env vars (all opt-in):

* ``HY_VENDOR_PROFILE=1`` -- enable the patch.
* ``HY_VENDOR_STATS_JSON`` -- absolute path for the per-AR-step
  JSON dump; defaults to ``./stats_hy-worldplay-wan-i2v-5b.json``
  next to the working dir (matching ``bench.sh``'s expected output).

References:
    * ``integrations/lingbot/tests/parity_check/changes.patch:185-246``
      -- the in-place patch shape Ruilong cited.
    * ``integrations/omnidreams/omnidreams/runner.py:473-478``
      -- the stats JSON dump shape both sides now share.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any, Callable

_RUNNER_NAME = "hy-worldplay-wan-i2v-5b"
"""Filename stem the bench harness expects for vendor's stats JSON."""

_CHUNK_SIZE = 4
"""Vendor's ``wan/inference/helper.py:CHUNK_SIZE`` -- one chunk emits
this many latents, and ``decode_next_latent`` runs once per latent
inside the bench loop. The chunk's ``decode`` stage closes when the
``CHUNK_SIZE``-th call returns."""

_install_done = False
"""Idempotency latch -- ``install_vendor_profile_patch`` is safe to call multiple times."""

_records: list[dict[str, Any]] = []
"""Append-only per-AR-step stats; dumped to JSON on ``atexit``."""

_current_chunk_idx: int = -1
"""Chunk index of the currently in-flight pipeline call."""

_current_profiler: Any = None
"""``EventProfiler`` instance for the in-flight chunk; created in ``__call__``."""

_decode_count_this_chunk: int = 0
"""How many ``decode_next_latent`` calls have completed since the last ``__call__``."""


def install_vendor_profile_patch() -> None:
    """Wrap vendor's ``WanPipeline.__call__`` / ``decode_next_latent`` with EventProfiler.

    No-op when ``HY_VENDOR_PROFILE`` is unset. Idempotent.
    """
    global _install_done
    if _install_done:
        return
    if os.environ.get("HY_VENDOR_PROFILE", "") != "1":
        return

    from wan.inference import pipeline_wan_w_mem_relative_rope as _mod

    pipeline_cls = _mod.WanPipeline
    pipeline_cls.__call__ = _wrap_pipeline_call(pipeline_cls.__call__)
    pipeline_cls.decode_next_latent = _wrap_decode_next_latent(
        pipeline_cls.decode_next_latent
    )

    atexit.register(_dump_records)
    print("[vendor_profile] vendor EventProfiler patch installed.", flush=True)
    _install_done = True


def _wrap_pipeline_call(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Open a fresh ``EventProfiler`` for the chunk, then record ``diffuse`` on return."""
    import torch

    from flashdreams.infra.profiler import EventProfiler

    def timed_call(self: Any, *args: object, **kwargs: object) -> Any:
        global _current_profiler, _current_chunk_idx, _decode_count_this_chunk
        chunk_i = kwargs.get("chunk_i")
        _current_chunk_idx = int(chunk_i) if chunk_i is not None else 0
        _current_profiler = EventProfiler()
        _decode_count_this_chunk = 0
        result = original(self, *args, **kwargs)
        _current_profiler.record("diffuse")
        return result

    return timed_call


def _wrap_decode_next_latent(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Increment the per-chunk decode counter; on the last call, close the chunk's stats."""
    import torch

    def timed_decode(self: Any, *args: object, **kwargs: object) -> Any:
        global _decode_count_this_chunk
        result = original(self, *args, **kwargs)
        _decode_count_this_chunk += 1
        if _decode_count_this_chunk >= _CHUNK_SIZE and _current_profiler is not None:
            _current_profiler.record("decode")
            stats_ms = _current_profiler.sync_and_summarize()
            stats: dict[str, Any] = {
                f"{stage}_ms": round(float(ms), 3) for stage, ms in stats_ms.items()
            }
            total_ms = sum(stats_ms.values())
            stats["total_ms"] = round(total_ms, 3)
            # Vendor has no ``finalize`` stage; mirror native's
            # ``total_ms_wo_finalize`` field as the same number so the
            # bench-summary "wall clock" row works on both sides.
            stats["total_ms_wo_finalize"] = round(total_ms, 3)
            if torch.cuda.is_available():
                gib = 1024**3
                stats["mem_alloc_gib"] = round(torch.cuda.memory_allocated() / gib, 3)
                stats["mem_reserved_gib"] = round(torch.cuda.memory_reserved() / gib, 3)
                stats["mem_peak_gib"] = round(
                    torch.cuda.max_memory_allocated() / gib, 3
                )
            _records.append({"autoregressive_index": _current_chunk_idx, **stats})
        return result

    return timed_decode


def _dump_records() -> None:
    """Write the accumulated per-AR-step stats to JSON."""
    if not _records:
        return
    out_path = Path(
        os.environ.get("HY_VENDOR_STATS_JSON", f"stats_{_RUNNER_NAME}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_records, indent=2))
    print(
        f"[vendor_profile] wrote {len(_records)} per-AR-step stats -> {out_path}",
        flush=True,
    )
