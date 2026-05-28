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

"""Vendor-side DiT-only CUDA-event profiler + optional cuDNN SDPA backend.

The patch operates on the *class* method
``WanTransformer3DModel.forward`` rather than wrapping an instance,
because diffusers introspects ``WanPipeline.__init__``'s signature for
component registration and a wrapped ``__init__`` makes it lose the
component names (raises ``"expected ['args', 'kwargs'], but only set()
were passed"``).

Each forward call records a ``(start, end)`` CUDA-event pair; on
``atexit`` we sync and dump the millisecond timings to a JSON file
matching the native side's ``stats_dit_native.json`` shape.

Env vars (all opt-in; module is a no-op when ``HY_DIT_PROFILE`` is
unset):

* ``HY_DIT_PROFILE=1`` -- enable the per-step timer + JSON dump.
* ``HY_VENDOR_CUDNN_SDPA=1`` -- enter
  :func:`torch.nn.attention.sdpa_kernel` with the cuDNN backend
  around each forward.
* ``HY_DIT_OUTPUT_JSON`` -- absolute path for the JSON dump; defaults
  to ``./stats_dit_vendor.json`` next to the working dir.

``torch.compile`` on vendor was attempted in an earlier draft but
vendor's KV cache uses dynamic shapes per chunk (rolling window +
memory-prefill resizing) that Inductor cannot trace cleanly. Native
ships ``compile_network=True`` via the static pipeline config; for
matched-stack measurement, prefer reporting the post-warmup median
DiT-forward ms on both sides at fixed cuDNN SDPA -- compile parity is
not in scope.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any, Callable

_records: list[tuple[Any, Any]] = []
"""Append-only ``(start_event, end_event)`` pairs across all DiT forwards."""

_install_done = False
"""Idempotency latch -- ``install_dit_profile_patch`` is safe to call multiple times."""


def install_dit_profile_patch() -> None:
    """Wrap vendor's ``WanTransformer3DModel.forward`` with CUDA-event timing.

    No-op when ``HY_DIT_PROFILE`` is unset. Idempotent across patch
    installers.
    """
    global _install_done
    if _install_done:
        return
    if os.environ.get("HY_DIT_PROFILE", "") != "1":
        return

    try:
        from wan.models.dits.arwan_w_action_w_mem_relative_rope import (
            WanTransformer3DModel,
        )
    except ImportError as exc:
        print(
            f"[dit_profile] cannot import vendor WanTransformer3DModel "
            f"({exc}); skipping vendor instrumentation.",
            flush=True,
        )
        return

    do_cudnn = os.environ.get("HY_VENDOR_CUDNN_SDPA", "") == "1"
    WanTransformer3DModel.forward = _wrap_forward(
        WanTransformer3DModel.forward, do_cudnn
    )
    print(
        f"[dit_profile] vendor DiT timer installed "
        f"(cuDNN SDPA={'on' if do_cudnn else 'off'}).",
        flush=True,
    )
    atexit.register(_dump_records)
    _install_done = True


def _wrap_forward(
    forward_callable: Callable[..., Any], do_cudnn: bool
) -> Callable[..., Any]:
    """Return a CUDA-event-timed wrapper around the class-bound ``forward``."""
    import torch

    if do_cudnn:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def timed_with_cudnn(self: Any, *args: object, **kwargs: object) -> Any:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                out = forward_callable(self, *args, **kwargs)
            end.record()
            _records.append((start, end))
            return out

        return timed_with_cudnn

    def timed_default(self: Any, *args: object, **kwargs: object) -> Any:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = forward_callable(self, *args, **kwargs)
        end.record()
        _records.append((start, end))
        return out

    return timed_default


def _dump_records() -> None:
    """Sync CUDA + write the collected DiT step timings to JSON."""
    if not _records:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times_ms = [start.elapsed_time(end) for start, end in _records]
    out_path = Path(
        os.environ.get("HY_DIT_OUTPUT_JSON", "stats_dit_vendor.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": "vendor",
        "compile_enabled": False,
        "cudnn_sdpa_enabled": os.environ.get("HY_VENDOR_CUDNN_SDPA", "") == "1",
        "dit_per_step_ms": [round(v, 3) for v in times_ms],
        "n_steps": len(times_ms),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[dit_profile] wrote {len(times_ms)} vendor DiT step ms -> {out_path}",
        flush=True,
    )
