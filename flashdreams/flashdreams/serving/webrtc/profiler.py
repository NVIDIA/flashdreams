# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC pipeline profiler for diagnosing timing gaps and stalls.

Enable at runtime by setting ``WEBRTC_PROFILE=1``.  Events are written
as newline-delimited JSON to ``WEBRTC_PROFILE_PATH`` (default:
``/tmp/webrtc_profile.jsonl``).  Use ``plot_webrtc_timeline.py`` to
visualise the resulting timeline.

Each event carries *relative* perf-counter timestamps (seconds since
the profiler was enabled) for sub-millisecond accuracy across threads.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from loguru import logger

_ENABLED = os.environ.get("WEBRTC_PROFILE", "").strip() in ("1", "true", "yes")
_OUTPUT_PATH = Path(
    os.environ.get("WEBRTC_PROFILE_PATH", "/tmp/webrtc_profile.jsonl")
)

_lock = threading.Lock()
_epoch: float = 0.0
_file = None


def _ensure_open() -> None:
    global _file, _epoch
    if _file is not None:
        return
    _epoch = time.perf_counter()
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _file = open(_OUTPUT_PATH, "w")
    logger.info(
        "WebRTC pipeline profiler enabled → {}  (epoch perf_counter={:.6f})",
        _OUTPUT_PATH,
        _epoch,
    )


def _write_event(record: dict[str, Any]) -> None:
    global _file
    if _file is None:
        return
    with _lock:
        _file.write(json.dumps(record, separators=(",", ":")) + "\n")
        _file.flush()


def is_enabled() -> bool:
    return _ENABLED


@contextmanager
def measure(
    stage: str,
    *,
    chunk_index: int = -1,
    **metadata: Any,
) -> Generator[None, None, None]:
    if not _ENABLED:
        yield
        return
    _ensure_open()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        _write_event(
            {
                "stage": stage,
                "start": round(t0 - _epoch, 6),
                "end": round(t1 - _epoch, 6),
                "dur_ms": round((t1 - t0) * 1000.0, 3),
                "chunk": chunk_index,
                "tid": threading.current_thread().name,
                **metadata,
            }
        )


def instant(
    stage: str,
    *,
    chunk_index: int = -1,
    **metadata: Any,
) -> None:
    if not _ENABLED:
        return
    _ensure_open()
    now = time.perf_counter()
    _write_event(
        {
            "stage": stage,
            "start": round(now - _epoch, 6),
            "end": round(now - _epoch, 6),
            "dur_ms": 0.0,
            "chunk": chunk_index,
            "tid": threading.current_thread().name,
            **metadata,
        }
    )


def close() -> None:
    global _file
    with _lock:
        if _file is not None:
            _file.close()
            _file = None
            logger.info("WebRTC pipeline profiler closed → {}", _OUTPUT_PATH)
