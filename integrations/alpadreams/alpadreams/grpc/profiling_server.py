"""
Server-side profiling wrapper for the gRPC server.

This module adds timing instrumentation to the gRPC server to track:
- Total request processing time
- HDMap rendering time
- Video model inference time
- Image encoding time
- Road state computation time
- gRPC serialization overhead

Usage:
    Add --enable_profiling and --profile_output arguments to grpc_server.py

The profiler auto-saves periodically and on session end, so you don't need
to stop the server to get profiling data.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import torch  # type: ignore[import-untyped]

    _CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None  # type: ignore[assignment]  # ty:ignore[invalid-assignment]
    _CUDA_AVAILABLE = False

# Thread-local storage for passing profiling context through nested calls
_profiling_context = threading.local()


@contextmanager
def profiling_context(session_id: str, chunk_idx: int):
    """
    Context manager to set session_id and chunk_idx for nested profiling calls.

    Usage in grpc_server.py:
        with profiling_context(session_id, chunk_idx):
            result = self.api.continue_generation(...)

    Then in bbox_conditioned_api.py, get_profiling_context() will return
    the session_id and chunk_idx set by the caller.
    """
    old_session_id = getattr(_profiling_context, "session_id", None)
    old_chunk_idx = getattr(_profiling_context, "chunk_idx", None)

    _profiling_context.session_id = session_id
    _profiling_context.chunk_idx = chunk_idx

    try:
        yield
    finally:
        _profiling_context.session_id = old_session_id
        _profiling_context.chunk_idx = old_chunk_idx


def get_profiling_context() -> tuple[str, int]:
    """
    Get the current profiling context (session_id, chunk_idx).

    Returns:
        Tuple of (session_id, chunk_idx). Defaults to ("unknown", 0) if not set.
    """
    session_id = getattr(_profiling_context, "session_id", None) or "unknown"
    chunk_idx = getattr(_profiling_context, "chunk_idx", None) or 0
    return session_id, chunk_idx


@dataclass
class TimingRecord:
    """Single timing measurement."""

    category: str
    duration_ms: float
    timestamp: float
    chunk_idx: int
    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ServerProfiler:
    """Collects timing data on the server side.

    Profiling data is saved when the server shuts down (Ctrl+C).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.records: list[TimingRecord] = []
        self.session_chunk_counters: dict[str, int] = {}
        self._lock = threading.Lock()

        self.context_depth = 0
        self.pending_msgs: list[str | None] = []

    def get_chunk_idx(self, session_id: str) -> int:
        """Get current chunk index for a session."""
        with self._lock:
            return self.session_chunk_counters.get(session_id, 0)

    def increment_chunk_idx(self, session_id: str):
        """Increment chunk counter for a session."""
        with self._lock:
            self.session_chunk_counters[session_id] = (
                self.session_chunk_counters.get(session_id, 0) + 1
            )

    @contextmanager
    def measure(
        self,
        category: str,
        session_id: str = "unknown",
        chunk_idx: int | None = None,
        **metadata,
    ):
        """Context manager to measure execution time."""
        if not self.enabled:
            yield
            return

        if chunk_idx is None:
            chunk_idx = self.get_chunk_idx(session_id)

        start = time.perf_counter()
        timestamp = time.time()

        with self._lock:
            i = len(self.pending_msgs)
            self.pending_msgs.append(None)
            self.context_depth += 1

        try:
            yield
        finally:
            if _CUDA_AVAILABLE and torch is not None:
                torch.cuda.synchronize()
            duration_ms = (time.perf_counter() - start) * 1000

            record = TimingRecord(
                category=category,
                duration_ms=duration_ms,
                timestamp=timestamp,
                chunk_idx=chunk_idx,
                session_id=session_id,
                metadata=metadata,
            )

            indents = (
                "  " * (self.context_depth - 1) + "-" if self.context_depth > 1 else ""
            )
            msg = f"[Profile]{indents} {category}: {duration_ms:.2f} ms (chunk {chunk_idx}, session {session_id[:8]})"

            with self._lock:
                self.records.append(record)

                self.pending_msgs[i] = msg

                self.context_depth -= 1
                if self.context_depth == 0:
                    for msg in self.pending_msgs:
                        logger.info(msg)
                    self.pending_msgs = []
                assert self.context_depth >= 0, "Context depth cannot be negative"

    def save(self, output_path: str | Path):
        """Save profiling data to JSON."""
        output_path = Path(output_path)

        with self._lock:
            data = {
                "profiler": "server",
                "num_records": len(self.records),
                "records": [
                    {
                        "category": r.category,
                        "duration_ms": r.duration_ms,
                        "timestamp": r.timestamp,
                        "chunk_idx": r.chunk_idx,
                        "session_id": r.session_id,
                        "metadata": r.metadata,
                    }
                    for r in self.records
                ],
            }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.records)} profiling records to {output_path}")

    def print_summary(self):
        """Print summary statistics."""
        from collections import defaultdict

        import numpy as np

        with self._lock:
            if not self.records:
                logger.info("No profiling data collected")
                return

            by_category = defaultdict(list)
            for r in self.records:
                by_category[r.category].append(r.duration_ms)

        logger.info("=" * 80)
        logger.info("Server Profiling Summary")
        logger.info("=" * 80)

        for category in sorted(by_category.keys()):
            times = np.array(by_category[category])
            logger.info(
                f"{category:30s}: "
                f"mean={times.mean():7.2f}ms, "
                f"std={times.std():7.2f}ms, "
                f"min={times.min():7.2f}ms, "
                f"max={times.max():7.2f}ms, "
                f"n={len(times)}"
            )

        logger.info("=" * 80)


# Global profiler instance
_profiler: ServerProfiler | None = None


def init_profiler(enabled: bool = True) -> ServerProfiler:
    """Initialize the global profiler."""
    global _profiler
    _profiler = ServerProfiler(enabled=enabled)
    return _profiler


def get_profiler() -> ServerProfiler:
    """Get the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = ServerProfiler(enabled=False)
    return _profiler
