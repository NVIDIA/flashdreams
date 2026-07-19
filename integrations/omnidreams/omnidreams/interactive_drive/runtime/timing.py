# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility exports for shared realtime timing helpers."""

from flashdreams.serving.realtime.timing import (
    ChunkHistory,
    ChunkPrediction,
    ChunkTimes,
    FrameTimes,
    InputToPresentProfileWindow,
    InputToPresentSummary,
    RecentTimingSummary,
    RollingChunkTimingSummary,
    StageDurationSummary,
    TraceComponentValue,
    TraceContext,
    TraceSink,
    VideoModelTimings,
    VideoModelTraceEvents,
    chunk_stage_durations_ms,
    emit_video_model_timing_ranges,
    event_dependencies,
    summarize_chunk_history,
    summarize_stage_durations,
    trace_time_ns,
)

__all__ = [
    "ChunkHistory",
    "ChunkPrediction",
    "ChunkTimes",
    "FrameTimes",
    "InputToPresentProfileWindow",
    "InputToPresentSummary",
    "RecentTimingSummary",
    "RollingChunkTimingSummary",
    "StageDurationSummary",
    "TraceComponentValue",
    "TraceContext",
    "TraceSink",
    "VideoModelTimings",
    "VideoModelTraceEvents",
    "chunk_stage_durations_ms",
    "emit_video_model_timing_ranges",
    "event_dependencies",
    "summarize_chunk_history",
    "summarize_stage_durations",
    "trace_time_ns",
]
