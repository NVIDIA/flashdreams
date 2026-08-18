# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared WebRTC data-channel message helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MESSAGE_TYPE_ACTION = "action"
MESSAGE_TYPE_CHUNK_DONE = "chunk_done"
MESSAGE_TYPE_DISCONNECT = "disconnect"
MESSAGE_TYPE_ERROR = "error"
MESSAGE_TYPE_EVENT = "event"
MESSAGE_TYPE_EVENT_ACK = "event_ack"
MESSAGE_TYPE_HEARTBEAT = "heartbeat"


def make_error_payload(message: str) -> dict[str, str]:
    return {"type": MESSAGE_TYPE_ERROR, "message": message}


def make_event_ack_payload(
    *,
    event_id: str | None,
    state: str,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": MESSAGE_TYPE_EVENT_ACK,
        "event_id": event_id,
        "state": state,
    }
    if result is not None:
        for key, value in result.items():
            if key not in payload:
                payload[key] = value
    return payload


def make_chunk_done_payload(
    *,
    chunk_index: int,
    num_frames: int,
    enqueued_frames: int,
    fps: int,
    width: int,
    height: int,
    model: str,
    gen_ms: float,
    enqueue_ms: float,
    play_ms: float,
    queue_depth: int,
    lag_ms: float,
    control_latency_ms: float | None = None,
    consumed_actions: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": MESSAGE_TYPE_CHUNK_DONE,
        "chunk_index": chunk_index,
        "num_frames": num_frames,
        "enqueued_frames": enqueued_frames,
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "model": model,
        "gen_ms": round(gen_ms, 1),
        "enqueue_ms": round(enqueue_ms, 1),
        "play_ms": round(play_ms, 1),
        "queue_depth": queue_depth,
        "lag_ms": round(lag_ms, 1),
    }
    if extra is not None:
        payload.update(extra)
    if control_latency_ms is not None:
        rounded_latency_ms = round(control_latency_ms, 1)
        payload["latency_ms"] = rounded_latency_ms
        payload["control_latency_ms"] = rounded_latency_ms
        payload["consumed_actions"] = (
            0 if consumed_actions is None else consumed_actions
        )
    return payload
