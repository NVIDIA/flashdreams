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

"""Serving-model registrations for LingBot-World variants."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from flashdreams.serving.api import (
    ModelCapabilities,
    ModelDescriptor,
    ResourceRequest,
    StreamInput,
    StreamOutput,
)
from flashdreams.serving.backend import ModelWorker
from flashdreams.serving.config import ServeModelConfig
from lingbot.config import PIPELINE_CONFIGS
from lingbot.runner import ensure_example_data_downloaded
from lingbot.webrtc.session import LingbotRuntimeConfig, LingbotWebRTCSessionManager


class LingbotServingWorker(ModelWorker):
    """Adapt the Lingbot WebRTC manager to the shared serving worker contract."""

    def __init__(self, worker_id: str, model: str) -> None:
        """Initialize an unloaded Lingbot worker.

        Args:
            worker_id: Scheduler-assigned worker identifier.
            model: Lingbot pipeline configuration slug.
        """
        self.worker_id = worker_id
        self._descriptor = _descriptor(model)
        self._manager = LingbotWebRTCSessionManager(
            runtime_config=LingbotRuntimeConfig(config_name=model)
        )
        self._session_id: str | None = None

    @property
    def descriptor(self) -> ModelDescriptor:
        """Return the hosted Lingbot model descriptor."""
        return self._descriptor

    async def start(self) -> None:
        """Download default inputs and load the Lingbot pipeline."""
        await asyncio.to_thread(
            ensure_example_data_downloaded, is_rank_zero=True, example_idx=0
        )
        await self._manager.preload_runtime()

    async def create_session(
        self, session_id: str, parameters: Mapping[str, Any]
    ) -> None:
        """Reserve the manager for a session before WebRTC negotiation."""
        del parameters
        if self._session_id is not None:
            raise RuntimeError("Lingbot worker already has a live session.")
        self._session_id = session_id

    async def stream(
        self, session_id: str, request: StreamInput
    ) -> AsyncIterator[StreamOutput]:
        """Reject WebSocket inference because Lingbot emits WebRTC media tracks."""
        del session_id, request
        raise NotImplementedError("Lingbot currently streams output over WebRTC.")
        yield  # pragma: no cover

    async def create_webrtc_answer(
        self, session_id: str, offer: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Negotiate a Lingbot WebRTC peer connection."""
        if session_id != self._session_id:
            raise RuntimeError("Session is not assigned to this Lingbot worker.")
        sdp = offer.get("sdp")
        offer_type = offer.get("type")
        if not isinstance(sdp, str) or not isinstance(offer_type, str):
            raise ValueError("WebRTC offer requires string sdp and type fields.")
        return await self._manager.create_answer(offer_sdp=sdp, offer_type=offer_type)

    async def close_session(self, session_id: str) -> None:
        """Close the peer connection and release Lingbot rollout state."""
        if session_id == self._session_id:
            await self._manager.close_active_session()
            self._session_id = None

    async def close(self) -> None:
        """Shut down the Lingbot manager and model runtime."""
        await self._manager.shutdown()
        self._session_id = None


def _descriptor(model: str) -> ModelDescriptor:
    return ModelDescriptor(
        id=model,
        capabilities=ModelCapabilities(
            inputs=("image", "text", "camera_actions"),
            outputs=("video",),
            transports=("webrtc",),
            sessions_per_worker=1,
        ),
        resources=ResourceRequest(gpu_count=1, placement="single-node"),
        metadata={"family": "lingbot-world"},
    )


def _config(model: str) -> ServeModelConfig:
    if model not in PIPELINE_CONFIGS:
        raise KeyError(f"Unknown Lingbot pipeline config {model!r}.")
    return ServeModelConfig(
        descriptor=_descriptor(model),
        worker_factory=lambda worker_id: LingbotServingWorker(worker_id, model),
    )


SERVE_LINGBOT_WORLD_FAST = _config("lingbot-world-fast")
"""Serving registration for the Lingbot World Fast model."""

SERVE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3 = _config(
    "lingbot-world-fast-taehv-window15-sink3"
)
"""Serving registration for the low-latency Lingbot World v1 model."""

SERVE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST = _config("lingbot-world-v2-14b-causal-fast")
"""Serving registration for the Lingbot World v2 causal-fast model."""

SERVE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3 = _config(
    "lingbot-world-v2-14b-causal-fast-taehv-window15-sink3"
)
"""Serving registration for the low-latency Lingbot World v2 model."""
