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

"""LingBot-World WebRTC runtime and session management."""

from __future__ import annotations

import http.client
import io
import ipaddress
import re
import socket
import ssl
import threading
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from flashdreams.core.distributed.rank_orchestration import distributed_op
from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.infra.config import derive_config
from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.inputs import (
    InferenceInput,
    UserInputCapability,
    UserInputSchema,
)
from flashdreams.runtime.types import StepRequest, StepResult
from flashdreams.serving.webrtc.encoders import EncoderBackend
from flashdreams.serving.webrtc.manager import (
    DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    BaseWebRTCSessionManager,
    WebRTCControlSignal,
)
from flashdreams.serving.webrtc.runtime import (
    ThreadAffineDistributedWebRTCRuntime,
)
from flashdreams.serving.webrtc.server import SessionBusyError
from lingbot.controls import CameraPoseIntegrator, PoseSegment
from lingbot.encoder.utils import preprocess_example_poses
from lingbot.input_mapping import (
    FIELD_CAMERA_INTRINSICS,
    FIELD_CAMERA_TRAJECTORY,
    KeyboardToCameraCommand,
    LingbotInputMapping,
    TextEventSelection,
)
from lingbot.model_session import LingbotModelSessionCore

_INTRINSICS_REFERENCE_HEIGHT = 480
_INTRINSICS_REFERENCE_WIDTH = 832
_DEFAULT_INTRINSICS = (
    502.9115905761719,
    503.1081237792969,
    415.7778625488281,
    239.7777862548828,
)
# Aligned with the world scale computed from the first LingBot-World demo scene.
_DEFAULT_WORLD_SCALE = 1.271182656288147
_DEFAULT_DEMO_BASE_URL = (
    "https://raw.githubusercontent.com/Robbyant/lingbot-world-v2/main/examples/00"
)
_DEFAULT_IMAGE_URL = f"{_DEFAULT_DEMO_BASE_URL}/image.jpg"
_DEFAULT_INTRINSICS_URL = f"{_DEFAULT_DEMO_BASE_URL}/intrinsics.npy"
_DEFAULT_POSES_URL = f"{_DEFAULT_DEMO_BASE_URL}/poses.npy"
_MAX_REMOTE_IMAGE_BYTES = 15 * 1024 * 1024
_MAX_REMOTE_NUMPY_BYTES = 64 * 1024 * 1024
_REMOTE_READ_TIMEOUT_S = 20.0
_MAX_REMOTE_REDIRECTS = 5
_BLOCKED_REMOTE_HOSTNAMES = {"localhost", "localhost.localdomain"}
_MAX_TEXT_EVENTS = 12
_MAX_TEXT_EVENT_LABEL_CHARS = 64
_MAX_TEXT_EVENT_PROMPT_CHARS = 1_000
_TEXT_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class LingbotRuntimeError(RuntimeError):
    """Raised when the Lingbot runtime is used incorrectly."""


@dataclass(frozen=True, slots=True)
class TextEventSpec:
    """Server-owned text event exposed to WebRTC clients by stable id."""

    event_id: str
    """Stable identifier sent over the WebRTC data channel."""

    label: str
    """Short client-facing event label."""

    prompt: str
    """Text context activated when the event is triggered."""

    category: str = "environment"
    """Client-facing group used to organize event controls."""

    def as_public_dict(self) -> dict[str, str]:
        """Return the client-facing event payload."""
        return {
            "event_id": self.event_id,
            "label": self.label,
            "prompt": self.prompt,
            "category": self.category,
        }


DEFAULT_TEXT_EVENTS: tuple[TextEventSpec, ...] = (
    TextEventSpec(
        event_id="portal",
        label="Portal",
        prompt=(
            "A luminous magical portal opens in the scene, casting colored light "
            "and swirling particles into the environment."
        ),
    ),
    TextEventSpec(
        event_id="storm",
        label="Storm",
        prompt=(
            "A dramatic storm rolls in with dark clouds, wind, rain, and flashes "
            "of lightning reshaping the atmosphere."
        ),
    ),
    TextEventSpec(
        event_id="fireworks",
        label="Fireworks",
        prompt=(
            "Bright fireworks burst overhead, filling the sky with colorful sparks "
            "and reflections across the scene."
        ),
    ),
)
"""Default text events advertised by the interactive viewer."""


def _content_type_for_image_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _normalize_github_blob_url(url: str, parsed: urllib.parse.ParseResult) -> str:
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"github.com", "www.github.com"}:
        return url

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 5 or path_parts[2] != "blob":
        return url

    owner, repo, _, ref, *file_path = path_parts
    raw_path = "/" + "/".join([owner, repo, ref, *file_path])
    return urllib.parse.urlunparse(
        ("https", "raw.githubusercontent.com", raw_path, "", "", "")
    )


def _resolve_remote_host(
    hostname: str,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        return (ipaddress.ip_address(hostname),)
    except ValueError:
        pass

    try:
        address_infos = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(
            f"Remote URL host {hostname!r} could not be resolved."
        ) from exc

    addresses: dict[str, ipaddress.IPv4Address | ipaddress.IPv6Address] = {}
    for address_info in address_infos:
        socket_address = address_info[4]
        if not socket_address:
            continue
        try:
            address = ipaddress.ip_address(socket_address[0])
        except ValueError:
            continue
        addresses[str(address)] = address
    if not addresses:
        raise ValueError(
            f"Remote URL host {hostname!r} did not resolve to an IP address."
        )
    return tuple(addresses.values())


def _validate_remote_hostname(hostname: str | None, *, field_name: str) -> None:
    if not hostname:
        raise ValueError(f"{field_name} must include a host.")
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname in _BLOCKED_REMOTE_HOSTNAMES or normalized_hostname.endswith(
        ".localhost"
    ):
        raise ValueError(f"{field_name} host must be publicly routable.")

    addresses = _resolve_remote_host(normalized_hostname)
    if any(not address.is_global for address in addresses):
        raise ValueError(f"{field_name} host must be publicly routable.")


def _validate_remote_url(url: str, *, field_name: str) -> str:
    normalized = url.strip()
    parsed = urllib.parse.urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an http(s) URL.")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} has an invalid port.") from exc
    normalized = _normalize_github_blob_url(normalized, parsed)
    parsed = urllib.parse.urlparse(normalized)
    _validate_remote_hostname(parsed.hostname, field_name=field_name)
    return normalized


def _open_resolved_socket(
    resolved_address: str,
    *,
    port: int,
    timeout: float,
) -> socket.socket:
    """Open a socket to the already-validated public address."""
    connection = socket.create_connection(
        (resolved_address, port),
        timeout=timeout,
    )
    try:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    return connection


class _ResolvedHTTPConnection(http.client.HTTPConnection):
    def __init__(
        self,
        host: str,
        *,
        port: int | None,
        timeout: float,
        resolved_address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> None:
        super().__init__(host=host, port=port, timeout=timeout)
        self._resolved_address = str(resolved_address)
        self._resolved_timeout = timeout

    def connect(self) -> None:
        self.sock = _open_resolved_socket(
            self._resolved_address,
            port=self.port,
            timeout=self._resolved_timeout,
        )


class _ResolvedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        *,
        port: int | None,
        timeout: float,
        resolved_address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> None:
        ssl_context = ssl.create_default_context()
        super().__init__(host=host, port=port, timeout=timeout, context=ssl_context)
        self._resolved_address = str(resolved_address)
        self._resolved_timeout = timeout
        self._ssl_context = ssl_context

    def connect(self) -> None:
        raw_socket = _open_resolved_socket(
            self._resolved_address,
            port=self.port,
            timeout=self._resolved_timeout,
        )
        try:
            self.sock = self._ssl_context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except Exception:
            raw_socket.close()
            raise


def _remote_request_target(parsed: urllib.parse.ParseResult) -> str:
    return urllib.parse.urlunparse(
        ("", "", parsed.path or "/", parsed.params, parsed.query, "")
    )


def _read_remote_bytes_once(
    url: str, *, max_bytes: int, field_name: str
) -> tuple[bytes, str, str | None]:
    normalized = _validate_remote_url(url, field_name=field_name)
    parsed = urllib.parse.urlparse(normalized)
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(f"{field_name} must include a host.")
    addresses = _resolve_remote_host(hostname.rstrip(".").lower())
    if any(not address.is_global for address in addresses):
        raise ValueError(f"{field_name} host must be publicly routable.")

    connection_cls: type[http.client.HTTPConnection] = (
        _ResolvedHTTPSConnection
        if parsed.scheme == "https"
        else _ResolvedHTTPConnection
    )
    last_error: Exception | None = None
    for address in addresses:
        connection = connection_cls(
            hostname,
            port=parsed.port,
            timeout=_REMOTE_READ_TIMEOUT_S,
            resolved_address=address,
        )
        try:
            connection.request(
                "GET",
                _remote_request_target(parsed),
                headers={"User-Agent": "flashdreams-lingbot-webrtc/1.0"},
            )
            response = connection.getresponse()
            try:
                location = response.getheader("Location")
                if response.status in {301, 302, 303, 307, 308}:
                    response.read()
                    if not location:
                        raise ValueError(f"{field_name} redirect missing Location.")
                    return b"", "", urllib.parse.urljoin(normalized, location)
                if response.status >= 400:
                    response.read()
                    raise ValueError(
                        f"{field_name} returned HTTP status {response.status}."
                    )
                data = response.read(max_bytes + 1)
                content_type = response.headers.get_content_type()
                return data, content_type, None
            finally:
                response.close()
        except ValueError:
            raise
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            last_error = exc
        finally:
            connection.close()

    if last_error is None:
        raise ValueError(f"Failed to fetch {field_name}.")
    raise ValueError(f"Failed to fetch {field_name}: {last_error}") from last_error


def _read_remote_bytes(
    url: str, *, max_bytes: int, field_name: str
) -> tuple[bytes, str]:
    current_url = url
    for redirect_idx in range(_MAX_REMOTE_REDIRECTS + 1):
        data, content_type, redirect_url = _read_remote_bytes_once(
            current_url,
            max_bytes=max_bytes,
            field_name=field_name if redirect_idx == 0 else f"{field_name} redirect",
        )
        if redirect_url is None:
            if len(data) > max_bytes:
                raise ValueError(f"{field_name} exceeds {max_bytes} bytes.")
            if not data:
                raise ValueError(f"{field_name} returned an empty response.")
            return data, content_type
        current_url = _validate_remote_url(
            redirect_url, field_name=f"{field_name} redirect"
        )
    raise ValueError(f"{field_name} exceeded {_MAX_REMOTE_REDIRECTS} redirects.")


def _decode_image_bytes_rgb(image_bytes: bytes, *, field_name: str) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"{field_name} could not be decoded as an image.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _load_npy_payload(source: Path | str, *, field_name: str) -> np.ndarray:
    if isinstance(source, Path):
        return np.load(source, allow_pickle=False)
    data, _ = _read_remote_bytes(
        source, max_bytes=_MAX_REMOTE_NUMPY_BYTES, field_name=field_name
    )
    return np.load(io.BytesIO(data), allow_pickle=False)


def _pipeline_configs() -> dict[str, Any]:
    from lingbot.config import PIPELINE_CONFIGS  # noqa: PLC0415

    return PIPELINE_CONFIGS


def _transform_intrinsics(
    intrinsics: torch.Tensor,
    *,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
) -> torch.Tensor:
    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)
    scale_x = width_resize / width_org
    scale_y = height_resize / height_org
    transformed = torch.zeros_like(intrinsics)
    transformed[..., 0:1] = fx * scale_x
    transformed[..., 1:2] = fy * scale_y
    transformed[..., 2:3] = cx * scale_x - (width_resize - width_final) / 2
    transformed[..., 3:4] = cy * scale_y - (height_resize - height_final) / 2
    return transformed


@dataclass(slots=True)
class LingbotRuntimeConfig:
    config_name: str = "lingbot-world-fast-taehv-window15-sink3"
    compile_network: bool = True
    seed: int = 42
    context_parallel_size: int = 1
    device: str = "cuda:0"
    video_height: int = 464
    video_width: int = 832
    fps: int = 16
    world_scale: float | None = None
    default_intrinsics: tuple[float, float, float, float] | None = None
    default_prompt: str = ""
    """Prompt used when the selected example does not provide ``prompt.txt``."""
    default_image_url: str | None = _DEFAULT_IMAGE_URL
    default_intrinsics_url: str | None = _DEFAULT_INTRINSICS_URL
    default_poses_url: str | None = _DEFAULT_POSES_URL
    warmup_chunks: int = 10
    warmup_timeout_s: float = 600.0
    encoder_backend: EncoderBackend = "auto"
    encoder_bitrate_bps: int = 6_000_000
    encoder_gop: int = 30

    example_data_dir: Path = field(
        default_factory=lambda: default_flashdreams_cache_dir()
        / "example_data/lingbot_world"
    )
    first_frame_filename: str = "image.jpg"
    intrinsics_filename: str = "intrinsics.npy"
    poses_filename: str = "poses.npy"
    prompt_filename: str = "prompt.txt"
    text_events: tuple[TextEventSpec, ...] = field(
        default_factory=lambda: DEFAULT_TEXT_EVENTS
    )
    pipeline_config: Any | None = None
    """Optional pre-resolved pipeline config used by shared demo adapters."""


@dataclass(frozen=True, slots=True)
class LingbotImagePayload:
    data: bytes
    content_type: str


@dataclass(frozen=True, slots=True)
class LingbotSessionInput:
    prompt: str | None = None
    first_frame_image_bytes: bytes | None = None
    first_frame_image_url: str | None = None
    first_frame_content_type: str = "image/jpeg"
    first_frame_remote_payload: LingbotImagePayload | None = None
    text_events: tuple[TextEventSpec, ...] | None = None


def normalize_prompt_text(prompt: str) -> str:
    """Collapse prompt whitespace into a single line."""
    return " ".join(prompt.split())


def _slugify_event_id(label: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return (slug or f"event-{index + 1}")[:64]


def _normalize_text_event_field(value: object) -> str:
    return normalize_prompt_text(str(value)) if value is not None else ""


def normalize_text_events(raw_events: object) -> tuple[TextEventSpec, ...]:
    """Validate and normalize a client-supplied text-event catalog."""
    if not isinstance(raw_events, (list, tuple)):
        raise ValueError("Text events must be a list.")

    text_events: list[TextEventSpec] = []
    seen_ids: set[str] = set()
    for index, raw_event in enumerate(raw_events):
        if isinstance(raw_event, TextEventSpec):
            event_id = raw_event.event_id.strip()
            label = normalize_prompt_text(raw_event.label)
            prompt = normalize_prompt_text(raw_event.prompt)
            category = normalize_prompt_text(raw_event.category) or "custom"
        elif isinstance(raw_event, dict):
            label = _normalize_text_event_field(raw_event.get("label"))
            prompt = _normalize_text_event_field(raw_event.get("prompt"))
            raw_event_id = _normalize_text_event_field(
                raw_event.get("event_id", raw_event.get("id"))
            )
            event_id = raw_event_id or _slugify_event_id(label, index)
            category = _normalize_text_event_field(raw_event.get("category"))
        else:
            raise ValueError("Each text event must be an object.")

        if not event_id and not label and not prompt:
            continue
        if not prompt:
            raise ValueError("Text event prompt is required.")
        if not label:
            label = event_id
        if len(label) > _MAX_TEXT_EVENT_LABEL_CHARS:
            raise ValueError(
                f"Text event labels must be <= {_MAX_TEXT_EVENT_LABEL_CHARS} characters."
            )
        if len(prompt) > _MAX_TEXT_EVENT_PROMPT_CHARS:
            raise ValueError(
                "Text event prompts must be "
                f"<= {_MAX_TEXT_EVENT_PROMPT_CHARS} characters."
            )
        if not _TEXT_EVENT_ID_RE.fullmatch(event_id):
            raise ValueError(
                "Text event ids must be 1-64 characters using only letters, "
                "numbers, '_', '.', ':', or '-'."
            )
        if event_id in seen_ids:
            raise ValueError(f"Duplicate text event id={event_id!r}.")
        seen_ids.add(event_id)
        text_events.append(
            TextEventSpec(
                event_id=event_id,
                label=label,
                prompt=prompt,
                category=category or "custom",
            )
        )

    if len(text_events) > _MAX_TEXT_EVENTS:
        raise ValueError(f"At most {_MAX_TEXT_EVENTS} text events are supported.")
    return tuple(text_events)


class LingbotInferenceRuntime(
    ThreadAffineDistributedWebRTCRuntime[
        LingbotRuntimeConfig,
        LingbotSessionInput,
    ]
):
    """Single-session Lingbot runtime with action-bound chunk generation."""

    def __init__(self, config: LingbotRuntimeConfig | None = None) -> None:
        super().__init__(
            config=config or LingbotRuntimeConfig(),
            runtime_error_type=LingbotRuntimeError,
            thread_name="lingbot-webrtc-runtime",
        )

        self.pose_integrator = CameraPoseIntegrator()

        self._pipeline: Any | None = None
        self._model_session: LingbotModelSessionCore | None = None
        self._base_intrinsics: torch.Tensor | None = None
        self._first_frames: torch.Tensor | None = None
        self._prompt: str | None = None
        self._base_text_embeddings: torch.Tensor | None = None
        self._event_embeddings: dict[str, torch.Tensor] = {}
        self._prompt_embeddings: dict[str, torch.Tensor] = {}
        self._active_event_id: str | None = None
        self._input_mapping: LingbotInputMapping | None = None
        self._input_canonicalizer: InputCanonicalizer | None = None
        self._sync_step_lock = threading.Lock()
        self._world_scale = 1.0

    async def trigger_event(
        self, *, event_id: str, state: str = "trigger"
    ) -> dict[str, str | None]:
        """Activate or clear a precomputed text event for subsequent chunks."""
        if self._closed:
            raise LingbotRuntimeError("Runtime is closed.")
        if self._pipeline is None or self._model_session is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        event_id, state = self._validate_event_request(event_id=event_id, state=state)
        async with self._step_lock:
            if self._closed:
                raise LingbotRuntimeError("Runtime is closed.")
            if self._pipeline is None or self._model_session is None:
                raise LingbotRuntimeError("Runtime is not initialized.")
            return await self._worker.call(
                self._trigger_event_sync_all_ranks,
                event_id,
                state,
            )

    async def start_inference_session(self) -> LingbotWebRTCInferenceSession:
        """Return an ``InferenceSession`` view of the current rollout.

        Shared demo providers prepare per-step model inputs before handing them
        to this session. The legacy direct WebRTC path may still expose
        ``input_mapping``/``input_canonicalizer`` to the shared manager, but
        starting a session only requires an initialized rollout.
        """
        if self._closed:
            raise LingbotRuntimeError("Runtime is closed.")
        if not self._is_runtime_initialized():
            raise LingbotRuntimeError("Runtime is not initialized.")
        return LingbotWebRTCInferenceSession(runtime=self)

    @property
    def input_mapping(self) -> LingbotInputMapping:
        if self._input_mapping is None:
            raise LingbotRuntimeError("Runtime input mapping is not initialized.")
        return self._input_mapping

    @property
    def input_canonicalizer(self) -> InputCanonicalizer:
        if self._input_canonicalizer is None:
            raise LingbotRuntimeError("Runtime canonicalizer is not initialized.")
        return self._input_canonicalizer

    @property
    def input_source_schema(self) -> UserInputSchema:
        return LINGBOT_WEBRTC_SOURCE_SCHEMA

    def validate_user_event(
        self, *, event_type: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Validate one raw WebRTC user event before it is acknowledged."""
        if event_type != "text_event":
            return payload
        event_id_value = payload.get("event_id")
        event_id = "" if event_id_value is None else str(event_id_value)
        state = str(payload.get("state", "trigger")).strip().lower() or "trigger"
        event_id, state = self._validate_event_request(event_id=event_id, state=state)
        clears = state in {"clear", "release", "off", "none"}
        return {"event_id": None if clears else event_id, "state": state}

    def _build_input_layers_sync(self, text_events: tuple[TextEventSpec, ...]) -> None:
        """Build the canonicalizer and mapping for the current rollout."""
        if self._base_intrinsics is None:
            self._input_mapping = None
            self._input_canonicalizer = None
            return
        self._input_canonicalizer = InputCanonicalizer(
            [KeyboardToCameraCommand(), TextEventSelection()]
        )
        self._input_mapping = LingbotInputMapping(
            fps=int(self.config.fps),
            base_intrinsics=self._base_intrinsics.detach().reshape(4).cpu(),
            world_scale=self._world_scale or 1.0,
            text_event_prompts={event.event_id: event.prompt for event in text_events},
        )
        self._input_mapping.set_base_prompt(self._prompt or "")

    def _next_step_request_sync(self) -> StepRequest:
        """Describe the next provider-prepared chunk for the session branch."""
        if self._model_session is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        step_index = self._model_session.step_index
        num_frames = self._model_session.next_num_frames()
        return StepRequest(
            step_index=step_index,
            metadata={
                "input_frame_count": num_frames,
                "num_frames": num_frames,
                "frame_start": step_index * num_frames,
            },
        )

    def _step_blocking(self, inputs: InferenceInput) -> StepResult:
        """Run one mapped step from synchronous ``InferenceSession`` code."""
        if self._closed:
            raise LingbotRuntimeError("Session is closed.")
        with self._sync_step_lock:
            if self._closed:
                raise LingbotRuntimeError("Session is closed.")
            return self._worker.call_blocking(self._step_sync_all_ranks, inputs)

    # Arbitrary index well past the AR-step transient; for the Wan/lingbot
    # pipelines used here the per-step count is constant for any index
    # ``>= 1`` (only AR 0 emits fewer frames due to causal first-frame
    # padding). Picking a large number is a robust way to ask "what is
    # the steady-state chunk size?" without leaning on the exact
    # boundary of that transient.
    _STEADY_STATE_AR_PROBE_INDEX: int = 1000

    def _is_runtime_initialized(self) -> bool:
        return self._pipeline is not None and self._model_session is not None

    def _runtime_step_index(self) -> int:
        if self._model_session is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        return self._model_session.step_index

    def _next_input_frame_count(self) -> int:
        if self._model_session is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        return self._model_session.next_num_frames()

    def _steady_output_frame_count(self) -> int:
        """Return the steady-state per-chunk frame count.

        AR step 0 emits *fewer* frames than every subsequent step
        because of the decoder's causal first-frame padding (e.g. AR 0
        → 9 frames vs AR ≥ 1 → 12 frames for the current config). The
        video track's bounded queue must be sized to the *steady-state*
        chunk size so that the producer is not forced to block on the
        very next chunk after the AR-0 transient. Probing at a large AR
        index returns that steady-state value directly.

        Master-only read with no distributed broadcast.
        """
        if self._pipeline is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        return int(
            self._pipeline.get_num_output_frames(self._STEADY_STATE_AR_PROBE_INDEX)
        )

    @distributed_op(WebRTCControlSignal.SESSION_STEP)
    def _step_sync_all_ranks(self, inputs: InferenceInput) -> StepResult:
        return self._step_sync(inputs)

    @distributed_op(WebRTCControlSignal.EVENT)
    def _trigger_event_sync_all_ranks(
        self,
        event_id: str,
        state: str = "trigger",
    ) -> dict[str, str | None]:
        return self._trigger_event_sync(event_id=event_id, state=state)

    def _initialize_sync(self) -> None:
        if self._pipeline is not None:
            return

        pipeline_config_base = self.config.pipeline_config
        if pipeline_config_base is None:
            pipeline_configs = _pipeline_configs()
            if self.config.config_name not in pipeline_configs:
                supported = ", ".join(sorted(pipeline_configs))
                raise ValueError(
                    f"Unknown config_name={self.config.config_name!r}. "
                    f"Supported: {supported}"
                )
            pipeline_config_base = pipeline_configs[self.config.config_name]

        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Lingbot runtime.")

        self._base_intrinsics = self._build_base_intrinsics()
        self._world_scale = self._resolve_world_scale()

        rollout_seed = (
            self.config.seed + self.rank
            if self.config.context_parallel_size > 1
            else self.config.seed
        )
        pipeline_config = derive_config(
            base_config=pipeline_config_base,
            enable_sync_and_profile=True,
            diffusion_model=dict(
                seed=rollout_seed,
                transformer=dict(
                    compile_network=self.config.compile_network,
                    init_device=str(self._device),
                ),
            ),
        )
        self._pipeline = pipeline_config.setup().to(device=self._device)
        self._model_session = LingbotModelSessionCore(
            pipeline=self._pipeline,
            output_stream_factory=lambda: VideoOutputStream(
                postprocess_stream=None,
                output_layout="tchw",
            ),
        )
        self._reset_rollout_sync()
        self._initialize_video_encoder_sync()

    def _encode_text_embeddings_sync(self, texts: list[str]) -> torch.Tensor:
        if self._pipeline is None:
            raise LingbotRuntimeError("Runtime pipeline is not initialized.")
        self._pipeline._ensure_oneshot_encoders_loaded()
        assert self._pipeline.text_encoder is not None
        return self._pipeline.text_encoder(texts).to(device=self._device)

    def _precompute_event_embeddings_sync(
        self, text_events: tuple[TextEventSpec, ...]
    ) -> None:
        if not text_events:
            self._event_embeddings = {}
            self._prompt_embeddings = {}
            return
        event_ids = [event.event_id for event in text_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("text event ids must be unique.")
        prompts = [event.prompt for event in text_events]
        embeddings = self._encode_text_embeddings_sync(prompts)
        self._event_embeddings = {
            event_id: embeddings[index : index + 1].contiguous()
            for index, event_id in enumerate(event_ids)
        }
        # The session branch receives a prompt rather than an event id, so keep
        # a prompt-keyed view of the same tensors. Without it a live text event
        # would pay a text-encoder pass mid-rollout.
        self._prompt_embeddings = {
            prompt: self._event_embeddings[event_id]
            for prompt, event_id in zip(prompts, event_ids, strict=True)
        }

    def _build_base_intrinsics(self) -> torch.Tensor:
        intrinsics_path = self.config.example_data_dir / self.config.intrinsics_filename
        if self.config.default_intrinsics is not None:
            intrinsics = np.asarray(self.config.default_intrinsics, dtype=np.float32)
        elif intrinsics_path.exists():
            intrinsics = _load_npy_payload(
                intrinsics_path, field_name="Lingbot default intrinsics"
            )
        elif self.config.default_intrinsics_url:
            intrinsics = _load_npy_payload(
                self.config.default_intrinsics_url,
                field_name="Lingbot default intrinsics URL",
            )
        else:
            intrinsics = np.asarray(_DEFAULT_INTRINSICS, dtype=np.float32)

        base_intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if base_intrinsics.ndim == 2 and base_intrinsics.shape[1] == 4:
            base_intrinsics = base_intrinsics[0]
        if base_intrinsics.shape != (4,):
            raise ValueError(
                f"Expected default Lingbot intrinsics shape (4,) or [N, 4], "
                f"got {base_intrinsics.shape}."
            )

        base_intrinsics_t = torch.from_numpy(base_intrinsics).to(
            device=self._device, dtype=torch.float32
        )
        return _transform_intrinsics(
            base_intrinsics_t.view(1, 4),
            height_org=_INTRINSICS_REFERENCE_HEIGHT,
            width_org=_INTRINSICS_REFERENCE_WIDTH,
            height_resize=self.config.video_height,
            width_resize=self.config.video_width,
            height_final=self.config.video_height,
            width_final=self.config.video_width,
        ).view(4)

    def _resolve_world_scale(self) -> float:
        if self.config.world_scale is not None:
            world_scale = float(self.config.world_scale)
            if world_scale <= 0:
                raise ValueError(f"world_scale must be > 0, got {world_scale}.")
            return world_scale

        poses_path = self.config.example_data_dir / self.config.poses_filename
        if poses_path.exists():
            poses = _load_npy_payload(poses_path, field_name="Lingbot default poses")
        elif self.config.default_poses_url:
            poses = _load_npy_payload(
                self.config.default_poses_url,
                field_name="Lingbot default poses URL",
            )
        else:
            return _DEFAULT_WORLD_SCALE

        _, world_scale = preprocess_example_poses(np.asarray(poses, dtype=np.float32))
        world_scale = float(world_scale)
        if world_scale <= 0:
            return _DEFAULT_WORLD_SCALE
        return world_scale

    def _load_default_prompt(self) -> str:
        prompt_path = self.config.example_data_dir / self.config.prompt_filename
        if prompt_path.exists():
            with prompt_path.open("r", encoding="utf-8") as handle:
                prompt = normalize_prompt_text(handle.readline())
            if prompt:
                return prompt
        prompt = normalize_prompt_text(self.config.default_prompt)
        if not prompt and (not dist.is_initialized() or dist.get_rank() == 0):
            logger.warning(
                "LingBot prompt.txt is missing or empty at {}; "
                "proceeding with an empty prompt.",
                prompt_path,
            )
        return prompt

    def _load_default_first_frame_rgb(self) -> np.ndarray:
        first_frame_path = (
            self.config.example_data_dir / self.config.first_frame_filename
        )
        if first_frame_path.exists():
            image_bgr = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(
                    f"Failed to read first frame from {first_frame_path}"
                )
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.config.default_image_url:
            return self._load_remote_first_frame_rgb(self.config.default_image_url)

        return np.full(
            (self.config.video_height, self.config.video_width, 3),
            127,
            dtype=np.uint8,
        )

    def _load_remote_first_frame_rgb(self, image_url: str) -> np.ndarray:
        image_bytes, _ = _read_remote_bytes(
            image_url,
            max_bytes=_MAX_REMOTE_IMAGE_BYTES,
            field_name="Lingbot first-frame image URL",
        )
        return _decode_image_bytes_rgb(
            image_bytes, field_name="Lingbot first-frame image URL"
        )

    def _load_uploaded_first_frame_rgb(self, image_bytes: bytes) -> np.ndarray:
        return _decode_image_bytes_rgb(
            image_bytes, field_name="Uploaded first-frame image"
        )

    def _first_frame_to_tensor(self, image_rgb: np.ndarray) -> torch.Tensor:
        # Bicubic to match the upstream Lingbot World demo / generate_fast.py
        # (which uses ``F.interpolate(mode='bicubic')`` over the ``[-1, 1]``
        # tensor); bilinear here would give a different first-frame VAE latent.
        image_rgb = cv2.resize(
            image_rgb,
            (self.config.video_width, self.config.video_height),
            interpolation=cv2.INTER_CUBIC,
        )
        first_frame_t = (
            torch.from_numpy(image_rgb).to(device=self._device, dtype=torch.bfloat16)
            / 127.5
            - 1.0
        )
        # Lingbot's shipped configs pin ``batch_shape=()`` (single-rollout
        # layout), so the pipeline expects the first frame in shape
        # ``[T=1, C, H, W]``; the leading ``unsqueeze(0)`` lifts ``[C, H, W]``
        # to that ``T=1`` axis the I2V encoder pads/slices against.
        return first_frame_t.permute(2, 0, 1).unsqueeze(0)

    def _prepare_session_input_state(
        self, session_input: LingbotSessionInput | None
    ) -> None:
        prompt = (
            normalize_prompt_text(session_input.prompt)
            if session_input is not None and session_input.prompt is not None
            else self._load_default_prompt()
        )
        if not prompt:
            raise ValueError("Lingbot prompt is empty.")

        if session_input is not None and session_input.first_frame_image_bytes:
            image_rgb = self._load_uploaded_first_frame_rgb(
                session_input.first_frame_image_bytes
            )
        elif (
            session_input is not None
            and session_input.first_frame_remote_payload is not None
        ):
            image_rgb = self._load_uploaded_first_frame_rgb(
                session_input.first_frame_remote_payload.data
            )
        elif session_input is not None and session_input.first_frame_image_url:
            image_rgb = self._load_remote_first_frame_rgb(
                session_input.first_frame_image_url
            )
        else:
            image_rgb = self._load_default_first_frame_rgb()

        self._first_frames = self._first_frame_to_tensor(image_rgb)
        self._prompt = prompt
        self._base_text_embeddings = self._encode_text_embeddings_sync([prompt])

    def _reset_rollout_sync(
        self, session_input: LingbotSessionInput | None = None
    ) -> None:
        if self._pipeline is None or self._model_session is None:
            raise LingbotRuntimeError("Runtime pipeline is not initialized.")

        self._prepare_session_input_state(session_input)
        text_events = (
            session_input.text_events
            if session_input is not None and session_input.text_events is not None
            else self.config.text_events
        )
        self._precompute_event_embeddings_sync(text_events)
        if self._first_frames is None or self._prompt is None:
            raise LingbotRuntimeError("Runtime input state is not initialized.")

        self.pose_integrator = CameraPoseIntegrator()
        self._active_event_id = None
        self._model_session.reset(
            prompt=self._prompt,
            first_frames=self._first_frames,
        )
        # Rebuilt per rollout: the mapping carries the rollout's text-event
        # catalog, base prompt, and pose integrator state.
        self._build_input_layers_sync(text_events)

    def _replace_rollout_text_embeddings(self, text_embeddings: torch.Tensor) -> None:
        if self._pipeline is None or self._model_session is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        self._model_session.replace_text_embeddings(text_embeddings)

    def _validate_event_request(self, *, event_id: str, state: str) -> tuple[str, str]:
        state = state.strip().lower() or "trigger"
        if state in {"clear", "release", "off", "none"}:
            return event_id.strip(), state
        if state not in {"trigger", "hold", "on"}:
            raise ValueError(
                "Event state must be one of trigger, hold, on, clear, release, off."
            )
        event_id = event_id.strip()
        if event_id not in self._event_embeddings:
            supported = ", ".join(sorted(self._event_embeddings))
            raise ValueError(f"Unknown event_id={event_id!r}. Supported: {supported}")
        return event_id, state

    def _trigger_event_sync(
        self,
        *,
        event_id: str,
        state: str = "trigger",
    ) -> dict[str, str | None]:
        event_id, state = self._validate_event_request(event_id=event_id, state=state)
        if state in {"clear", "release", "off", "none"}:
            if self._base_text_embeddings is None:
                raise LingbotRuntimeError("Base prompt embeddings are not ready.")
            self._replace_rollout_text_embeddings(self._base_text_embeddings)
            self._active_event_id = None
            return {"active_event_id": None}
        self._replace_rollout_text_embeddings(self._event_embeddings[event_id])
        self._active_event_id = event_id
        return {"active_event_id": event_id}

    def _close_sync(self) -> None:
        model_session = self._model_session
        pipeline = self._pipeline
        self._model_session = None
        self._pipeline = None
        self._base_intrinsics = None
        self._first_frames = None
        self._prompt = None
        self._base_text_embeddings = None
        self._event_embeddings = {}
        self._active_event_id = None
        if model_session is not None:
            model_session.close()
        if pipeline is not None:
            del pipeline

        if self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[Any],
        frame_times: list[float],
    ) -> StepResult:
        if (
            self._pipeline is None
            or self._model_session is None
            or self._base_intrinsics is None
        ):
            raise LingbotRuntimeError("Runtime is not initialized.")
        step_index = self._runtime_step_index()
        num_frames = int(self._pipeline.get_num_output_frames(step_index))
        if len(frame_times) != num_frames:
            raise LingbotRuntimeError(
                f"Expected {num_frames} frame_times for "
                f"chunk={step_index}, got {len(frame_times)}."
            )
        if not segments:
            raise LingbotRuntimeError(f"Chunk={step_index} received empty segments.")
        pose_segments = cast(list[PoseSegment], segments)
        poses = self.pose_integrator.integrate_chunk(
            segments=pose_segments, frame_times=frame_times
        )
        poses_t = torch.from_numpy(poses).to(device=self._device, dtype=torch.float32)
        poses_t = poses_t.view(num_frames, 4, 4)
        intrinsics_t = self._base_intrinsics.view(1, 4).repeat(num_frames, 1)
        return self._generate_from_camera_inputs(
            poses=poses_t,
            intrinsics=intrinsics_t,
            num_frames=num_frames,
        )

    def _generate_from_camera_inputs(
        self,
        *,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        num_frames: int,
    ) -> StepResult:
        """Generate one chunk from an already-resolved camera trajectory.

        Shared by the segment path and the provider-prepared session path so
        both reach the model through identical conditioning.
        """
        if self._pipeline is None or self._model_session is None:
            raise LingbotRuntimeError("Runtime is not initialized.")

        from lingbot.encoder.camctrl import CamCtrlInput  # noqa: PLC0415

        camctrl_input = CamCtrlInput(
            intrinsics=intrinsics.to(device=self._device, dtype=torch.float32),
            poses=poses.to(device=self._device, dtype=torch.float32),
            world_scale=self._world_scale,
        )
        try:
            result = self._model_session.step(
                camctrl_input,
                metadata={"active_event_id": self._active_event_id},
            )
        except RuntimeError as exc:
            raise LingbotRuntimeError(str(exc)) from exc
        return result

    def _step_sync(self, inputs: InferenceInput) -> StepResult:
        """Generate one chunk from mapped model inputs."""
        if self._pipeline is None or self._model_session is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        num_frames = self._model_session.next_num_frames()
        self._apply_conditioning_update_sync(inputs)
        poses = _require_camera_tensor(
            inputs, FIELD_CAMERA_TRAJECTORY, expected_shape=(num_frames, 4, 4)
        )
        intrinsics = _require_camera_tensor(
            inputs, FIELD_CAMERA_INTRINSICS, expected_shape=(num_frames, 4)
        )
        return self._generate_from_camera_inputs(
            poses=poses,
            intrinsics=intrinsics,
            num_frames=num_frames,
        )

    def _apply_conditioning_update_sync(self, inputs: InferenceInput) -> None:
        """Apply a text-event prompt swap requested by the mapping."""
        prompt = inputs.global_conditioning.get("prompt")
        if prompt is None or prompt == self._prompt:
            return
        embeddings = self._prompt_embeddings.get(prompt)
        if embeddings is None:
            embeddings = self._encode_text_embeddings_sync([prompt])
        self._replace_rollout_text_embeddings(embeddings)
        self._prompt = prompt
        self._active_event_id = next(
            (
                event_id
                for event_id, tensor in self._event_embeddings.items()
                if tensor is embeddings
            ),
            None,
        )


def _require_camera_tensor(
    inputs: InferenceInput,
    name: str,
    *,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    """Return one required per-step camera tensor, shape-checked."""
    if name not in inputs.step:
        raise LingbotRuntimeError(
            f"Lingbot step inputs are missing {name!r}; the selected input "
            f"mapping must produce it for every step."
        )
    value = inputs.step[name]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if tuple(value.shape) != expected_shape:
        raise LingbotRuntimeError(
            f"Lingbot step input {name!r} must have shape {expected_shape}, got "
            f"{tuple(value.shape)}."
        )
    return value


LINGBOT_WEBRTC_SOURCE_SCHEMA = UserInputSchema(
    capabilities=(
        UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"})),
        UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
        UserInputCapability(
            event_type="text_event", payload_fields=frozenset({"event_id"})
        ),
    ),
    description="Lingbot WebRTC data-channel input.",
)


class LingbotWebRTCInferenceSession:
    """``InferenceSession`` view of a live Lingbot WebRTC rollout.

    The rollout itself is owned by :class:`LingbotInferenceRuntime`; this only
    adapts it to the runtime-API stepping surface so the shared manager can
    drive it with provider-prepared inputs.
    """

    def __init__(self, *, runtime: LingbotInferenceRuntime) -> None:
        self._runtime = runtime

    def next_step_request(self) -> StepRequest | None:
        return self._runtime._next_step_request_sync()

    def step(self, inputs: InferenceInput) -> StepResult:
        return self._runtime._step_blocking(inputs)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        raise LingbotRuntimeError(
            "Reset a Lingbot WebRTC rollout through the runtime's session "
            "lifecycle, not through the inference session."
        )

    def close(self) -> None:
        # The runtime outlives the session and is closed by the serve loop.
        return None


def create_lingbot_webrtc_session_manager(
    *,
    runtime: LingbotInferenceRuntime | None = None,
    runtime_config: LingbotRuntimeConfig | None = None,
    fps: int | None = None,
    client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
) -> BaseWebRTCSessionManager[LingbotInferenceRuntime, LingbotRuntimeConfig]:
    """Configure the shared WebRTC manager for the Lingbot runtime."""
    runtime_config = runtime_config or getattr(runtime, "config", None)
    if not isinstance(runtime_config, LingbotRuntimeConfig):
        runtime_config = LingbotRuntimeConfig()
    fps = runtime_config.fps if fps is None else fps
    if fps <= 0:
        raise ValueError("fps must be > 0")
    runtime = runtime or LingbotInferenceRuntime(config=runtime_config)
    return BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=fps,
        identity=runtime_config.config_name,
        busy_message="A Lingbot session is already active.",
        warmup_label="Lingbot WebRTC",
        client_liveness_timeout_s=client_liveness_timeout_s,
    )


class LingbotWebRTCSessionController:
    """Own Lingbot browser inputs and preview data outside the transport manager."""

    def __init__(
        self,
        manager: BaseWebRTCSessionManager[
            LingbotInferenceRuntime,
            LingbotRuntimeConfig,
        ],
    ) -> None:
        self._manager = manager
        self._runtime = manager.runtime
        self._runtime_config = manager.runtime_config

    def _effective_text_events(self) -> tuple[TextEventSpec, ...]:
        pending_session_input = self._manager.pending_session_input
        if (
            pending_session_input is not None
            and pending_session_input.text_events is not None
        ):
            return pending_session_input.text_events
        return self._runtime_config.text_events

    def get_initial_scene(self) -> dict[str, object]:
        pending_input = self._manager.pending_session_input
        text_events = self._effective_text_events()
        prompt = (
            normalize_prompt_text(pending_input.prompt)
            if pending_input is not None and pending_input.prompt is not None
            else self._runtime._load_default_prompt()
        )
        if pending_input is not None and pending_input.first_frame_image_url:
            image_url = pending_input.first_frame_image_url
        else:
            image_url = self._runtime_config.default_image_url
        input_source = "uploaded" if pending_input is not None else "default"
        first_frame_path = (
            self._runtime_config.example_data_dir
            / self._runtime_config.first_frame_filename
        )
        has_first_frame = (
            bool(
                pending_input is not None
                and (
                    pending_input.first_frame_image_bytes
                    or pending_input.first_frame_image_url
                )
            )
            or first_frame_path.exists()
            or bool(self._runtime_config.default_image_url)
        )
        return {
            "first_frame_url": "/api/session/first_frame",
            "image_url": image_url,
            "default_image_url": self._runtime_config.default_image_url,
            "has_first_frame": has_first_frame,
            "prompt": prompt,
            "input_source": input_source,
            "model": self._runtime_config.config_name,
            "capabilities": {"text_events": bool(text_events)},
            "event_catalog": [event.as_public_dict() for event in text_events],
            "active_event_id": getattr(self._runtime, "_active_event_id", None),
            "resolution": {
                "width": self._runtime_config.video_width,
                "height": self._runtime_config.video_height,
            },
        }

    def get_first_frame(self) -> LingbotImagePayload:
        pending_input = self._manager.pending_session_input
        if pending_input is not None and pending_input.first_frame_image_bytes:
            return LingbotImagePayload(
                data=pending_input.first_frame_image_bytes,
                content_type=pending_input.first_frame_content_type,
            )
        if (
            pending_input is not None
            and pending_input.first_frame_remote_payload is not None
        ):
            return pending_input.first_frame_remote_payload
        if pending_input is not None and pending_input.first_frame_image_url:
            image_bytes, content_type = _read_remote_bytes(
                pending_input.first_frame_image_url,
                max_bytes=_MAX_REMOTE_IMAGE_BYTES,
                field_name="Lingbot first-frame image URL",
            )
            return LingbotImagePayload(data=image_bytes, content_type=content_type)

        first_frame_path = (
            self._runtime_config.example_data_dir
            / self._runtime_config.first_frame_filename
        )
        if first_frame_path.exists():
            return LingbotImagePayload(
                data=first_frame_path.read_bytes(),
                content_type=_content_type_for_image_path(first_frame_path),
            )

        image_rgb = self._runtime._load_default_first_frame_rgb()
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Failed to encode default Lingbot first frame.")
        return LingbotImagePayload(data=encoded.tobytes(), content_type="image/jpeg")

    def set_pending_session_input(self, session_input: LingbotSessionInput) -> None:
        if self._manager.has_active_session():
            raise SessionBusyError(
                "Cannot update Lingbot input while a session is active."
            )
        current = self._manager.pending_session_input

        first_frame_image_bytes = (
            current.first_frame_image_bytes if current is not None else None
        )
        first_frame_image_url = (
            current.first_frame_image_url if current is not None else None
        )
        first_frame_content_type = (
            current.first_frame_content_type
            if current is not None
            else session_input.first_frame_content_type
        )
        first_frame_remote_payload = (
            current.first_frame_remote_payload if current is not None else None
        )

        if session_input.first_frame_image_bytes is not None:
            self._runtime._load_uploaded_first_frame_rgb(
                session_input.first_frame_image_bytes
            )
            first_frame_image_bytes = session_input.first_frame_image_bytes
            first_frame_image_url = None
            first_frame_content_type = session_input.first_frame_content_type
            first_frame_remote_payload = None
        elif session_input.first_frame_image_url is not None:
            first_frame_image_url = _validate_remote_url(
                session_input.first_frame_image_url,
                field_name="Lingbot first-frame image URL",
            )
            image_bytes, content_type = _read_remote_bytes(
                first_frame_image_url,
                max_bytes=_MAX_REMOTE_IMAGE_BYTES,
                field_name="Lingbot first-frame image URL",
            )
            self._runtime._load_uploaded_first_frame_rgb(image_bytes)
            first_frame_image_bytes = None
            first_frame_content_type = content_type
            first_frame_remote_payload = LingbotImagePayload(
                data=image_bytes,
                content_type=content_type,
            )

        text_events = (
            normalize_text_events(session_input.text_events)
            if session_input.text_events is not None
            else (current.text_events if current is not None else None)
        )
        self._manager.set_pending_session_input(
            LingbotSessionInput(
                prompt=(
                    normalize_prompt_text(session_input.prompt)
                    if session_input.prompt is not None
                    else (current.prompt if current is not None else None)
                ),
                first_frame_image_bytes=first_frame_image_bytes,
                first_frame_image_url=first_frame_image_url,
                first_frame_content_type=first_frame_content_type,
                first_frame_remote_payload=first_frame_remote_payload,
                text_events=text_events,
            )
        )
