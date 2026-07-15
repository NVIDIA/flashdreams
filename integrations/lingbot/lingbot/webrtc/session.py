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

import asyncio
import http.client
import io
import ipaddress
import re
import socket
import ssl
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from flashdreams.core.distributed.rank_orchestration import (
    RankCoordinator,
    distributed_op,
)
from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.infra.config import derive_config
from flashdreams.serving.webrtc.controls import (
    CameraPoseIntegrator,
    PoseSegment,
)
from flashdreams.serving.webrtc.manager import (
    DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    BaseWebRTCSessionManager,
    ManagedWebRTCSession,
    WebRTCControlSignal,
    WebRTCStepResult,
)
from flashdreams.serving.webrtc.server import SessionBusyError
from lingbot.encoder.utils import preprocess_example_poses

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
    world_scale: float | None = None
    default_intrinsics: tuple[float, float, float, float] | None = None
    default_prompt: str = ""
    """Prompt used when the selected example does not provide ``prompt.txt``."""
    default_image_url: str | None = _DEFAULT_IMAGE_URL
    default_intrinsics_url: str | None = _DEFAULT_INTRINSICS_URL
    default_poses_url: str | None = _DEFAULT_POSES_URL
    warmup_chunks: int = 10
    warmup_timeout_s: float = 600.0

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


class LingbotInferenceRuntime:
    """Single-session Lingbot runtime with action-bound chunk generation."""

    def __init__(self, config: LingbotRuntimeConfig | None = None) -> None:
        self.config = config or LingbotRuntimeConfig()
        self.MASTER_RANK = 0
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

        control_device = torch.device(self.config.device)
        if control_device.type == "cuda" and control_device.index is None:
            control_device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cuda:0"
            )

        self.pose_integrator = CameraPoseIntegrator()
        self.autoregressive_index = 0

        self._device: torch.device | None = None
        self._pipeline: Any | None = None
        self._cache: Any | None = None
        self._base_intrinsics: torch.Tensor | None = None
        self._first_frames: torch.Tensor | None = None
        self._prompt: str | None = None
        self._base_text_embeddings: torch.Tensor | None = None
        self._event_embeddings: dict[str, torch.Tensor] = {}
        self._active_event_id: str | None = None
        self._world_scale = 1.0
        self._closed = False

        self._step_lock = asyncio.Lock()
        self.rank_coordinator = RankCoordinator(
            device=control_device,
            signal_type=WebRTCControlSignal,
            is_master=self.is_master,
            master_rank=self.MASTER_RANK,
        )
        self.rank_coordinator.register_distributed_ops(self)

    @property
    def is_master(self) -> bool:
        return self.rank == self.MASTER_RANK

    def wait_for_termination(self) -> None:
        self.rank_coordinator.worker_loop(exit_signal=WebRTCControlSignal.EXIT)

    def send_exit_signal(self) -> None:
        if self.is_master:
            self.rank_coordinator.send_exit(exit_signal=WebRTCControlSignal.EXIT)

    async def initialize(self) -> None:
        if self._pipeline is not None:
            return
        await asyncio.to_thread(self._initialize_sync_all_ranks)

    async def reset_for_new_session(
        self, session_input: LingbotSessionInput | None = None
    ) -> None:
        if self._closed:
            raise LingbotRuntimeError("Runtime is closed.")
        if self._pipeline is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        await asyncio.to_thread(self._reset_rollout_sync_all_ranks, session_input)

    async def close(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._close_sync_all_ranks)

    async def trigger_event(
        self, *, event_id: str, state: str = "trigger"
    ) -> dict[str, str | None]:
        """Activate or clear a precomputed text event for subsequent chunks."""
        if self._closed:
            raise LingbotRuntimeError("Runtime is closed.")
        if self._pipeline is None or self._cache is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        event_id, state = self._validate_event_request(event_id=event_id, state=state)
        async with self._step_lock:
            if self._closed:
                raise LingbotRuntimeError("Runtime is closed.")
            if self._pipeline is None or self._cache is None:
                raise LingbotRuntimeError("Runtime is not initialized.")
            return await asyncio.to_thread(
                self._trigger_event_sync_all_ranks,
                event_id,
                state,
            )

    async def generate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> WebRTCStepResult:
        """Generate one autoregressive chunk from a piecewise-constant timeline.

        Args:
            segments: Piecewise-constant keyboard-state segments
                covering the chunk's virtual-time window; produced by
                :meth:`KeyboardResampler.sample_chunk`.
            frame_times: Virtual times at which to sample the camera
                pose; must have length equal to
                :meth:`peek_next_chunk_num_frames` at call time.

        Returns:
            :class:`WebRTCStepResult` carrying the produced video chunk
            and the post-generation pipeline stats.

        Raises:
            LingbotRuntimeError: Runtime is closed or not initialized.
        """
        if self._closed:
            raise LingbotRuntimeError("Session is closed.")
        if self._pipeline is None or self._cache is None:
            raise LingbotRuntimeError("Runtime is not initialized.")

        async with self._step_lock:
            if self._closed:
                raise LingbotRuntimeError("Session is closed.")
            return await asyncio.to_thread(
                self._generate_chunk_sync_all_ranks, segments, frame_times
            )

    def peek_next_chunk_num_frames(self) -> int:
        """Return the number of frames the next chunk's pipeline call will emit.

        Master-only read with no distributed broadcast; safe to call from
        the master rank's asyncio event loop to size the resampler's
        per-chunk request.
        """
        if self._pipeline is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        return int(self._pipeline.get_num_output_frames(self.autoregressive_index))

    # Arbitrary index well past the AR-step transient; for the Wan/lingbot
    # pipelines used here the per-step count is constant for any index
    # ``>= 1`` (only AR 0 emits fewer frames due to causal first-frame
    # padding). Picking a large number is a robust way to ask "what is
    # the steady-state chunk size?" without leaning on the exact
    # boundary of that transient.
    _STEADY_STATE_AR_PROBE_INDEX: int = 1000

    def peek_steady_chunk_num_frames(self) -> int:
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

    @distributed_op(WebRTCControlSignal.INITIALIZE)
    def _initialize_sync_all_ranks(self) -> None:
        self._initialize_sync()

    @distributed_op(WebRTCControlSignal.RESET_SESSION)
    def _reset_rollout_sync_all_ranks(
        self, session_input: LingbotSessionInput | None = None
    ) -> None:
        self._reset_rollout_sync(session_input=session_input)

    @distributed_op(WebRTCControlSignal.ACTION_STEP)
    def _generate_chunk_sync_all_ranks(
        self,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> WebRTCStepResult:
        return self._generate_one_chunk_sync(segments=segments, frame_times=frame_times)

    @distributed_op(WebRTCControlSignal.EVENT)
    def _trigger_event_sync_all_ranks(
        self,
        event_id: str,
        state: str = "trigger",
    ) -> dict[str, str | None]:
        return self._trigger_event_sync(event_id=event_id, state=state)

    @distributed_op(WebRTCControlSignal.CLOSE)
    def _close_sync_all_ranks(self) -> None:
        self._close_sync()

    def _initialize_sync(self) -> None:
        if self._pipeline is not None:
            return

        pipeline_configs = _pipeline_configs()
        if self.config.config_name not in pipeline_configs:
            supported = ", ".join(sorted(pipeline_configs))
            raise ValueError(
                f"Unknown config_name={self.config.config_name!r}. "
                f"Supported: {supported}"
            )

        self._device = torch.device(self.config.device)
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
            base_config=pipeline_configs[self.config.config_name],
            enable_sync_and_profile=True,
            diffusion_model=dict(
                seed=rollout_seed,
                transformer=dict(compile_network=self.config.compile_network),
            ),
        )
        self._pipeline = pipeline_config.setup().to(device=self._device)
        self._reset_rollout_sync()

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

    def _build_base_intrinsics(self) -> torch.Tensor:
        if self._device is None:
            raise LingbotRuntimeError("Runtime device is not initialized.")
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
        if self._device is None:
            raise LingbotRuntimeError("Runtime device is not initialized.")
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
        if self._pipeline is None:
            raise LingbotRuntimeError("Runtime pipeline is not initialized.")

        if self._cache is not None:
            del self._cache
            self._cache = None

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
        self.autoregressive_index = 0
        self._active_event_id = None
        self._cache = self._pipeline.initialize_cache(
            text=[self._prompt],
            image=self._first_frames,
        )

    def _replace_rollout_text_embeddings(self, text_embeddings: torch.Tensor) -> None:
        if self._pipeline is None or self._cache is None:
            raise LingbotRuntimeError("Runtime is not initialized.")
        transformer = self._pipeline.diffusion_model.transformer
        replace_text_embeddings = getattr(transformer, "replace_text_embeddings", None)
        if not callable(replace_text_embeddings):
            raise LingbotRuntimeError(
                "Current pipeline does not support runtime text-event swapping."
            )
        replace_text_embeddings(self._cache.transformer_cache, text_embeddings)

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
        cache = self._cache
        pipeline = self._pipeline
        self._cache = None
        self._pipeline = None
        self._base_intrinsics = None
        self._first_frames = None
        self._prompt = None
        self._base_text_embeddings = None
        self._event_embeddings = {}
        self._active_event_id = None

        if cache is not None:
            del cache
        if pipeline is not None:
            del pipeline

        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> WebRTCStepResult:
        if (
            self._pipeline is None
            or self._cache is None
            or self._base_intrinsics is None
        ):
            raise LingbotRuntimeError("Runtime is not initialized.")
        if self._device is None:
            raise LingbotRuntimeError("Runtime device is not initialized.")

        num_frames = int(
            self._pipeline.get_num_output_frames(self.autoregressive_index)
        )
        if len(frame_times) != num_frames:
            raise LingbotRuntimeError(
                f"Expected {num_frames} frame_times for "
                f"chunk={self.autoregressive_index}, got {len(frame_times)}."
            )
        if not segments:
            raise LingbotRuntimeError(
                f"Chunk={self.autoregressive_index} received empty segments."
            )
        poses = self.pose_integrator.integrate_chunk(
            segments=segments, frame_times=frame_times
        )
        poses_t = torch.from_numpy(poses).to(device=self._device, dtype=torch.float32)
        poses_t = poses_t.view(num_frames, 4, 4)
        intrinsics_t = self._base_intrinsics.view(1, 4).repeat(num_frames, 1)

        from lingbot.encoder.camctrl import CamCtrlInput  # noqa: PLC0415

        camctrl_input = CamCtrlInput(
            intrinsics=intrinsics_t,
            poses=poses_t,
            world_scale=self._world_scale,
        )
        video_chunk = self._pipeline.generate(
            autoregressive_index=self.autoregressive_index,
            cache=self._cache,
            input=camctrl_input,
        )
        stats = self._pipeline.finalize(self.autoregressive_index, self._cache)

        result = WebRTCStepResult(
            chunk_index=self.autoregressive_index,
            num_frames=num_frames,
            video_chunk=video_chunk.detach().cpu(),
            stats=stats,
        )
        self.autoregressive_index += 1
        return result


_ManagedLingbotSession = ManagedWebRTCSession


class LingbotWebRTCSessionManager(BaseWebRTCSessionManager):
    """Owns one active WebRTC session and forwards actions into Lingbot runtime."""

    _busy_message = "A Lingbot session is already active."
    _warmup_label = "Lingbot WebRTC"
    _runtime_error_types = (LingbotRuntimeError,)

    def __init__(
        self,
        *,
        runtime_config: LingbotRuntimeConfig | None = None,
        fps: int = 16,
        client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        runtime_config = runtime_config or LingbotRuntimeConfig()
        super().__init__(
            runtime=LingbotInferenceRuntime(config=runtime_config),
            runtime_config=runtime_config,
            fps=fps,
            client_liveness_timeout_s=client_liveness_timeout_s,
        )
        self._pending_session_input: LingbotSessionInput | None = None

    def _model_name(self) -> str:
        return self.runtime_config.config_name

    def _chunk_done_extra(self) -> dict[str, object]:
        return {"active_event_id": getattr(self._runtime, "_active_event_id", None)}

    def _peek_pending_session_input(self) -> LingbotSessionInput | None:
        return self._pending_session_input

    def _clear_pending_session_input(self) -> None:
        self._pending_session_input = None

    async def _reset_runtime_for_session(
        self, session_input: LingbotSessionInput | None
    ) -> None:
        await self._runtime.reset_for_new_session(session_input=session_input)

    def _effective_text_events(self) -> tuple[TextEventSpec, ...]:
        if (
            self._pending_session_input is not None
            and self._pending_session_input.text_events is not None
        ):
            return self._pending_session_input.text_events
        return self.runtime_config.text_events

    def get_initial_scene(self) -> dict[str, object]:
        pending_input = self._pending_session_input
        text_events = self._effective_text_events()
        prompt = (
            normalize_prompt_text(pending_input.prompt)
            if pending_input is not None and pending_input.prompt is not None
            else self._runtime._load_default_prompt()
        )
        if pending_input is not None and pending_input.first_frame_image_url:
            image_url = pending_input.first_frame_image_url
        else:
            image_url = self.runtime_config.default_image_url
        input_source = "uploaded" if pending_input is not None else "default"
        first_frame_path = (
            self.runtime_config.example_data_dir
            / self.runtime_config.first_frame_filename
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
            or bool(self.runtime_config.default_image_url)
        )
        return {
            "first_frame_url": "/api/session/first_frame",
            "image_url": image_url,
            "default_image_url": self.runtime_config.default_image_url,
            "has_first_frame": has_first_frame,
            "prompt": prompt,
            "input_source": input_source,
            "model": self.runtime_config.config_name,
            "capabilities": {"text_events": bool(text_events)},
            "event_catalog": [event.as_public_dict() for event in text_events],
            "active_event_id": getattr(self._runtime, "_active_event_id", None),
            "resolution": {
                "width": self.runtime_config.video_width,
                "height": self.runtime_config.video_height,
            },
        }

    def get_first_frame(self) -> LingbotImagePayload:
        pending_input = self._pending_session_input
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
            self.runtime_config.example_data_dir
            / self.runtime_config.first_frame_filename
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
        if self.has_active_session():
            raise SessionBusyError(
                "Cannot update Lingbot input while a session is active."
            )
        current = self._pending_session_input

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
        self._pending_session_input = LingbotSessionInput(
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
