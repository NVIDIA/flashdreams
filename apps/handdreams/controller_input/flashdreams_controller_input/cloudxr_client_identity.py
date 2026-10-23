# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection-scoped IWER identification for the pinned Isaac web client."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Self
from urllib.parse import urlsplit

_METRICS_CADENCE = "flashdreams_input_runtime"

# The SDK sets xrDevice only after successfully installing IWER. Session storage
# and browser/device names can survive a reconnect and cannot identify the runtime.
_CLIENT_SCRIPT = b"""<script>
(() => {
  const send = WebSocket.prototype.send;
  WebSocket.prototype.send = function(data) {
    let message;
    try {
      if (new URL(this.url).pathname === '/oob/v1/ws' && typeof data === 'string') {
        message = JSON.parse(data);
      }
    } catch (_) {}
    if (message?.type === 'streamStatus') {
      const emulated = message.payload?.streaming === true &&
        typeof window.IWER?.XRDevice === 'function' &&
        window.xrDevice instanceof window.IWER.XRDevice;
      send.call(this, JSON.stringify({type: 'clientMetrics', payload: {
        cadence: 'flashdreams_input_runtime', t: Date.now(),
        metrics: {iwer: emulated ? 1 : 0}
      }}));
    }
    return send.call(this, data);
  };
})();
</script>"""


def _stream_identity(snapshot: dict[str, Any]) -> tuple[str | None, bool]:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("headsets"), list):
        return None, False
    if any(not isinstance(peer, dict) for peer in snapshot["headsets"]):
        return None, False
    peers = [
        peer
        for peer in snapshot["headsets"]
        if isinstance(peer, dict)
        and peer.get("connected") is True
        and peer.get("streaming") is True
    ]
    if len(peers) != 1:
        return None, False
    peer = peers[0]
    client_id, since = peer.get("clientId"), peer.get("streamingSince")
    if (
        not isinstance(client_id, str)
        or not client_id
        or type(since) is not int
        or since < 0
        or len(f"{client_id}:{since}") > 512
    ):
        return None, False
    metrics = peer.get("metricsByCadence", {})
    for key in (_METRICS_CADENCE, "metrics", "iwer"):
        metrics = metrics.get(key) if isinstance(metrics, dict) else None
    iwer = metrics
    return f"{client_id}:{since}", type(iwer) in (int, float) and iwer == 1


class CloudXRClientIdentity:
    """Read the client-declared IWER mode of one unique streaming peer.

    The pinned SDK's OOB hub supplies connection identity and numeric metrics.
    The hosted HTML response adds the runtime metric without modifying cached
    assets. Unknown and ambiguous peers retain guided setup. Poll failures retain
    the last confirmed connection; lifecycle hooks invalidate actual handoffs.
    """

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        self._identity: tuple[str | None, bool] = (None, False)
        self._active_ws: Any = None
        self._mutation_epoch = 0
        self._pending_mutations = 0
        self._task: asyncio.Task | None = None
        self._factory: Any = None
        self._wss: Any = None
        self._hub_originals: tuple[Any, Any, Any, Any] | None = None

    def __enter__(self) -> Self:
        if self._enabled:
            from isaacteleop.cloudxr import wss  # ty: ignore[unresolved-import]

            self._wss = wss
            self._factory = wss._make_http_handler
            wss._make_http_handler = self._make_http_handler
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
        if self._hub_originals is not None:
            hub, register, dispatch, connection = self._hub_originals
            hub._handle_register = register
            hub._dispatch_headset = dispatch
            hub.handle_connection = connection
            self._hub_originals = None
        if self._wss is not None:
            self._wss._make_http_handler = self._factory

    def close(self) -> None:
        """Invalidate identity and cancel monitoring before the SDK closes its loop."""
        with self._lock:
            self._enabled = False
            self._identity = (None, False)
            self._active_ws = None
            task = self._task
        if task is not None and not task.get_loop().is_closed():
            try:
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # The SDK may close its loop concurrently during shutdown.

    def sample(self) -> tuple[str | None, bool]:
        """Return confirmed connection identity without blocking controller sampling."""
        with self._lock:
            if not self._enabled:
                return None, False
            return self._identity

    async def _refresh(self, hub: Any) -> None:
        with self._lock:
            if not self._enabled or self._pending_mutations:
                return
            epoch = self._mutation_epoch
        snapshot = await hub.get_snapshot()
        identity = _stream_identity(snapshot)
        with self._lock:
            if (
                self._enabled
                and not self._pending_mutations
                and epoch == self._mutation_epoch
            ):
                self._identity = identity
                # The pinned SDK owns websocket membership, including removal
                # before handle_connection returns to our disconnect wrapper.
                peers = getattr(hub, "_headsets", None)
                matches = (
                    [
                        ws
                        for ws, peer in peers.items()
                        if identity[0] is not None
                        and getattr(peer, "client_id", None)
                        == identity[0].rsplit(":", 1)[0]
                        and getattr(peer, "streaming", None) is True
                    ]
                    if isinstance(peers, dict)
                    else []
                )
                self._active_ws = matches[0] if len(matches) == 1 else None

    async def _watch(self, hub: Any) -> None:
        failed = False
        try:
            while self._enabled:
                try:
                    await self._refresh(hub)
                except Exception:  # noqa: BLE001 - retry without interrupting controller input
                    if not failed:
                        logging.getLogger(__name__).warning(
                            "Client runtime identity unavailable; retrying without "
                            "changing the last confirmed connection"
                        )
                    failed = True
                else:
                    failed = False
                await asyncio.sleep(0.05)
        finally:
            with self._lock:
                self._identity = (None, False)
                self._active_ws = None

    def _observe_hub(self, hub: Any) -> None:
        register = hub._handle_register
        dispatch = hub._dispatch_headset
        connection = hub.handle_connection
        self._hub_originals = hub, register, dispatch, connection

        async def refresh() -> None:
            try:
                await self._refresh(hub)
            except Exception:  # noqa: BLE001 - identity must not stop the SDK hub
                logging.getLogger(__name__).warning(
                    "Client runtime identity unavailable; monitoring will retry"
                )

        async def mutate(operation: Any, *args: Any, invalidate: bool) -> Any:
            with self._lock:
                self._mutation_epoch += 1
                self._pending_mutations += 1
                if invalidate:
                    self._identity = (None, False)
            try:
                return await operation(*args)
            finally:
                with self._lock:
                    self._pending_mutations -= 1
                await refresh()

        async def handle_register(ws: Any, client_id: str, payload: dict) -> Any:
            peers = getattr(hub, "_headsets", None)
            # A new SDK registration is idle; only replacement of an existing
            # streaming socket can affect the current controller connection.
            invalidate = not isinstance(peers, dict) or (
                ws in peers and getattr(peers[ws], "streaming", None) is not False
            )
            return await mutate(register, ws, client_id, payload, invalidate=invalidate)

        async def dispatch_headset(ws: Any, kind: str, payload: dict) -> None:
            if kind == "streamStatus" or (
                kind == "clientMetrics" and payload.get("cadence") == _METRICS_CADENCE
            ):
                peers = getattr(hub, "_headsets", None)
                peer = peers.get(ws) if isinstance(peers, dict) else None
                streaming = getattr(peer, "streaming", None)
                invalidate = True
                if type(streaming) is bool:
                    if kind == "streamStatus":
                        invalidate = bool(payload.get("streaming", False)) != streaming
                    elif not streaming:
                        invalidate = False
                    else:
                        metrics = payload.get("metrics")
                        iwer = (
                            metrics.get("iwer") if isinstance(metrics, dict) else None
                        )
                        # Match the SDK's numeric metric coercion, including bools.
                        emulated = isinstance(iwer, (int, float)) and iwer == 1
                        with self._lock:
                            invalidate = (
                                ws is not self._active_ws
                                or emulated != self._identity[1]
                            )
                await mutate(dispatch, ws, kind, payload, invalidate=invalidate)
            else:
                await dispatch(ws, kind, payload)

        async def handle_connection(ws: Any) -> None:
            try:
                await connection(ws)
            finally:
                # The SDK removes the disconnected peer before returning here.
                with self._lock:
                    self._mutation_epoch += 1
                    if self._active_ws is None or ws is self._active_ws:
                        self._identity = (None, False)
                        self._active_ws = None
                await refresh()

        hub._handle_register = handle_register
        hub._dispatch_headset = dispatch_headset
        hub.handle_connection = handle_connection

    def _make_http_handler(
        self,
        backend_host: str,
        backend_port: int,
        hub: Any = None,
        static_dir: Any = None,
    ) -> Any:
        handler = self._factory(
            backend_host, backend_port, hub=hub, static_dir=static_dir
        )
        with self._lock:
            if hub is not None and self._enabled:
                self._observe_hub(hub)
                self._task = asyncio.create_task(self._watch(hub))

        async def handle(connection: Any, request: Any) -> Any:
            response = await handler(connection, request)
            if (
                static_dir is not None
                and response is not None
                and response.status_code == 200
                and urlsplit(request.path).path
                in ("/client", "/client/", "/client/index.html")
            ):
                # Inject before the app bundle; never rewrite a minified SDK asset.
                response.body = response.body.replace(
                    b"<head>", b"<head>" + _CLIENT_SCRIPT, 1
                )
                if "Content-Length" in response.headers:
                    del response.headers["Content-Length"]
                    response.headers["Content-Length"] = str(len(response.body))
            return response

        return handle
