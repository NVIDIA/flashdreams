# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission policy for the standalone V2 browser server."""

import errno
import hmac
import ipaddress
import os
import secrets
import stat
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web
from yarl import URL


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_token(token: str) -> None:
    if (
        not 32 <= len(token) <= 256
        or not token.isascii()
        or not all(character.isalnum() or character in "-_" for character in token)
    ):
        raise ValueError("Use a 32–256 character random URL-safe WebRTC token.")


def _token_from_file() -> str | None:
    path = os.environ.get("FLASHDREAMS_WEBRTC_TOKEN_FILE")
    return None if path is None else _read_token_file(path)


def _read_token_file(path: str | os.PathLike) -> str:
    # Refuse shared, substituted, or special files without printing their contents.
    # Windows has neither flag; the user's profile ACLs protect the default file.
    flags = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | flags)
    except OSError as error:
        reason = errno.errorcode.get(error.errno or 0, "OSError")
        raise ValueError(
            f"WebRTC token file {path} could not be opened safely: {reason}"
        ) from None
    with os.fdopen(descriptor, "r", encoding="ascii") as source:
        metadata = os.fstat(source.fileno())
        # Windows reports synthetic owner and mode bits; ACLs govern access there.
        if not stat.S_ISREG(metadata.st_mode) or (
            os.name != "nt"
            and (
                metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            )
        ):
            raise ValueError(
                f"WebRTC token file {path} must be a regular file owned by this "
                f"user with mode 600: chmod 600 {path}"
            )
        value = source.read(258)
        if len(value) > 257:
            raise ValueError("WebRTC token file is too long.")
        token = value.strip()
    _validate_token(token)
    return token


def pairing_file() -> Path:
    """Return the operator's token file, else this user's default credential."""
    configured = os.environ.get("FLASHDREAMS_WEBRTC_TOKEN_FILE")
    if configured is not None:
        return Path(configured)
    config = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(config, "flashdreams", "webrtc-token")


def load_pairing_token() -> str:
    """Read the pairing file, creating the default credential on first use.

    Call only where the server runs. The default is reused across launches, so
    a paired browser tab survives a restart; delete the file to rotate it.
    """
    path = pairing_file()
    if "FLASHDREAMS_WEBRTC_TOKEN_FILE" not in os.environ and not os.path.lexists(path):
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Link a complete owner-only file into place: a crash or a second
            # first launch never leaves an empty credential behind.
            descriptor, staging = tempfile.mkstemp(dir=path.parent, prefix=".pair-")
            try:
                with os.fdopen(descriptor, "w", encoding="ascii") as target:
                    target.write(secrets.token_urlsafe(32) + "\n")
                os.link(staging, path)
            finally:
                os.unlink(staging)
        except FileExistsError:
            pass
        except OSError as error:
            reason = errno.errorcode.get(error.errno or 0, "OSError")
            raise ValueError(
                f"WebRTC pairing credential {path} could not be created: {reason}; "
                "set FLASHDREAMS_WEBRTC_TOKEN_FILE to a private file."
            ) from None
    return _read_token_file(path)


def check_launch(host: str, *, no_login: bool) -> None:
    """Validate a launch's admission policy on any rank without reading secrets."""
    if no_login:
        configured = any(
            os.environ.get(name) is not None
            for name in (
                "FLASHDREAMS_WEBRTC_TOKEN_FILE",
                "FLASHDREAMS_WEBRTC_ALLOWED_ORIGINS",
            )
        )
        if configured or not _loopback(host):
            raise ValueError(
                "--no-login serves only a loopback --host without "
                "FLASHDREAMS_WEBRTC_TOKEN_FILE or FLASHDREAMS_WEBRTC_ALLOWED_ORIGINS."
            )
    else:
        # A placeholder bearer checks Host/Origin policy; the real one stays unread.
        BrowserSessionAuth(host, auth_token="0" * 32)


class BrowserSessionAuth:
    """Authorize one reusable owner and fence requests across credential rotation.

    Mutations run on the server event loop. A generation captured before an
    await must still be current before the request can affect session state.
    """

    def __init__(
        self,
        host: str,
        *,
        auth_token: str | None = None,
        allowed_origins: tuple[str, ...] | None = None,
    ) -> None:
        self._host = host
        self._token = auth_token if auth_token is not None else _token_from_file()
        if self._token is not None:
            _validate_token(self._token)
        if allowed_origins is None:
            configured = os.environ.get("FLASHDREAMS_WEBRTC_ALLOWED_ORIGINS")
            allowed_origins = (
                tuple(part.strip() for part in configured.split(","))
                if configured is not None
                else ()
            )
        for origin in allowed_origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
                or origin != f"{parsed.scheme}://{parsed.netloc}"
                or (parsed.scheme == "http" and not _loopback(parsed.hostname))
            ):
                raise ValueError(
                    "WebRTC origins require HTTPS except for loopback, and no paths."
                )
            try:
                parsed.port
            except ValueError as error:
                raise ValueError("Invalid WebRTC origin port.") from error
        self._origins = frozenset(
            str(URL(origin).origin()) for origin in allowed_origins
        )
        self._local = _loopback(host) and not self._origins and self._token is None
        if not self._local and self._token is None:
            raise ValueError("Nonlocal WebRTC requires FLASHDREAMS_WEBRTC_TOKEN_FILE.")
        if not _loopback(host) and not self._origins:
            raise ValueError(
                "Nonlocal WebRTC requires FLASHDREAMS_WEBRTC_ALLOWED_ORIGINS."
            )
        self.generation = 0

    @property
    def no_login(self) -> bool:
        """Whether direct loopback peers are admitted without a bearer."""
        return self._local

    @property
    def token(self) -> str | None:
        """Return the current owner bearer for the operator's pairing link."""
        return self._token

    @property
    def browser_origin(self) -> str | None:
        """Return a configured browser origin for the operator's connection hint."""
        return min(self._origins, default=None)

    def rotate(self, token: str | None) -> None:
        """Revoke all previous admissions and optionally install a new bearer."""
        if token is not None:
            _validate_token(token)
        self.generation += 1
        self._token = token
        # Revocation never restores the initial no-login policy.
        self._local = False

    def check_generation(self, generation: int) -> None:
        """Refuse an authorization that was revoked during an await."""
        if generation != self.generation:
            raise web.HTTPUnauthorized(
                reason="WebRTC authorization changed; pair again."
            )

    def authorize(self, request: web.Request, port: int, *, protected: bool) -> int:
        """Validate the direct Host, browser Origin, and optional owner bearer."""
        suffix = "" if port == 80 else f":{port}"
        bound_host = f"[{self._host}]" if ":" in self._host else self._host
        local_origins = {
            f"http://{host}{suffix}"
            for host in ("127.0.0.1", "localhost", "[::1]", bound_host)
        }
        origins = self._origins or local_origins
        host = request.headers.get("Host", "")
        matching = {origin for origin in origins if urlsplit(origin).netloc == host}
        if not matching:
            raise web.HTTPForbidden(reason="Unapproved WebRTC Host.")
        origin = request.headers.get("Origin")
        if origin is not None and origin not in matching:
            raise web.HTTPForbidden(reason="Unapproved WebRTC Origin.")
        if protected and request.headers.get("Sec-Fetch-Site") == "cross-site":
            raise web.HTTPForbidden(reason="Cross-site WebRTC request refused.")
        if not protected:
            return self.generation
        if self._local:
            peer = (
                request.transport.get_extra_info("peername")
                if request.transport
                else None
            )
            if (
                peer is None
                or not _loopback(peer[0])
                or any(
                    header.lower() in {"forwarded", "x-real-ip"}
                    or header.lower().startswith("x-forwarded-")
                    for header in request.headers
                )
            ):
                raise web.HTTPForbidden(reason="Direct loopback access required.")
        elif self._token is None or not hmac.compare_digest(
            request.headers.get("Authorization", "").encode("utf-8", "surrogateescape"),
            f"Bearer {self._token}".encode("ascii"),
        ):
            raise web.HTTPUnauthorized(
                reason="Pair this browser with the operator's WebRTC token."
            )
        return self.generation
