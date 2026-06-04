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

"""Disk-space preflights and ENOSPC error formatting."""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

DEFAULT_CACHE_MIN_FREE_GB = 20.0
DEFAULT_OUTPUT_MIN_FREE_GB = 20.0
DEFAULT_TMP_MIN_FREE_GB = 20.0

CACHE_MIN_FREE_ENV = "FLASHDREAMS_MIN_CACHE_FREE_GB"
OUTPUT_MIN_FREE_ENV = "FLASHDREAMS_MIN_OUTPUT_FREE_GB"
TMP_MIN_FREE_ENV = "FLASHDREAMS_MIN_TMP_FREE_GB"

_DISK_ERROR_PHRASES = (
    "no space left on device",
    "errno 28",
    "enospc",
    "disk quota exceeded",
)


class DiskSpaceError(RuntimeError):
    """User-facing disk-space failure with an actionable message."""


def bytes_from_gib(gib: float) -> int:
    """Convert GiB to bytes."""
    return int(gib * 1024 * 1024 * 1024)


def min_free_bytes_from_env(env_var: str, default_gb: float) -> int:
    """Read a non-negative GiB threshold from ``env_var``."""
    raw = os.environ.get(env_var)
    if raw is None or raw == "":
        return bytes_from_gib(default_gb)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{env_var} must be a non-negative number of GiB (got {raw!r})"
        ) from exc
    if value < 0:
        raise ValueError(
            f"{env_var} must be a non-negative number of GiB (got {raw!r})"
        )
    return bytes_from_gib(value)


def default_flashdreams_cache_dir() -> Path:
    """Return the FlashDreams cache root."""
    return Path(
        os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams"))
    )


def default_huggingface_cache_dir() -> Path:
    """Return the Hugging Face Hub cache directory used for model downloads."""
    return Path(HUGGINGFACE_HUB_CACHE).expanduser()


def default_temp_dir() -> Path:
    """Return the configured temporary directory."""
    if "TMPDIR" in os.environ:
        return Path(os.path.expanduser(os.environ["TMPDIR"]))
    return Path(tempfile.gettempdir())


def cache_min_free_bytes(default_gb: float | None = None) -> int:
    """Return the cache-space preflight threshold.

    ``default_gb`` lets model-specific callers use a larger first-run
    requirement while preserving the same ``FLASHDREAMS_MIN_CACHE_FREE_GB``
    override used by the generic cache preflight.
    """
    return min_free_bytes_from_env(
        CACHE_MIN_FREE_ENV,
        DEFAULT_CACHE_MIN_FREE_GB if default_gb is None else default_gb,
    )


def output_min_free_bytes() -> int:
    """Return the output directory preflight threshold."""
    return min_free_bytes_from_env(OUTPUT_MIN_FREE_ENV, DEFAULT_OUTPUT_MIN_FREE_GB)


def tmp_min_free_bytes() -> int:
    """Return the temporary directory preflight threshold."""
    return min_free_bytes_from_env(TMP_MIN_FREE_ENV, DEFAULT_TMP_MIN_FREE_GB)


def _format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _nearest_existing_path(path: Path) -> Path | None:
    path = path.expanduser()
    if path.exists():
        return path
    for parent in path.parents:
        if parent.exists():
            return parent
    return None


def available_bytes(path: str | os.PathLike[str] | None) -> int | None:
    """Return free bytes for ``path`` or its nearest existing parent."""
    if path is None:
        return None
    usage_path = _nearest_existing_path(Path(path))
    if usage_path is None:
        return None
    try:
        return shutil.disk_usage(usage_path).free
    except OSError:
        return None


def _env_lines(env_vars: Sequence[str]) -> list[str]:
    lines: list[str] = []
    for name in env_vars:
        value = os.environ.get(name)
        rendered = value if value not in (None, "") else "<unset>"
        lines.append(f"  {name}: {rendered}")
    return lines


def _settings_lines(settings: Mapping[str, object] | None) -> list[str]:
    if not settings:
        return []
    return [f"  {key}: {value}" for key, value in settings.items()]


def _cause_text(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    for item in _iter_exception_chain(exc):
        if _is_single_disk_space_error(item):
            text = f"{type(item).__name__}: {item}"
            return text if len(text) <= 240 else f"{text[:237]}..."
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 240 else f"{text[:237]}..."


def format_disk_space_message(
    *,
    label: str,
    path: str | os.PathLike[str] | None,
    required_bytes: int | None,
    available: int | None,
    env_vars: Sequence[str] = (),
    settings: Mapping[str, object] | None = None,
    cause: BaseException | None = None,
    preflight: bool = False,
) -> str:
    """Build a concise user-facing disk-space error message."""
    first_line = (
        f"ERROR: Not enough free disk for {label}."
        if preflight
        else f"ERROR: Disk space exhausted while writing {label}."
    )
    lines = [
        first_line,
        f"  Path:     {Path(path).expanduser() if path is not None else 'unknown'}",
        f"  Free:     {_format_bytes(available)}",
        f"  Required: {_format_bytes(required_bytes)}",
    ]
    cause_line = _cause_text(cause)
    if cause_line:
        lines.append(f"  Cause:    {cause_line}")

    setting_lines = _settings_lines(settings)
    env_setting_lines = _env_lines(env_vars)
    if setting_lines or env_setting_lines:
        lines.append("")
        lines.append("Relevant settings:")
        lines.extend(setting_lines)
        lines.extend(env_setting_lines)

    lines.append("")
    hint_targets = [
        name for name in env_vars if not name.startswith("FLASHDREAMS_MIN_")
    ]
    if settings is not None:
        hint_targets.extend(key for key in settings if key.startswith("--"))
    hint = (
        ", ".join(dict.fromkeys(hint_targets))
        if hint_targets
        else "the relevant cache or output directory"
    )
    lines.append(f"Move {hint} to a filesystem with more free space.")
    min_envs = [name for name in env_vars if name.startswith("FLASHDREAMS_MIN_")]
    if min_envs:
        lines.append(
            "Set "
            + " or ".join(f"{name}=0" for name in min_envs)
            + " to skip the corresponding preflight."
        )
    return "\n".join(lines)


def ensure_free_disk(
    path: str | os.PathLike[str],
    *,
    required_bytes: int,
    label: str,
    env_vars: Sequence[str] = (),
    settings: Mapping[str, object] | None = None,
    mkdir: bool = True,
) -> None:
    """Ensure ``path`` has at least ``required_bytes`` free bytes.

    ``required_bytes=0`` disables the threshold check but still creates
    ``path`` when ``mkdir`` is true, matching the env-var override semantics
    used by the setup scripts.
    """
    target = Path(path).expanduser()
    if mkdir:
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise_if_disk_space_error(
                exc,
                path=target,
                label=label,
                required_bytes=required_bytes,
                env_vars=env_vars,
                settings=settings,
            )
            raise

    if required_bytes <= 0:
        return

    free = available_bytes(target)
    if free is None or free >= required_bytes:
        return

    raise DiskSpaceError(
        format_disk_space_message(
            label=label,
            path=target,
            required_bytes=required_bytes,
            available=free,
            env_vars=env_vars,
            settings=settings,
            preflight=True,
        )
    )


def preflight_runtime_write_paths(
    *,
    output_dir: str | os.PathLike[str] | None,
) -> None:
    """Preflight common runtime write-heavy directories."""
    ensure_free_disk(
        default_flashdreams_cache_dir(),
        required_bytes=cache_min_free_bytes(),
        label="FlashDreams cache",
        env_vars=("FLASHDREAMS_CACHE_DIR", CACHE_MIN_FREE_ENV),
    )
    ensure_free_disk(
        default_temp_dir(),
        required_bytes=tmp_min_free_bytes(),
        label="temporary directory",
        env_vars=("TMPDIR", "TEMP", "TMP", TMP_MIN_FREE_ENV),
    )
    if output_dir is not None:
        ensure_free_disk(
            output_dir,
            required_bytes=output_min_free_bytes(),
            label="output directory",
            env_vars=(OUTPUT_MIN_FREE_ENV,),
            settings={"--output-dir": Path(output_dir).expanduser()},
        )


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    stack = [exc]
    seen: set[int] = set()
    out: list[BaseException] = []
    while stack:
        item = stack.pop()
        ident = id(item)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(item)
        cause = item.__cause__
        if cause is not None:
            stack.append(cause)
        elif not item.__suppress_context__ and item.__context__ is not None:
            stack.append(item.__context__)
    return out


def _is_single_disk_space_error(exc: BaseException) -> bool:
    if isinstance(exc, DiskSpaceError):
        return True
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return True
    text = str(exc).lower()
    return any(phrase in text for phrase in _DISK_ERROR_PHRASES)


def is_disk_space_error(exc: BaseException) -> bool:
    """Return whether ``exc`` or its chain indicates disk exhaustion."""
    return any(_is_single_disk_space_error(item) for item in _iter_exception_chain(exc))


def _exception_path(exc: BaseException) -> str | os.PathLike[str] | None:
    for item in _iter_exception_chain(exc):
        if isinstance(item, OSError):
            for attr in ("filename", "filename2"):
                value = getattr(item, attr, None)
                if value:
                    return os.fsdecode(value)
    return None


def disk_space_error_from_exception(
    exc: BaseException,
    *,
    path: str | os.PathLike[str] | None = None,
    label: str = "FlashDreams output",
    required_bytes: int | None = None,
    env_vars: Sequence[str] = (),
    settings: Mapping[str, object] | None = None,
) -> DiskSpaceError | None:
    """Convert ENOSPC-like failures into :class:`DiskSpaceError`."""
    if isinstance(exc, DiskSpaceError):
        return exc
    if not is_disk_space_error(exc):
        return None

    failing_path = path if path is not None else _exception_path(exc)
    return DiskSpaceError(
        format_disk_space_message(
            label=label,
            path=failing_path,
            required_bytes=required_bytes,
            available=available_bytes(failing_path),
            env_vars=env_vars,
            settings=settings,
            cause=exc,
            preflight=False,
        )
    )


def raise_if_disk_space_error(
    exc: BaseException,
    *,
    path: str | os.PathLike[str] | None = None,
    label: str = "FlashDreams output",
    required_bytes: int | None = None,
    env_vars: Sequence[str] = (),
    settings: Mapping[str, object] | None = None,
) -> None:
    """Raise :class:`DiskSpaceError` if ``exc`` looks like disk exhaustion."""
    disk_error = disk_space_error_from_exception(
        exc,
        path=path,
        label=label,
        required_bytes=required_bytes,
        env_vars=env_vars,
        settings=settings,
    )
    if disk_error is not None:
        raise disk_error from exc
