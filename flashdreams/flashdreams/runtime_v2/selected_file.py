# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected-file types, request policy, and size/type checks."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

MAX_SELECTED_FILE_BYTES = 32 * 1024 * 1024
"""Hard ceiling for one selected file. Windows clamp requested budgets to this."""

# ponytail: 8× the per-file ceiling is the batch cap; raise it or
# make it per-request if an app needs a larger set in one pick.
MAX_SELECTED_FILE_BATCH_BYTES = 8 * MAX_SELECTED_FILE_BYTES
"""Hard ceiling for the total bytes of one selected-files request."""


def clamp_selected_file_max_bytes(max_bytes: int | None) -> int:
    """Return a file-size budget that cannot exceed ``MAX_SELECTED_FILE_BYTES``.

    Args:
        max_bytes: Requested budget in bytes, or ``None`` for the ceiling.

    Returns:
        The requested budget when it is below the ceiling, otherwise the ceiling.

    Raises:
        TypeError: ``max_bytes`` is not ``None`` or an integer.
        ValueError: ``max_bytes`` is not positive.
    """
    if max_bytes is None:
        return MAX_SELECTED_FILE_BYTES
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer.")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0.")
    return min(max_bytes, MAX_SELECTED_FILE_BYTES)


def normalize_selected_file_accept(accept: Sequence[str] = ()) -> tuple[str, ...]:
    """Return accepted filename suffixes such as ``.png``.

    Args:
        accept: Suffixes the client may choose. Empty means any type.

    Returns:
        The suffixes as a tuple, unchanged except for the sequence type.

    Raises:
        TypeError: ``accept`` is not a sequence of strings.
        ValueError: A suffix is missing the leading ``.`` or contains a path.
    """
    if isinstance(accept, str) or not isinstance(accept, Sequence):
        raise TypeError("accept must be a sequence of suffixes.")
    suffixes: list[str] = []
    for suffix in accept:
        if not isinstance(suffix, str):
            raise TypeError("accept suffixes must be strings.")
        if (
            not suffix.startswith(".")
            or suffix == "."
            or "/" in suffix
            or "\\" in suffix
        ):
            raise ValueError(
                "accept suffixes must look like ``.png`` and cannot contain a path."
            )
        suffixes.append(suffix)
    return tuple(suffixes)


def selected_file_suffix_allowed(name: str, accept: tuple[str, ...]) -> bool:
    """Return whether ``name`` matches ``accept``.

    Empty ``accept`` allows any name. Comparison is case-insensitive.
    """
    if not accept:
        return True
    lower = name.lower()
    return any(lower.endswith(suffix.lower()) for suffix in accept)


class SelectedFilesStatus(Enum):
    """Outcome of one client file-selector request."""

    OK = "ok"
    """The client chose files that passed type and size checks."""

    CANCELLED = "cancelled"
    """The user dismissed the selector."""

    TOO_LARGE = "too_large"
    """The chosen set exceeded the size budget."""

    DISALLOWED_TYPE = "disallowed_type"
    """A chosen file did not match ``accept``. Checked before size."""

    UNAVAILABLE = "unavailable"
    """The selector could not complete."""


def selected_file_policy_status(
    name: str,
    size: int | None,
    *,
    accept: tuple[str, ...] = (),
    max_bytes: int,
) -> SelectedFilesStatus | None:
    """Return why one chosen file is rejected, or ``None`` if it is allowed.

    Type is checked before size. A too-large disallowed file is
    ``DISALLOWED_TYPE``. Pass ``size=None`` to check only the name.

    Args:
        name: File name to match against ``accept``.
        size: File size in bytes, or ``None`` to skip the size check.
        accept: Filename suffixes such as ``.png``. Empty allows any type.
        max_bytes: Maximum allowed size in bytes.

    Returns:
        ``DISALLOWED_TYPE``, ``TOO_LARGE``, or ``None`` when the file passes.
    """
    if not selected_file_suffix_allowed(name, accept):
        return SelectedFilesStatus.DISALLOWED_TYPE
    if size is not None and size > max_bytes:
        return SelectedFilesStatus.TOO_LARGE
    return None


def selected_files_policy_status(
    files: Sequence[tuple[str, int | None]],
    *,
    accept: tuple[str, ...] = (),
    max_bytes: int,
) -> SelectedFilesStatus | None:
    """Return why a chosen set is rejected, or ``None`` if every file is allowed.

    Type is checked before size, across the whole set.

    Args:
        files: ``(name, size)`` pairs. ``size`` of ``None`` skips that file's
            size check and omits it from the batch total.
        accept: Filename suffixes such as ``.png``. Empty allows any type.
        max_bytes: Maximum allowed size of each file in bytes.

    Returns:
        ``DISALLOWED_TYPE``, ``TOO_LARGE``, or ``None`` when the set passes.
    """
    for name, _size in files:
        if not selected_file_suffix_allowed(name, accept):
            return SelectedFilesStatus.DISALLOWED_TYPE
    total = 0
    for _name, size in files:
        if size is None:
            continue
        if size > max_bytes:
            return SelectedFilesStatus.TOO_LARGE
        total += size
    if total > MAX_SELECTED_FILE_BATCH_BYTES:
        return SelectedFilesStatus.TOO_LARGE
    return None


@dataclass(frozen=True, slots=True)
class SelectedFile:
    """One file chosen by a client file selector."""

    name: str
    """File name shown to the application."""

    data: bytes
    """File contents."""


@dataclass(frozen=True, slots=True)
class FileSelectionRequest:
    """One client file-selector request queued by a UI loop."""

    request_id: str
    """Stable id for which button this request is for. Echoed on the matching event."""

    initial_path: str
    """Directory hint for the selector."""

    accept: tuple[str, ...] = ()
    """Filename suffixes the client may choose, empty for any type."""

    max_bytes: int | None = None
    """Requested size budget per file in bytes, or ``None`` for the ceiling."""

    multiple: bool = False
    """Whether the selector may return more than one file."""
