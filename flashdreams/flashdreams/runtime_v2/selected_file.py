# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected-file types, request policy, and size/type checks."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

DEFAULT_MAX_FILE_BYTES = 1024 * 1024 * 1024
"""Default ``max_file_bytes`` (1 GiB): the largest single file a request accepts."""

DEFAULT_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
"""Default ``max_total_bytes`` (1 GiB): the most bytes all chosen files add up to."""

DEFAULT_MAX_FILE_COUNT = 1
"""Default ``max_file_count``: one file, unless the app asks for more."""


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
    """A file was larger than ``max_file_bytes``, the files together were larger
    than ``max_total_bytes``, or more than ``max_file_count`` files were chosen."""

    DISALLOWED_TYPE = "disallowed_type"
    """A chosen file did not match ``accept``. Checked before size."""

    BUSY = "busy"
    """Another selector was already open, so this request was not shown. The
    open selector keeps going and answers its own request."""

    UNAVAILABLE = "unavailable"
    """No pick was made: the client has no picker or left, the picker failed,
    or a chosen file was unreadable."""


@dataclass(frozen=True, slots=True)
class SelectedFile:
    """One file chosen by a client file selector."""

    name: str
    """Base name of the chosen file, without any host path."""

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

    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    """Largest single file in bytes."""

    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    """Most bytes all chosen files may add up to."""

    max_file_count: int = DEFAULT_MAX_FILE_COUNT
    """Most files the selector may return. Above one, the user may pick several."""


def selected_files_policy_status(
    files: Sequence[tuple[str, int | None]],
    request: FileSelectionRequest,
) -> SelectedFilesStatus | None:
    """Return why a chosen set breaks ``request``, or ``None`` if it passes.

    Type is checked before size and count, across the whole set.

    Args:
        files: ``(name, size)`` pairs. ``size`` of ``None`` skips that file's
            size check and omits it from the total.
        request: The request whose ``accept`` and limits apply.

    Returns:
        ``DISALLOWED_TYPE``, ``TOO_LARGE``, or ``None`` when the set passes.
    """
    for name, _size in files:
        if not selected_file_suffix_allowed(name, request.accept):
            return SelectedFilesStatus.DISALLOWED_TYPE
    if len(files) > request.max_file_count:
        return SelectedFilesStatus.TOO_LARGE
    total = 0
    for _name, size in files:
        if size is None:
            continue
        if size > request.max_file_bytes:
            return SelectedFilesStatus.TOO_LARGE
        total += size
    if total > request.max_total_bytes:
        return SelectedFilesStatus.TOO_LARGE
    return None
