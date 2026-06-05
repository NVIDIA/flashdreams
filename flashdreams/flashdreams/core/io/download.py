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

"""Atomic URL → local-file downloader with optional content validation.

Designed for runner-side asset fetches (I2V first frames, demo prompts,
etc.) where the alternative -- bundling the file inside the installed
package -- breaks for non-editable / read-only installs. Cosmos and Wan
plugins both pull demo images from public URLs at first run.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger

from flashdreams.core.io.disk import (
    CACHE_MIN_FREE_ENV,
    cache_min_free_bytes,
    ensure_free_disk,
    raise_if_disk_space_error,
)

__all__ = ["download_to_cache"]


def download_to_cache(
    url: str,
    *,
    cache_dir: Path,
    filename: str | None = None,
    validator: Callable[[Path], object] | None = None,
    timeout: float = 30.0,
) -> Path:
    """Atomically download ``url`` into ``cache_dir / filename``.

    Streams the response into a sibling temp file (so ``os.replace`` is
    atomic on the same filesystem), runs ``validator`` against the
    fully-written temp file, then publishes it to the final cache slot
    via ``os.replace``. A network hiccup, HTTP error, or validator
    failure leaves the cache slot empty so the next call retries cleanly
    instead of reusing a half-written file.

    Args:
        url: The ``http(s)://`` URL to fetch.
        cache_dir: Directory to download into; created if missing.
        filename: Destination filename within ``cache_dir``. Defaults to
            the last path component of ``url``, or
            ``"downloaded_file.bin"`` if the URL has no path component.
        validator: Optional callable receiving the temp-file path; must
            raise on invalid content (return value is ignored).
            Typically a decoder check such as
            ``lambda p: media.read_image(str(p))`` so the consumer's
            decode is the source of truth for "valid".
        timeout: Per-request socket timeout in seconds.

    Returns:
        Absolute path to the cached file (existing or newly written).

    Raises:
        RuntimeError: if the download fails, times out, or the validator
            rejects the response. The original exception is chained.
    """
    cache_dir = cache_dir.expanduser()
    filename = filename or Path(urlsplit(url).path).name or "downloaded_file.bin"
    local_path = cache_dir / filename
    if local_path.exists():
        return local_path

    min_bytes = cache_min_free_bytes()
    ensure_free_disk(
        cache_dir,
        required_bytes=min_bytes,
        label="FlashDreams cache download",
        env_vars=("FLASHDREAMS_CACHE_DIR", CACHE_MIN_FREE_ENV),
        settings={"url": url},
    )

    logger.info(f"Downloading {url} -> {local_path}")
    # Stream into a sibling temp file, validate, then atomic-rename;
    # keeping the temp file on the same filesystem makes ``os.replace``
    # atomic, so a partial download cannot poison the cache slot.
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".part", dir=cache_dir
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                shutil.copyfileobj(resp, out)
        if validator is not None:
            # Run the caller-provided decoder check before publishing so
            # the cache only ever contains files the consumer accepts.
            validator(tmp_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise_if_disk_space_error(
            exc,
            path=local_path,
            label="FlashDreams cache download",
            required_bytes=min_bytes,
            env_vars=("FLASHDREAMS_CACHE_DIR", CACHE_MIN_FREE_ENV),
            settings={"url": url},
        )
        raise RuntimeError(
            f"Failed to download {url!r} into {local_path}: {exc}"
        ) from exc
    try:
        os.replace(tmp_path, local_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise_if_disk_space_error(
            exc,
            path=local_path,
            label="FlashDreams cache download",
            required_bytes=min_bytes,
            env_vars=("FLASHDREAMS_CACHE_DIR", CACHE_MIN_FREE_ENV),
            settings={"url": url},
        )
        raise
    return local_path
