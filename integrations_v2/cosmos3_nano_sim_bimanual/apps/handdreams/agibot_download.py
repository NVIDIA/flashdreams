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

"""Authenticated, bounded downloads of pinned AgiBot archive members."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from filelock import FileLock
from huggingface_hub import cached_assets_path, constants, hf_hub_url
from huggingface_hub.utils import build_hf_headers

from flashdreams.core.io.disk import (
    CACHE_MIN_FREE_ENV,
    DiskSpaceContext,
    cache_min_free_bytes,
    ensure_free_disk,
    raise_if_disk_space_error,
)

_MAX_MEMBER_BYTES = 512 * 1024 * 1024
"""Maximum selected video/state size; whole multi-gigabyte archives are forbidden."""


class _SafeRedirect(HTTPRedirectHandler):
    """HTTPS redirects without forwarding Hub credentials to storage hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("Dataset download redirected outside HTTPS")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            redirected is not None
            and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


def _valid_cache(path: Path, size: int, sha256: str) -> bool:
    try:
        if path.stat().st_size != size:
            return False
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == sha256
    except FileNotFoundError:
        return False


def download_archive_member(
    *,
    repo_id: str,
    revision: str,
    archive: str,
    member_name: str,
    offset: int,
    size: int,
    sha256: str,
) -> Path:
    """Download one verified tar member using the caller's Hugging Face login.

    Args:
        repo_id: Original dataset repository, including its namespace.
        revision: Full pinned dataset commit SHA.
        archive: Repository-relative uncompressed tar archive path.
        member_name: Exact regular-file name in the tar archive.
        offset: Payload byte offset, immediately following its tar header.
        size: Payload byte count, at most 512 MiB.
        sha256: Expected SHA-256 of the original member bytes.

    Returns:
        Verified local member in the Hugging Face assets cache. Cache hits
        require neither authentication nor network access.

    Raises:
        ValueError: The manifest, HTTP range, tar header, or hash is invalid.
        RuntimeError: Offline mode prevents fetching or the request fails.
        DiskSpaceError: Cache space is insufficient or a write exhausts it.
    """
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise ValueError("Invalid dataset repository ID")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be a full commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Archive member requires a SHA-256 digest")
    for name in (archive, member_name):
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Archive paths must be relative and traversal-free")
    if not archive.endswith(".tar"):
        raise ValueError("Selected archive must be an uncompressed tar")
    if type(offset) is not int or offset < 512 or offset % 512:
        raise ValueError("Archive member payload offset must be 512-byte aligned")
    if type(size) is not int or not 0 < size <= _MAX_MEMBER_BYTES:
        raise ValueError("Archive member size is outside the bounded download limit")

    identity = json.dumps(
        [repo_id, revision, archive, member_name, offset, size, sha256]
    )
    key = hashlib.sha256(identity.encode()).hexdigest()
    root = cached_assets_path("flashdreams", namespace=repo_id, subfolder=revision)
    target = root / f"{key}-{PurePosixPath(member_name).name}"
    with FileLock(str(target) + ".lock"):
        if _valid_cache(target, size, sha256):
            return target
        if constants.HF_HUB_OFFLINE:
            raise RuntimeError(
                f"Selected dataset member is not cached: {member_name}. "
                "Disable HF_HUB_OFFLINE and download with your Hugging Face login."
            )
        min_bytes = cache_min_free_bytes()
        disk_context: DiskSpaceContext = dict(
            label="AgiBot scene cache",
            required_bytes=min_bytes,
            env_vars=("HF_HOME", "HF_ASSETS_CACHE", CACHE_MIN_FREE_ENV),
            settings={"repo": repo_id, "member": member_name},
        )
        ensure_free_disk(root, **disk_context)
        headers = build_hf_headers(token=True)
        start, end = offset - 512, offset + size - 1
        headers.update({"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        url = hf_hub_url(repo_id, archive, repo_type="dataset", revision=revision)
        if urlsplit(url).scheme != "https":
            raise ValueError("Dataset downloads require an HTTPS Hugging Face endpoint")
        request = Request(url, headers=headers)
        temporary: Path | None = None
        try:
            with build_opener(_SafeRedirect()).open(request, timeout=60) as response:
                content_range = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)",
                    response.headers.get("Content-Range", ""),
                )
                if (
                    response.status != 206
                    or content_range is None
                    or tuple(map(int, content_range.groups()[:2])) != (start, end)
                    or int(content_range.group(3)) <= end
                    or response.headers.get("Content-Encoding", "identity")
                    != "identity"
                ):
                    raise ValueError(
                        "Dataset server did not honor the exact byte range"
                    )
                length = response.headers.get("Content-Length")
                if length is not None and length != str(size + 512):
                    raise ValueError("Dataset range has an unexpected Content-Length")
                try:
                    member = tarfile.TarInfo.frombuf(
                        response.read(512), "utf-8", "strict"
                    )
                except (tarfile.HeaderError, UnicodeError) as error:
                    raise ValueError("Invalid archive member header") from error
                if (
                    not member.isfile()
                    or member.name != member_name
                    or member.size != size
                ):
                    raise ValueError(
                        "Archive member header does not match the pinned manifest"
                    )
                digest = hashlib.sha256()
                with tempfile.NamedTemporaryFile(
                    dir=root, suffix=".part", delete=False
                ) as output:
                    temporary = Path(output.name)
                    remaining = size
                    while remaining:
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Dataset member response ended early")
                        output.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                if response.read(1):
                    raise ValueError(
                        "Dataset member response exceeded the requested range"
                    )
                if digest.hexdigest() != sha256:
                    raise ValueError(
                        "Dataset member SHA-256 does not match the pinned manifest"
                    )
                temporary.replace(target)
                return target
        except HTTPError as error:
            raise RuntimeError(
                f"Dataset download failed (HTTP {error.code}). Request access to "
                f"{repo_id} and use your own HF_TOKEN or hf auth login credentials."
            ) from None
        except (URLError, TimeoutError):
            raise RuntimeError(
                "Dataset download failed; check network access and retry"
            ) from None
        except OSError as error:
            raise_if_disk_space_error(error, path=root, **disk_context)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
