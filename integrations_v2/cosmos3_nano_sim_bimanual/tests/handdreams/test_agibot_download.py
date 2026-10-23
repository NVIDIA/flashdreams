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

"""Bounded dataset-member download and user-owned credential contracts."""

import errno
import hashlib
import io
import tarfile
from pathlib import Path
from urllib.request import Request

import pytest
from cosmos3_nano_sim_bimanual.apps.handdreams import agibot_download as download

from flashdreams.core.io import disk

pytestmark = pytest.mark.ci_cpu


def _manifest(payload=b"selected original video"):
    return {
        "repo_id": "agibot-world/AgiBotWorld-Beta",
        "revision": "a" * 40,
        "archive": "observations/522/750000-760000.tar",
        "member_name": "754102/videos/head_color.mp4",
        "offset": 4096,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _header(manifest, **changes):
    member = tarfile.TarInfo(manifest["member_name"])
    member.size = manifest["size"]
    for name, value in changes.items():
        setattr(member, name, value)
    return member.tobuf(format=tarfile.USTAR_FORMAT)


class _Response(io.BytesIO):
    def __init__(self, manifest, payload, *, header=None):
        super().__init__((header or _header(manifest)) + payload)
        self.status = 206
        self.headers = {
            "Content-Range": f"bytes {manifest['offset'] - 512}-{manifest['offset'] + manifest['size'] - 1}/999999",
            "Content-Length": str(manifest["size"] + 512),
        }


def _install_http(monkeypatch, tmp_path, response):
    calls = []
    credentials = []

    def assets(library_name, **kwargs):
        assert library_name == "flashdreams"
        assert kwargs == {
            "namespace": "agibot-world/AgiBotWorld-Beta",
            "subfolder": "a" * 40,
        }
        return tmp_path

    def headers(*, token):
        credentials.append(token)
        return {"authorization": "Bearer caller-owned-test-token"}

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            assert timeout == 60
            return response

    monkeypatch.setattr(download, "cached_assets_path", assets)
    monkeypatch.setattr(download, "build_hf_headers", headers)
    monkeypatch.setattr(download, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(download.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setenv(disk.CACHE_MIN_FREE_ENV, "0")
    return calls, credentials


def test_download_uses_pinned_original_and_caller_credentials(tmp_path, monkeypatch):
    payload = b"selected original video"
    manifest = _manifest(payload)
    calls, credentials = _install_http(
        monkeypatch, tmp_path, _Response(manifest, payload)
    )
    path = download.download_archive_member(**manifest)
    assert path.read_bytes() == payload
    assert len(calls) == 1
    request = calls[0]
    assert request.full_url == (
        "https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/resolve/"
        + "a" * 40
        + "/observations/522/750000-760000.tar"
    )
    assert request.get_header("Range") == f"bytes=3584-{4096 + len(payload) - 1}"
    assert request.get_header("Authorization") == "Bearer caller-owned-test-token"
    assert request.get_header("Accept-encoding") == "identity"
    assert credentials == [True]
    assert not list(tmp_path.glob("*.part"))

    monkeypatch.setattr(download.constants, "HF_HUB_OFFLINE", True)
    assert download.download_archive_member(**manifest) == path
    assert len(calls) == len(credentials) == 1


def test_offline_miss_does_not_request_credentials_or_network(tmp_path, monkeypatch):
    manifest = _manifest()
    calls, credentials = _install_http(monkeypatch, tmp_path, None)
    monkeypatch.setattr(download.constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(RuntimeError, match="not cached"):
        download.download_archive_member(**manifest)
    assert calls == credentials == []


def test_disk_preflight_prevents_fetch_and_preserves_cached_members(
    tmp_path, monkeypatch
):
    payload = b"selected original video"
    manifest = _manifest(payload)
    calls, credentials = _install_http(
        monkeypatch, tmp_path, _Response(manifest, payload)
    )
    monkeypatch.setenv(disk.CACHE_MIN_FREE_ENV, "1")
    monkeypatch.setattr(disk, "available_bytes", lambda _: 0)
    with pytest.raises(disk.DiskSpaceError, match="AgiBot scene cache"):
        download.download_archive_member(**manifest)
    assert calls == credentials == []

    monkeypatch.setenv(disk.CACHE_MIN_FREE_ENV, "0")
    path = download.download_archive_member(**manifest)
    monkeypatch.setenv(disk.CACHE_MIN_FREE_ENV, "1")
    assert download.download_archive_member(**manifest) == path
    assert len(calls) == len(credentials) == 1


@pytest.mark.parametrize("failure", ["create", "write", "publish"])
def test_disk_exhaustion_reports_cache_settings_and_cleans_partial_files(
    tmp_path, monkeypatch, failure
):
    payload = b"selected original video"
    manifest = _manifest(payload)
    _install_http(monkeypatch, tmp_path, _Response(manifest, payload))

    def no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    original = download.tempfile.NamedTemporaryFile

    def exhausted_output(*args, **kwargs):
        output = original(*args, **kwargs)
        output.write = no_space
        return output

    if failure == "publish":
        monkeypatch.setattr(Path, "replace", no_space)
    else:
        monkeypatch.setattr(
            download.tempfile,
            "NamedTemporaryFile",
            no_space if failure == "create" else exhausted_output,
        )
    with pytest.raises(disk.DiskSpaceError, match="HF_ASSETS_CACHE") as caught:
        download.download_archive_member(**manifest)
    assert "AgiBot scene cache" in str(caught.value)
    assert manifest["member_name"] in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
    assert all(path.suffix == ".lock" for path in tmp_path.iterdir())


@pytest.mark.parametrize(
    "failure", ["status", "range", "length", "encoding", "short", "excess", "hash"]
)
def test_failed_response_never_creates_a_complete_cache(tmp_path, monkeypatch, failure):
    payload = b"selected original video"
    manifest = _manifest(payload)
    if failure == "short":
        payload = payload[:-1]
    elif failure == "excess":
        payload += b"x"
    elif failure == "hash":
        payload = b"x" * len(payload)
    response = _Response(manifest, payload)
    if failure == "status":
        response.status = 200
    elif failure == "range":
        response.headers["Content-Range"] = "bytes 0-512/999999"
    elif failure == "length":
        response.headers["Content-Length"] = "999999"
    elif failure == "encoding":
        response.headers["Content-Encoding"] = "gzip"
    _install_http(monkeypatch, tmp_path, response)
    with pytest.raises(ValueError):
        download.download_archive_member(**manifest)
    assert all(path.suffix == ".lock" for path in tmp_path.iterdir())


@pytest.mark.parametrize(
    "changes", [{"name": "wrong.mp4"}, {"size": 123}, {"type": tarfile.SYMTYPE}]
)
def test_header_must_match_regular_member(tmp_path, monkeypatch, changes):
    manifest = _manifest()
    response = _Response(
        manifest, b"selected original video", header=_header(manifest, **changes)
    )
    _install_http(monkeypatch, tmp_path, response)
    with pytest.raises(ValueError, match="header does not match"):
        download.download_archive_member(**manifest)
    assert not list(tmp_path.glob("*.mp4"))


def test_header_checksum_is_verified(tmp_path, monkeypatch):
    manifest = _manifest()
    header = bytearray(_header(manifest))
    header[0] ^= 1
    response = _Response(manifest, b"selected original video", header=bytes(header))
    _install_http(monkeypatch, tmp_path, response)
    with pytest.raises(ValueError, match="Invalid archive member header"):
        download.download_archive_member(**manifest)


def test_corrupt_cache_is_not_accepted_offline_and_can_be_repaired(
    tmp_path, monkeypatch
):
    payload = b"selected original video"
    manifest = _manifest(payload)
    _install_http(monkeypatch, tmp_path, _Response(manifest, payload))
    path = download.download_archive_member(**manifest)
    path.write_bytes(b"x" * len(payload))
    monkeypatch.setattr(download.constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(RuntimeError, match="not cached"):
        download.download_archive_member(**manifest)
    _install_http(monkeypatch, tmp_path, _Response(manifest, payload))
    assert download.download_archive_member(**manifest).read_bytes() == payload


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": "main"},
        {"sha256": ""},
        {"size": True},
        {"size": 0},
        {"size": 513 * 1024 * 1024},
        {"offset": 0},
        {"offset": 513},
        {"archive": "../other.tar"},
        {"archive": "observations.tar.gz"},
        {"member_name": "/absolute"},
        {"repo_id": "https://example.com"},
    ],
)
def test_invalid_manifest_is_rejected_before_network(monkeypatch, changes):
    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid manifest must not make a request")

    monkeypatch.setattr(download, "build_opener", unexpected)
    with pytest.raises(ValueError):
        download.download_archive_member(**(_manifest() | changes))


def test_redirect_strips_credentials_only_across_hosts():
    request = Request(
        "https://huggingface.co/datasets/repo",
        headers={
            "Authorization": "Bearer caller-owned-test-token",
            "Range": "bytes=0-511",
        },
    )
    handler = download._SafeRedirect()
    same = handler.redirect_request(
        request, None, 302, "Found", {}, "https://huggingface.co/cache"
    )
    assert same.get_header("Authorization") == "Bearer caller-owned-test-token"
    cross = handler.redirect_request(
        request, None, 302, "Found", {}, "https://cdn.example.org/file"
    )
    assert cross.get_header("Authorization") is None
    assert cross.get_header("Range") == "bytes=0-511"
    with pytest.raises(ValueError, match="outside HTTPS"):
        handler.redirect_request(
            request, None, 302, "Found", {}, "http://cdn.example.org/file"
        )
