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

"""Unified checkpoint loader that dispatches by source URL."""

from __future__ import annotations

import io
import json
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import Literal, overload
from urllib.parse import unquote, urlparse

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from loguru import logger
from safetensors.torch import load as load_safetensors
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

from flashdreams.core.io.disk import (
    CACHE_MIN_FREE_ENV,
    cache_min_free_bytes,
    default_huggingface_cache_dir,
    disk_space_error_from_exception,
    ensure_free_disk,
)
from flashdreams.core.io.s3_filesystem import S3FileSystem, S3StorageReader

_OMNIDREAMS_CHECKPOINT_CREDENTIAL_PATH = "credentials/s3_checkpoint.secret"
_OMNIDREAMS_CHECKPOINT_LOCAL_CACHE_DIR = os.path.expanduser(
    os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")
)


def _preflight_hf_cache(
    *,
    label: str,
    settings: dict[str, object] | None = None,
) -> int:
    min_bytes = cache_min_free_bytes()
    ensure_free_disk(
        default_huggingface_cache_dir(),
        required_bytes=min_bytes,
        label=label,
        env_vars=("HF_HOME", "HF_HUB_CACHE", CACHE_MIN_FREE_ENV),
        settings=settings,
    )
    return min_bytes


def _hf_cache_filename(filename: str, subfolder: str | None) -> str:
    return f"{subfolder.rstrip('/')}/{filename}" if subfolder else filename


def _is_hf_file_cached(
    *,
    repo_id: str,
    filename: str,
    subfolder: str | None,
    revision: str,
) -> bool:
    try:
        cached = try_to_load_from_cache(
            repo_id=repo_id,
            filename=_hf_cache_filename(filename, subfolder),
            revision=revision,
        )
    except Exception:
        return False
    return isinstance(cached, str) and os.path.exists(cached)


def _preflight_checkpoint_cache_requirement(
    *,
    label: str,
    min_free_gb: float | None,
    settings: dict[str, object] | None = None,
    local_cache_path: str | None = None,
) -> int | None:
    """Run a one-time first-run checkpoint storage preflight.

    This is intentionally separate from the generic per-write 20 GiB reserve:
    large sharded checkpoints consume space in stages, so reapplying a 200 GiB
    requirement before every shard or merged-cache write would fail after an
    otherwise valid partial download has already used disk.
    """
    if min_free_gb is None:
        return None
    min_bytes = cache_min_free_bytes(min_free_gb)
    if min_bytes <= 0:
        return min_bytes

    ensure_free_disk(
        default_huggingface_cache_dir(),
        required_bytes=min_bytes,
        label=label,
        env_vars=("HF_HOME", "HF_HUB_CACHE", CACHE_MIN_FREE_ENV),
        settings=settings,
    )
    if local_cache_path is not None:
        ensure_free_disk(
            os.path.dirname(local_cache_path) or ".",
            required_bytes=min_bytes,
            label=f"{label} merged cache",
            env_vars=("FLASHDREAMS_CACHE_DIR", CACHE_MIN_FREE_ENV),
            settings=settings,
        )
    return min_bytes


def _raise_hf_cache_disk_error(
    exc: BaseException,
    *,
    label: str,
    required_bytes: int,
    settings: dict[str, object] | None = None,
) -> None:
    disk_error = disk_space_error_from_exception(
        exc,
        path=default_huggingface_cache_dir(),
        label=label,
        required_bytes=required_bytes,
        env_vars=("HF_HOME", "HF_HUB_CACHE", CACHE_MIN_FREE_ENV),
        settings=settings,
    )
    if disk_error is not None:
        raise disk_error from exc


def _preflight_local_cache_path(
    path: str,
    *,
    label: str,
    settings: dict[str, object] | None = None,
) -> int:
    min_bytes = cache_min_free_bytes()
    ensure_free_disk(
        os.path.dirname(path) or ".",
        required_bytes=min_bytes,
        label=label,
        env_vars=("FLASHDREAMS_CACHE_DIR", CACHE_MIN_FREE_ENV),
        settings=settings,
    )
    return min_bytes


def _raise_local_cache_disk_error(
    exc: BaseException,
    *,
    path: str,
    label: str,
    required_bytes: int,
    settings: dict[str, object] | None = None,
) -> None:
    disk_error = disk_space_error_from_exception(
        exc,
        path=path,
        label=label,
        required_bytes=required_bytes,
        env_vars=("FLASHDREAMS_CACHE_DIR", CACHE_MIN_FREE_ENV),
        settings=settings,
    )
    if disk_error is not None:
        raise disk_error from exc


def _is_huggingface_checkpoint_url(path: str) -> bool:
    """Check whether path is a supported Hugging Face checkpoint URL."""
    if not path.startswith(("http://", "https://")):
        return False
    parsed = urlparse(path)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host.removeprefix("www.")
    if host != "huggingface.co":
        return False
    return "/blob/" in parsed.path or "/resolve/" in parsed.path


def _get_checkpoint_extension(checkpoint_path: str) -> str:
    """Get extension from local path, S3 path, or URL."""
    if checkpoint_path.startswith(("http://", "https://")):
        parsed = urlparse(checkpoint_path)
        return os.path.splitext(parsed.path)[1].lower()
    return os.path.splitext(checkpoint_path)[1].lower()


def _is_sharded_safetensors_index_checkpoint(path: str) -> bool:
    """Return whether ``path`` points to a Hugging Face-style sharded safetensors index file."""
    if path.startswith(("http://", "https://")):
        basename = os.path.basename(unquote(urlparse(path).path))
    else:
        basename = os.path.basename(path)
    return basename.endswith(".safetensors.index.json")


def _sharded_safetensors_merge_cache_path(
    checkpoint_path: str, local_cache_dir: str
) -> str:
    """Stable path for a single-file cache of merged sharded weights."""
    if checkpoint_path.startswith(("http://", "https://")):
        repo_id, filename, subfolder, revision = _parse_huggingface_checkpoint_url(
            checkpoint_path
        )
        sub = subfolder.replace("/", "__") if subfolder else "root"
        stem = f"{repo_id.replace('/', '__')}__{revision}__{sub}__{filename}"
    else:
        stem = os.path.abspath(checkpoint_path).replace(os.sep, "__")
        if os.name == "nt":
            stem = stem.replace(":", "_")
    return os.path.join(local_cache_dir, "merged_safetensors", stem + ".safetensors")


def _safetensors_device(map_location: str | torch.device) -> str:
    """Normalize a ``map_location`` argument to the string form expected by safetensors."""
    if isinstance(map_location, torch.device):
        return str(map_location)
    return str(map_location)


def _hf_hub_download_shard_task(
    args: tuple[str, str, str | None, str],
) -> tuple[str, str]:
    """Picklable worker: download one shard; used by ProcessPoolExecutor."""
    repo_id, shard_file, subfolder, revision = args
    settings: dict[str, object] = {
        "repo": repo_id,
        "filename": shard_file,
        "revision": revision,
    }
    min_bytes = _preflight_hf_cache(
        label="Hugging Face checkpoint shard cache",
        settings=settings,
    )
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=shard_file,
            subfolder=subfolder,
            revision=revision,
        )
    except Exception as exc:
        _raise_hf_cache_disk_error(
            exc,
            label="Hugging Face checkpoint shard cache",
            required_bytes=min_bytes,
            settings=settings,
        )
        raise
    return shard_file, path


def _parallel_hf_hub_download_shards(
    *,
    repo_id: str,
    shard_files: list[str],
    subfolder: str | None,
    revision: str,
) -> dict[str, str]:
    """Download unique shard files in parallel processes; returns shard -> local path."""
    if not shard_files:
        return {}
    if len(shard_files) == 1:
        s = shard_files[0]
        _, path = _hf_hub_download_shard_task((repo_id, s, subfolder, revision))
        return {s: path}

    _preflight_hf_cache(
        label="Hugging Face checkpoint shard cache",
        settings={"repo": repo_id, "revision": revision},
    )

    env_cap = os.getenv("FLASHDREAMS_HF_SHARD_DOWNLOAD_WORKERS")
    if env_cap is not None:
        max_workers = max(1, min(len(shard_files), int(env_cap)))
    else:
        max_workers = min(len(shard_files), min(32, max(4, (os.cpu_count() or 4) * 2)))

    work = [(repo_id, s, subfolder, revision) for s in shard_files]
    logger.info(
        f"Downloading {len(shard_files)} Hugging Face safetensors shards "
        f"with up to {max_workers} parallel processes"
    )
    shard_to_path: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for shard_file, path in pool.map(_hf_hub_download_shard_task, work):
            shard_to_path[shard_file] = path
    return shard_to_path


def _merge_sharded_safetensors_from_index(
    *,
    weight_map: dict[str, str],
    resolve_shard_path: Callable[[str], str],
    map_location: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Load each shard once and assemble tensors listed in weight_map."""
    device = _safetensors_device(map_location)
    keys_by_shard: dict[str, list[str]] = {}
    for tensor_name, shard_file in weight_map.items():
        keys_by_shard.setdefault(shard_file, []).append(tensor_name)

    merged: dict[str, torch.Tensor] = {}
    for shard_file in sorted(keys_by_shard):
        shard_path = resolve_shard_path(shard_file)
        shard_sd = load_safetensors_file(shard_path, device=device)
        for key in keys_by_shard[shard_file]:
            if key not in shard_sd:
                raise KeyError(
                    f"Key {key!r} missing from shard {shard_file!r} (path {shard_path!r})"
                )
            merged[key] = shard_sd[key]
    return merged


def _load_sharded_safetensors_index_checkpoint(
    checkpoint_path: str,
    local_cache_dir: str,
    map_location: str | torch.device,
    checkpoint_min_free_gb: float | None = None,
) -> dict[str, torch.Tensor]:
    """Load HF-style sharded safetensors (index.json + shards) into one state dict."""
    if local_cache_dir is None:
        raise ValueError(
            "local_cache_dir is required to cache merged sharded safetensors"
        )
    cache_path = _sharded_safetensors_merge_cache_path(checkpoint_path, local_cache_dir)
    if os.path.exists(cache_path):
        logger.info(f"Loading merged sharded checkpoint from cache: {cache_path}")
        return _load_checkpoint_from_local(cache_path, ".safetensors", map_location)

    is_hf_url = _is_huggingface_checkpoint_url(checkpoint_path)

    if is_hf_url:
        repo_id, index_filename, subfolder, revision = (
            _parse_huggingface_checkpoint_url(checkpoint_path)
        )
        logger.info(f"Merging sharded safetensors from Hugging Face: {checkpoint_path}")
        settings: dict[str, object] = {
            "repo": repo_id,
            "filename": index_filename,
            "revision": revision,
        }
        _preflight_checkpoint_cache_requirement(
            label="Hugging Face sharded checkpoint cache",
            min_free_gb=checkpoint_min_free_gb,
            settings=settings,
            local_cache_path=cache_path,
        )
        min_bytes = _preflight_hf_cache(
            label="Hugging Face checkpoint index cache",
            settings=settings,
        )
        try:
            index_local = hf_hub_download(
                repo_id=repo_id,
                filename=index_filename,
                subfolder=subfolder,
                revision=revision,
            )
        except Exception as exc:
            _raise_hf_cache_disk_error(
                exc,
                label="Hugging Face checkpoint index cache",
                required_bytes=min_bytes,
                settings=settings,
            )
            raise
        with open(index_local) as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"Invalid or empty weight_map in safetensors index: {index_local}"
            )

        unique_shards = sorted(set(weight_map.values()))
        shard_to_path = _parallel_hf_hub_download_shards(
            repo_id=repo_id,
            shard_files=unique_shards,
            subfolder=subfolder,
            revision=revision,
        )

        def resolve_shard_path(shard_file: str) -> str:
            return shard_to_path[shard_file]

        merged = _merge_sharded_safetensors_from_index(
            weight_map=weight_map,
            resolve_shard_path=resolve_shard_path,
            map_location=map_location,
        )
    else:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Sharded safetensors index not found: {checkpoint_path}"
            )
        logger.info(f"Merging sharded safetensors from local index: {checkpoint_path}")
        with open(checkpoint_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"Invalid or empty weight_map in safetensors index: {checkpoint_path}"
            )
        base_dir = os.path.dirname(os.path.abspath(checkpoint_path))

        def resolve_shard_path(shard_file: str) -> str:
            return os.path.join(base_dir, shard_file)

        merged = _merge_sharded_safetensors_from_index(
            weight_map=weight_map,
            resolve_shard_path=resolve_shard_path,
            map_location=map_location,
        )

    _save_to_local_cache(
        merged,
        cache_path,
        ".safetensors",
        label="merged sharded checkpoint cache",
    )
    logger.info(f"Saved merged sharded checkpoint to: {cache_path}")
    return merged


def _parse_huggingface_checkpoint_url(
    url: str,
) -> tuple[str, str, str | None, str]:
    """Parse a HF file URL into hf_hub_download args.

    Supports:
      - https://huggingface.co/<namespace>/<repo>/blob/<revision>/<subfolder...>/<file>
      - https://huggingface.co/<namespace>/<repo>/resolve/<revision>/<subfolder...>/<file>
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host.removeprefix("www.")
    if host != "huggingface.co":
        raise ValueError(f"Not a Hugging Face URL: {url}")

    parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 5:
        raise ValueError(
            f"Invalid Hugging Face checkpoint URL: {url}. Expected /<namespace>/<repo>/blob|resolve/<revision>/<path/to/file>"
        )

    namespace, repo, route = parts[0], parts[1], parts[2]
    if route not in ("blob", "resolve"):
        raise ValueError(
            f"Unsupported Hugging Face URL route '{route}' in {url}. Expected 'blob' or 'resolve'."
        )

    revision = parts[3]
    file_parts = parts[4:]
    if not file_parts:
        raise ValueError(f"Missing file path in Hugging Face URL: {url}")

    filename = file_parts[-1]
    subfolder = "/".join(file_parts[:-1]) or None
    repo_id = f"{namespace}/{repo}"
    return repo_id, filename, subfolder, revision


def _download_checkpoint_from_huggingface_url(
    url: str,
    *,
    checkpoint_min_free_gb: float | None = None,
) -> str:
    """Download a checkpoint from Hugging Face and return local cached path."""
    repo_id, filename, subfolder, revision = _parse_huggingface_checkpoint_url(url)
    logger.info(f"Downloading checkpoint from Hugging Face: {url}")
    settings: dict[str, object] = {
        "repo": repo_id,
        "filename": filename,
        "revision": revision,
    }
    if not _is_hf_file_cached(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        revision=revision,
    ):
        _preflight_checkpoint_cache_requirement(
            label="Hugging Face checkpoint cache",
            min_free_gb=checkpoint_min_free_gb,
            settings=settings,
        )
    min_bytes = _preflight_hf_cache(
        label="Hugging Face checkpoint cache",
        settings=settings,
    )
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
        )
    except Exception as exc:
        _raise_hf_cache_disk_error(
            exc,
            label="Hugging Face checkpoint cache",
            required_bytes=min_bytes,
            settings=settings,
        )
        raise
    logger.info(f"Checkpoint downloaded to local HF cache: {local_path}")
    return local_path


def get_storage_reader(
    checkpoint_path: str, credential_path: str = _OMNIDREAMS_CHECKPOINT_CREDENTIAL_PATH
) -> FileSystemReader:
    """Return the right storage reader for an S3 or local checkpoint path.

    Args:
        checkpoint_path: ``s3://`` URI or local path.
        credential_path: S3 credentials path (used only for S3 paths).

    Returns:
        ``S3StorageReader`` for ``s3://`` paths, ``FileSystemReader`` otherwise.
    """
    if checkpoint_path.startswith("s3://"):
        return S3StorageReader(credential_path=credential_path, path=checkpoint_path)
    else:
        return FileSystemReader(checkpoint_path)


def load_distributed_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    check_success: bool = False,
    local_cache_dir: str | None = _OMNIDREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _OMNIDREAMS_CHECKPOINT_CREDENTIAL_PATH,
) -> torch.nn.Module:
    """Load a DCP checkpoint into ``model`` in-place.

    Args:
        model: Model to load weights into.
        checkpoint_path: Directory path to a DCP checkpoint (S3 or local).
        check_success: Compare the state dict before/after to verify the load
            actually changed weights. Recommended since DCP load does not
            fail on missing keys. Raises ``RuntimeError`` if any state-dict
            entry remains unchanged.
        local_cache_dir: Directory used to cache S3 DCP loads as a local
            single-file checkpoint. Set to ``None`` to disable local cache
            reads/writes.
    """
    is_s3_checkpoint = checkpoint_path.startswith("s3://")

    # Cache the merged DCP shards as a single ``.pt`` next to the local
    # cache root so subsequent loads skip the S3 round trip.
    local_cache_checkpoint_path = None
    if is_s3_checkpoint and local_cache_dir is not None:
        local_cache_checkpoint_path = os.path.join(
            local_cache_dir,
            checkpoint_path.split("s3://")[1].rstrip("/") + ".pt",
        )

    # Local cache hit: trust it (the ``check_success`` path below only
    # makes sense for a fresh DCP load that may silently miss keys).
    if local_cache_checkpoint_path is not None and os.path.exists(
        local_cache_checkpoint_path
    ):
        state_dict = torch.load(local_cache_checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(
            f"Loaded successfully from the local cache: {local_cache_checkpoint_path}"
        )
        return model

    # If check_success is True, we check if the checkpoint is loaded successfully, by
    # comparing the state dict of the model before and after loading the checkpoint.
    if check_success:
        prev_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Load the DCP checkpoint. Note DCP load doesn't fail if there is no matching key.
    # So the best practice is to set check_success to True.
    storage_reader = get_storage_reader(
        checkpoint_path, credential_path=credential_path
    )
    state_dict = model.state_dict()
    dcp_load(
        state_dict,
        storage_reader=storage_reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )

    # Now check if the checkpoint is loaded successfully.
    if check_success:
        unchanged_keys: list[str] = []
        for k, v in model.state_dict().items():
            prev_v = prev_state_dict[k]
            if torch.equal(prev_v, v):
                unchanged_keys.append(k)
        if unchanged_keys:
            raise RuntimeError(
                "DCP load did not update all state_dict entries. "
                "This usually means the checkpoint path or model config does not "
                "match the target network. Unchanged keys: "
                f"{', '.join(unchanged_keys[:20])}"
                + (" ..." if len(unchanged_keys) > 20 else "")
            )

    # Cache the state dict locally if needed..
    if local_cache_checkpoint_path is not None:
        _save_to_local_cache(
            model.state_dict(),
            local_cache_checkpoint_path,
            ".pt",
            label="distributed checkpoint cache",
        )
        logger.info(f"Loaded successfully from the checkpoint: {checkpoint_path}")
        logger.info(f"Cached locally to {local_cache_checkpoint_path}")
    else:
        logger.info(f"Loaded successfully from the checkpoint: {checkpoint_path}")

    return model


def load_single_checkpoint(
    checkpoint_path: str,
    local_cache_dir: str = _OMNIDREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _OMNIDREAMS_CHECKPOINT_CREDENTIAL_PATH,
    map_location: str | torch.device = "cpu",
    checkpoint_min_free_gb: float | None = None,
) -> dict[str, torch.Tensor]:
    """Load a single-file checkpoint from local disk, S3, or a Hugging Face URL.

    S3 paths are cached locally for faster subsequent loads. HF file URLs are
    downloaded via ``hf_hub_download``.

    Args:
        checkpoint_path: Path/URL to a ``.pt`` / ``.pth`` / ``.ckpt`` /
            ``.safetensors`` file, or to an HF-style ``*.safetensors.index.json``
            (shards are merged on first load and cached).
        local_cache_dir: Directory for S3 / merged-safetensors caches.
        credential_path: S3 credentials path.
        map_location: Device to map tensors to (``.pt`` / ``.pth`` / ``.ckpt`` only).
        checkpoint_min_free_gb: Optional first-run free-space requirement in
            GiB for Hugging Face checkpoint downloads. The
            ``FLASHDREAMS_MIN_CACHE_FREE_GB`` environment override still wins.

    Returns:
        State dict.

    Raises:
        ValueError: Unsupported file extension or unsupported S3 sharded
            index input.
    """
    if _is_sharded_safetensors_index_checkpoint(checkpoint_path):
        if checkpoint_path.startswith("s3://"):
            raise ValueError(
                "Sharded safetensors index checkpoints are not supported on S3; "
                "use a Hugging Face file URL or a local index path."
            )
        return _load_sharded_safetensors_index_checkpoint(
            checkpoint_path,
            local_cache_dir,
            map_location,
            checkpoint_min_free_gb=checkpoint_min_free_gb,
        )

    is_s3_path = checkpoint_path.startswith("s3://")
    is_hf_url = _is_huggingface_checkpoint_url(checkpoint_path)

    # Determine file extension
    ext = _get_checkpoint_extension(checkpoint_path)
    if ext not in (".pt", ".pth", ".ckpt", ".safetensors"):
        raise ValueError(
            f"Unsupported checkpoint extension: {ext}. "
            f"Supported: .pt, .pth, .ckpt, .safetensors"
        )

    # For Hugging Face URLs, use HF cache and then load locally.
    if is_hf_url:
        local_path = _download_checkpoint_from_huggingface_url(
            checkpoint_path,
            checkpoint_min_free_gb=checkpoint_min_free_gb,
        )
        return _load_checkpoint_from_local(local_path, ext, map_location)

    # For S3 paths, check local cache first
    local_cache_path = None
    if is_s3_path and local_cache_dir is not None:
        local_cache_path = os.path.join(
            local_cache_dir, checkpoint_path.removeprefix("s3://")
        )
        if os.path.exists(local_cache_path):
            logger.info(f"Loading from local cache: {local_cache_path}")
            return _load_checkpoint_from_local(local_cache_path, ext, map_location)

    # Load from S3 or local
    if is_s3_path:
        state_dict = _load_checkpoint_from_s3(
            checkpoint_path, ext, credential_path, map_location
        )
        # Cache to local
        if local_cache_path is not None:
            _save_to_local_cache(
                state_dict,
                local_cache_path,
                ext,
                label="checkpoint cache",
            )
            logger.info(f"Cached checkpoint to: {local_cache_path}")
    else:
        state_dict = _load_checkpoint_from_local(checkpoint_path, ext, map_location)

    return state_dict


def _load_checkpoint_from_local(
    path: str,
    ext: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load checkpoint from local filesystem."""
    if ext == ".safetensors":
        with open(path, "rb") as f:
            return load_safetensors(f.read())
    else:
        return torch.load(path, map_location=map_location, weights_only=False)


def _load_checkpoint_from_s3(
    s3_path: str,
    ext: str,
    credential_path: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load checkpoint from S3."""
    logger.info(f"Downloading checkpoint from S3: {s3_path}")
    s3_fs = S3FileSystem(credential_path=credential_path)
    with s3_fs.create_stream(s3_path, "rb") as stream:
        data_bytes = stream.read()

    if ext == ".safetensors":
        return load_safetensors(data_bytes)
    else:
        return torch.load(
            io.BytesIO(data_bytes), map_location=map_location, weights_only=False
        )


def _save_to_local_cache(
    state_dict: Mapping[str, torch.Tensor],
    path: str,
    ext: str,
    *,
    label: str = "checkpoint cache",
) -> None:
    """Save state dict to local cache."""
    min_bytes = _preflight_local_cache_path(
        path,
        label=label,
        settings={"path": path},
    )
    try:
        if ext == ".safetensors":
            save_safetensors(dict(state_dict), path)
        else:
            torch.save(state_dict, path)
    except Exception as exc:
        _raise_local_cache_disk_error(
            exc,
            path=path,
            label=label,
            required_bytes=min_bytes,
            settings={"path": path},
        )
        raise


@overload
def load_checkpoint(
    checkpoint_path: str,
    model: None = None,
    checkpoint_type: Literal["auto", "single", "distributed"] = "auto",
    local_cache_dir: str = _OMNIDREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _OMNIDREAMS_CHECKPOINT_CREDENTIAL_PATH,
    map_location: str | torch.device = "cpu",
    check_success: bool = False,
    checkpoint_min_free_gb: float | None = None,
) -> dict[str, torch.Tensor]: ...


@overload
def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    checkpoint_type: Literal["auto", "single", "distributed"] = "auto",
    local_cache_dir: str = _OMNIDREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _OMNIDREAMS_CHECKPOINT_CREDENTIAL_PATH,
    map_location: str | torch.device = "cpu",
    check_success: bool = False,
    checkpoint_min_free_gb: float | None = None,
) -> torch.nn.Module: ...


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module | None = None,
    checkpoint_type: Literal["auto", "single", "distributed"] = "auto",
    local_cache_dir: str = _OMNIDREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _OMNIDREAMS_CHECKPOINT_CREDENTIAL_PATH,
    map_location: str | torch.device = "cpu",
    check_success: bool = False,
    checkpoint_min_free_gb: float | None = None,
) -> dict[str, torch.Tensor] | torch.nn.Module:
    """Load checkpoints from S3, local disk, or Hugging Face.

    Handles single-file checkpoints (``.pt`` / ``.pth`` / ``.ckpt`` /
    ``.safetensors``) and distributed checkpoints (DCP). Detection is
    automatic by default.

    Args:
        checkpoint_path: ``s3://`` URI, local path, or HF URL. Single-file or
            DCP directory.
        model: Model to load weights into. Required for DCP. Optional for
            single-file: when provided, ``load_state_dict`` is called.
        checkpoint_type: ``"auto"``, ``"single"``, or ``"distributed"``.
        local_cache_dir: Directory for caches.
        credential_path: S3 credentials path.
        map_location: Device to map tensors to (single-file only).
        check_success: Verify DCP load actually changed weights.
        checkpoint_min_free_gb: Optional first-run free-space requirement in
            GiB for Hugging Face checkpoint downloads. The
            ``FLASHDREAMS_MIN_CACHE_FREE_GB`` environment override still wins.

    Returns:
        State dict if ``model`` is ``None``, otherwise ``model`` with weights
        loaded.

    Raises:
        ValueError: ``checkpoint_type='distributed'`` without a ``model``, or
            an invalid ``checkpoint_type``.

    Examples:

      >>> state = load_checkpoint("s3://bucket/foo.safetensors")
      >>> model = load_checkpoint("s3://bucket/dcp_dir/", model=my_model)
    """
    # Auto-detect checkpoint type
    if checkpoint_type == "auto":
        if _is_sharded_safetensors_index_checkpoint(checkpoint_path):
            checkpoint_type = "single"
        else:
            ext = _get_checkpoint_extension(checkpoint_path)
            if ext in (".pt", ".pth", ".ckpt", ".safetensors"):
                checkpoint_type = "single"
            else:
                checkpoint_type = "distributed"

    if checkpoint_type == "single":
        state_dict = load_single_checkpoint(
            checkpoint_path=checkpoint_path,
            local_cache_dir=local_cache_dir,
            credential_path=credential_path,
            map_location=map_location,
            checkpoint_min_free_gb=checkpoint_min_free_gb,
        )
        if model is not None:
            model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint into model: {checkpoint_path}")
            return model
        return state_dict

    elif checkpoint_type == "distributed":
        if model is None:
            raise ValueError(
                "Model must be provided for distributed checkpoint loading"
            )
        return load_distributed_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            check_success=check_success,
            local_cache_dir=local_cache_dir,
            credential_path=credential_path,
        )

    else:
        raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}")
