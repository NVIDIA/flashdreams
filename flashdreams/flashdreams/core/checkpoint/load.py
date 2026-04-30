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

import io
import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Literal
from urllib.parse import unquote, urlparse

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from safetensors.torch import load as load_safetensors
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

from flashdreams.core.io.s3_filesystem import S3FileSystem, S3StorageReader

_ALPADREAMS_CHECKPOINT_CREDENTIAL_PATH = "credentials/s3_checkpoint.secret"
_ALPADREAMS_CHECKPOINT_LOCAL_CACHE_DIR = os.path.expanduser(
    os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")
)


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
    """True if path points to a Hugging Face-style sharded safetensors index file."""
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
    if isinstance(map_location, torch.device):
        return str(map_location)
    return str(map_location)


def _hf_hub_download_shard_task(
    args: tuple[str, str, str | None, str],
) -> tuple[str, str]:
    """Picklable worker: download one shard; used by ProcessPoolExecutor."""
    repo_id, shard_file, subfolder, revision = args
    path = hf_hub_download(
        repo_id=repo_id,
        filename=shard_file,
        subfolder=subfolder,
        revision=revision,
    )
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
        index_local = hf_hub_download(
            repo_id=repo_id,
            filename=index_filename,
            subfolder=subfolder,
            revision=revision,
        )
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

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_safetensors(merged, cache_path)
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


def _download_checkpoint_from_huggingface_url(url: str) -> str:
    """Download a checkpoint from Hugging Face and return local cached path."""
    repo_id, filename, subfolder, revision = _parse_huggingface_checkpoint_url(url)
    logger.info(f"Downloading checkpoint from Hugging Face: {url}")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        revision=revision,
    )
    logger.info(f"Checkpoint downloaded to local HF cache: {local_path}")
    return local_path


def get_storage_reader(
    checkpoint_path: str, credential_path: str = _ALPADREAMS_CHECKPOINT_CREDENTIAL_PATH
):
    """Get storage reader for S3 or local checkpoint.

    Args:
        checkpoint_path: The path to the checkpoint. Can be S3 or local path.

    Returns:
        The storage reader.
    """
    if checkpoint_path.startswith("s3://"):
        return S3StorageReader(credential_path=credential_path, path=checkpoint_path)
    else:
        return FileSystemReader(checkpoint_path)


def load_distributed_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    check_success: bool = False,
    local_cache_dir: str = _ALPADREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _ALPADREAMS_CHECKPOINT_CREDENTIAL_PATH,
) -> torch.nn.Module:
    """Load distributed checkpoint into a model (Inplace).

    Args:
        model: The model to load the DCP checkpoint into.
        checkpoint_path: The path to the DCP checkpoint. Can be S3 or local path. Should be a directory path.
        check_success: Whether to check if the checkpoint is loaded successfully,
            by comparing the state dict of the model before and after loading the checkpoint.
    """
    is_s3_checkpoint = checkpoint_path.startswith("s3://")

    # Set the cache checkpoint path so that next time we can just load the .pt file locally.
    local_cache_checkpoint_path = None
    if is_s3_checkpoint and local_cache_dir is not None:
        local_cache_checkpoint_path = os.path.join(
            local_cache_dir,
            checkpoint_path.split("s3://")[1].rstrip("/") + ".pt",
        )

    # Check if the local cache checkpoint path exists. If so, we load from the local cache.
    # In this case, we don't need to check for success.
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
    torch.distributed.checkpoint.load(  # ty:ignore[possibly-missing-submodule]
        state_dict,
        storage_reader=storage_reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )

    # Now check if the checkpoint is loaded successfully.
    if check_success:
        for k, v in model.state_dict().items():
            prev_v = prev_state_dict[k]
            if (prev_v == v).all():
                logger.error(
                    f"DCP load seems failed for key {k}. The values are not changed!"
                )

    # Cache the state dict locally if needed..
    if local_cache_checkpoint_path is not None:
        os.makedirs(os.path.dirname(local_cache_checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), local_cache_checkpoint_path)
        logger.info(f"Loaded successfully from the checkpoint: {checkpoint_path}")
        logger.info(f"Cached locally to {local_cache_checkpoint_path}")
    else:
        logger.info(f"Loaded successfully from the checkpoint: {checkpoint_path}")

    return model


def load_single_checkpoint(
    checkpoint_path: str,
    local_cache_dir: str = _ALPADREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _ALPADREAMS_CHECKPOINT_CREDENTIAL_PATH,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a single checkpoint file (.pt, .pth, .safetensors) from local, S3, or HF URL.

    Supports loading from S3 with local caching for faster subsequent loads, and
    Hugging Face file URLs via hf_hub_download.

    Args:
        checkpoint_path: Path/URL to checkpoint file. Supported forms:
            - local file path
            - S3 URL (s3://...)
            - Hugging Face URL (.../blob/... or .../resolve/...)
            - Hugging Face or local ``*.safetensors.index.json`` (shards merged; result cached
              as one ``.safetensors`` under ``local_cache_dir``/merged_safetensors)
            Supported extensions: .pt, .pth, .safetensors
        local_cache_dir: Directory to cache S3 checkpoints locally.
        credential_path: Path to S3 credentials file.
        map_location: Device to map tensors to (for .pt/.pth files).

    Returns:
        State dict loaded from the checkpoint.

    Raises:
        ValueError: If the file extension is not supported.
    """
    if _is_sharded_safetensors_index_checkpoint(checkpoint_path):
        if checkpoint_path.startswith("s3://"):
            raise ValueError(
                "Sharded safetensors index checkpoints are not supported on S3; "
                "use a Hugging Face file URL or a local index path."
            )
        return _load_sharded_safetensors_index_checkpoint(
            checkpoint_path, local_cache_dir, map_location
        )

    is_s3_path = checkpoint_path.startswith("s3://")
    is_hf_url = _is_huggingface_checkpoint_url(checkpoint_path)

    # Determine file extension
    ext = _get_checkpoint_extension(checkpoint_path)
    if ext not in (".pt", ".pth", ".safetensors"):
        raise ValueError(
            f"Unsupported checkpoint extension: {ext}. Supported: .pt, .pth, .safetensors"
        )

    # For Hugging Face URLs, use HF cache and then load locally.
    if is_hf_url:
        local_path = _download_checkpoint_from_huggingface_url(checkpoint_path)
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
            os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)
            _save_to_local_cache(state_dict, local_cache_path, ext)
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
    state_dict: dict[str, torch.Tensor], path: str, ext: str
) -> None:
    """Save state dict to local cache."""
    if ext == ".safetensors":
        save_safetensors(state_dict, path)
    else:
        torch.save(state_dict, path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module | None = None,
    checkpoint_type: Literal["auto", "single", "distributed"] = "auto",
    local_cache_dir: str = _ALPADREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _ALPADREAMS_CHECKPOINT_CREDENTIAL_PATH,
    map_location: str | torch.device = "cpu",
    check_success: bool = False,
) -> dict[str, torch.Tensor] | torch.nn.Module:
    """Unified API to load checkpoints from S3 or local filesystem.

    Supports both single file checkpoints (.pt, .pth, .safetensors) and
    distributed checkpoints (DCP format).

    Args:
        checkpoint_path: Path to checkpoint. Can be S3 (s3://...) or local.
            - For single files: path to .pt, .pth, or .safetensors file
            - For distributed: path to DCP checkpoint directory
        model: Model to load the checkpoint into. Required for distributed checkpoints.
            If provided for single checkpoints, will call model.load_state_dict().
        checkpoint_type: Type of checkpoint to load.
            - "auto": Automatically detect based on path (file vs directory)
            - "single": Force single file loading
            - "distributed": Force distributed checkpoint loading
        local_cache_dir: Directory to cache S3 checkpoints locally.
        credential_path: Path to S3 credentials file.
        map_location: Device to map tensors to (for single file checkpoints).
        check_success: For distributed checkpoints, verify loading succeeded.

    Returns:
        - If model is None: returns the state dict
        - If model is provided: returns the model with loaded weights

    Raises:
        ValueError: If checkpoint_type is "distributed" but model is not provided.
    """
    # Auto-detect checkpoint type
    if checkpoint_type == "auto":
        if _is_sharded_safetensors_index_checkpoint(checkpoint_path):
            checkpoint_type = "single"
        else:
            ext = _get_checkpoint_extension(checkpoint_path)
            if ext in (".pt", ".pth", ".safetensors"):
                checkpoint_type = "single"
            else:
                checkpoint_type = "distributed"

    if checkpoint_type == "single":
        state_dict = load_single_checkpoint(
            checkpoint_path=checkpoint_path,
            local_cache_dir=local_cache_dir,
            credential_path=credential_path,
            map_location=map_location,
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
