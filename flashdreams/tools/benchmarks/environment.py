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

"""Environment metadata for local benchmark manifests."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "FLASHDREAMS_CACHE_DIR",
    "HF_HOME",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
)
_SECRET_TOKENS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def collect_environment(repo_root: Path) -> dict[str, Any]:
    """Collect best-effort hardware, software, and repository metadata."""
    return {
        "collected_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "git": _git_metadata(repo_root),
        "torch": _torch_metadata(),
        "nvidia_smi": _nvidia_smi_metadata(),
        "environment": _selected_env(os.environ),
    }


def _git_metadata(repo_root: Path) -> dict[str, str | bool | None]:
    return {
        "branch": _run_text(
            ("git", "rev-parse", "--abbrev-ref", "HEAD"), cwd=repo_root
        ),
        "commit": _run_text(("git", "rev-parse", "HEAD"), cwd=repo_root),
        "describe": _run_text(
            ("git", "describe", "--always", "--dirty"), cwd=repo_root
        ),
        "dirty": _git_dirty(repo_root),
        "remote": _run_text(("git", "remote", "get-url", "origin"), cwd=repo_root),
    }


def _git_dirty(repo_root: Path) -> bool | None:
    output = _run_text(("git", "status", "--porcelain"), cwd=repo_root)
    if output is None:
        return None
    return bool(output.strip())


def _torch_metadata() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - metadata should never break a run
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    metadata: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": None,
        "devices": [],
    }
    try:
        metadata["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as exc:  # noqa: BLE001
        metadata["cudnn_error"] = f"{type(exc).__name__}: {exc}"

    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": props.total_memory / 1024**3,
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        metadata["devices"] = devices
    return metadata


def _nvidia_smi_metadata() -> list[dict[str, str]] | dict[str, str]:
    if shutil.which("nvidia-smi") is None:
        return {"available": "false"}
    output = _run_text(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        cwd=None,
        timeout_s=5.0,
    )
    if output is None:
        return {"available": "false"}
    gpus = []
    for index, line in enumerate(output.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, driver_version, memory_total_mib = parts
        gpus.append(
            {
                "index": str(index),
                "name": name,
                "driver_version": driver_version,
                "memory_total_mib": memory_total_mib,
            }
        )
    return gpus


def _selected_env(env: os._Environ[str]) -> dict[str, str | None]:
    selected: dict[str, str | None] = {}
    for key in _ENV_KEYS:
        if key in env:
            selected[key] = _redact_env_value(key, env[key])
    return selected


def _redact_env_value(key: str, value: str) -> str:
    upper_key = key.upper()
    if any(token in upper_key for token in _SECRET_TOKENS):
        return "<redacted>"
    return value


def _run_text(
    command: tuple[str, ...],
    *,
    cwd: Path | None,
    timeout_s: float = 2.0,
) -> str | None:
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
