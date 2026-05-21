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

"""CD-only verification for README inference commands that touch HF assets.

The test intentionally runs through ``docker run`` from the host. It keeps the
network-dependent Hugging Face checks out of regular CI while exercising the
same prebuilt container path users follow from the README.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

try:
    import pytest as _pytest
except ModuleNotFoundError:  # pragma: no cover - direct script mode in CD.
    _pytest = None  # ty:ignore[invalid-assignment]

if _pytest is not None:
    pytestmark = [_pytest.mark.manual, _pytest.mark.slow]
else:
    pytestmark = ()


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "ghcr.io/nvidia/flashdreams:base-v0.3-20260424-55bd566"
ENABLE_ENV_VAR = "FLASHDREAMS_HF_ACCESS_TEST"
README_COMMANDS = (
    (
        "alpadreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
        "uv run flashdreams-run "
        "alpadreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf "
        "--example-data True --total-blocks 20 --no-instantiate",
    ),
    (
        "wan21-i2v-14b-480p",
        "uv run flashdreams-run wan21-i2v-14b-480p --no-instantiate",
    ),
)


def _skip(message: str) -> None:
    if _pytest is not None:
        _pytest.skip(message)
    print(f"SKIP: {message}")


def _require_enabled() -> bool:
    if os.getenv(ENABLE_ENV_VAR) == "1":
        return True
    _skip(f"set {ENABLE_ENV_VAR}=1 to run the HF access verification")
    return False


def _require_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise AssertionError("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required")
    return token


def _cache_dir(env_var: str, leaf: str) -> Path:
    default_root = Path("~/.cache/flashdreams-hf-access-test").expanduser()
    return Path(os.getenv(env_var, str(default_root / leaf))).expanduser()


def _docker_env(token: str) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    env.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return env


def _docker_command(image: str, cache_dirs: dict[str, Path]) -> list[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "-v",
        f"{REPO_ROOT}:/workspace/flashdreams",
        "-v",
        f"{cache_dirs['uv']}:/root/.cache/uv",
        "-v",
        f"{cache_dirs['hf']}:/root/.cache/huggingface",
        "-v",
        f"{cache_dirs['flashdreams']}:/root/.cache/flashdreams",
        "-v",
        f"{cache_dirs['triton']}:/root/.cache/triton",
        "-e",
        "HF_TOKEN",
        "-e",
        "HUGGING_FACE_HUB_TOKEN",
        "-e",
        "HF_HOME=/root/.cache/huggingface",
        "-e",
        "FLASHDREAMS_CACHE_DIR=/root/.cache/flashdreams",
        "-e",
        "TRITON_CACHE_DIR=/root/.cache/triton",
        "-e",
        "UV_LINK_MODE=copy",
        "-e",
        "UV_PROJECT_ENVIRONMENT=/tmp/flashdreams-venv",
        "-w",
        "/workspace/flashdreams",
    ]
    netrc = Path.home() / ".netrc"
    if netrc.exists():
        cmd.extend(["-v", f"{netrc}:/root/.netrc:ro"])
    cmd.extend([image, "bash", "-lc", _inner_script()])
    return cmd


def _inner_script() -> str:
    command_block = "\n".join(
        f"echo '[hf-access] resolving README command: {name}'\n{command}"
        for name, command in README_COMMANDS
    )
    return f"""
set -euo pipefail

uv venv --clear
uv sync --frozen --extra runners

{command_block}

uv run python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download

from flashdreams.recipes.alpadreams.config import ALPADREAMS_RUNNERS
from flashdreams.recipes.alpadreams.runner import (
    DEFAULT_EXAMPLE_DATA_UUID_1V,
    _ensure_hf_single_view_example_data_synced,
)
from wan21.config import RUNNER_WAN21_I2V_14B_480P


def parse_hf_file_url(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    if parsed.netloc.lower().removeprefix("www.") != "huggingface.co":
        raise AssertionError(f"expected a Hugging Face URL, got {{url!r}}")
    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 5 or parts[2] not in ("blob", "resolve"):
        raise AssertionError(f"unsupported Hugging Face file URL: {{url}}")
    repo_id = f"{{parts[0]}}/{{parts[1]}}"
    revision = parts[3]
    filename = "/".join(parts[4:])
    return repo_id, filename, revision


def visible_repo_files(
    api: HfApi,
    *,
    repo_id: str,
    filename: str,
    revision: str,
    repo_type: str | None = None,
) -> set[str]:
    parent = str(PurePosixPath(filename).parent)
    path_in_repo = "" if parent == "." else parent
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
        recursive=False,
    )
    return {{
        entry.path
        for entry in entries
        if entry.__class__.__name__ == "RepoFile"
    }}


def assert_hf_file_visible(api: HfApi, url: str, *, repo_type: str | None = None):
    repo_id, filename, revision = parse_hf_file_url(url)
    files = visible_repo_files(
        api,
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type=repo_type,
    )
    if filename not in files:
        raise AssertionError(f"{{filename}} not visible in {{repo_id}}@{{revision}}")
    print(f"[hf-access] visible: {{repo_id}}/{{filename}}")
    return repo_id, filename, revision


token = os.environ.get("HF_TOKEN") or os.environ["HUGGING_FACE_HUB_TOKEN"]
api = HfApi(token=token)
api.whoami(token=token)
print("[hf-access] authenticated to Hugging Face")

alpa_slug = "alpadreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
alpa = ALPADREAMS_RUNNERS[alpa_slug].pipeline
alpa_paths = [
    alpa.image_encoder.checkpoint_path,
    alpa.encoder.checkpoint_path,
    alpa.decoder.checkpoint_path,
    alpa.diffusion_model.transformer.checkpoint_path,
]
for path in alpa_paths:
    assert_hf_file_visible(api, path)

hdmaps, first_frames = _ensure_hf_single_view_example_data_synced(
    DEFAULT_EXAMPLE_DATA_UUID_1V
)
for path in (*hdmaps, *first_frames):
    if not path.exists() or path.stat().st_size <= 0:
        raise AssertionError(f"downloaded example asset is empty or missing: {{path}}")
    print(f"[hf-access] downloaded example asset: {{path.name}}")

wan = RUNNER_WAN21_I2V_14B_480P.pipeline
wan_index_url = wan.diffusion_model.transformer.checkpoint_path
repo_id, filename, revision = assert_hf_file_visible(api, wan_index_url)
index_parent = str(PurePosixPath(filename).parent)
index_path = hf_hub_download(
    repo_id=repo_id,
    filename=PurePosixPath(filename).name,
    subfolder=None if index_parent == "." else index_parent,
    revision=revision,
    token=token,
)
with open(index_path) as f:
    index = json.load(f)
weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise AssertionError(f"invalid sharded checkpoint index: {{index_path}}")
print(f"[hf-access] downloaded Wan I2V checkpoint index with {{len(weight_map)}} tensors")

image_encoder_id = wan.image_encoder.model_id_or_local_path
api.model_info(image_encoder_id, token=token)
print(f"[hf-access] visible: {{image_encoder_id}}")

print("[hf-access] README HF inference asset verification passed")
PY
"""


def test_readme_hf_inference_assets_in_docker() -> None:
    if not _require_enabled():
        return
    if shutil.which("docker") is None:
        raise AssertionError("docker is required for the HF access verification")

    token = _require_hf_token()
    image = os.getenv("FLASHDREAMS_TEST_IMAGE", DEFAULT_IMAGE)
    cache_dirs = {
        "uv": _cache_dir("FLASHDREAMS_UV_CACHE_DIR", "uv"),
        "hf": _cache_dir("FLASHDREAMS_HF_CACHE_DIR", "huggingface"),
        "flashdreams": _cache_dir("FLASHDREAMS_CACHE_DIR", "flashdreams"),
        "triton": _cache_dir("FLASHDREAMS_TRITON_CACHE_DIR", "triton"),
    }
    for cache_dir in cache_dirs.values():
        cache_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        _docker_command(image, cache_dirs),
        check=True,
        env=_docker_env(token),
    )


if __name__ == "__main__":
    test_readme_hf_inference_assets_in_docker()
