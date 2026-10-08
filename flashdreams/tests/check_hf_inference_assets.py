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

"""Verify Hugging Face access to README inference assets.

Resolves the asset paths referenced by the README's headline recipes
(``omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf`` and
``wan21-i2v-14b-480p``) and downloads one small file per HF repo using
``HF_TOKEN``. Forces an authenticated round-trip, so gated-repo and
token-scope regressions surface here rather than at user runtime.

Usage::

    HF_TOKEN=hf_... uv run --no-sync python flashdreams/tests/check_hf_inference_assets.py
"""

from __future__ import annotations

import os
import sys
from urllib.parse import unquote, urlparse

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from omnidreams.config import OMNIDREAMS_RUNNERS
from wan21.config import RUNNER_WAN21_I2V_14B_480P

OMNIDREAMS_SLUG = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"

# (component-attr, url-attr) pairs walked off each pipeline. Pipelines use
# recipe-specific subclasses whose abstract bases do not declare these
# attrs, so the script reads them via ``getattr`` instead of typed access.
_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("image_encoder", "checkpoint_path"),
    ("image_encoder", "model_id_or_local_path"),
    ("encoder", "checkpoint_path"),
    ("decoder", "checkpoint_path"),
)

# Small files commonly present in HF model repos, tried in order. The first
# one that exists is downloaded; if none exist, the repo's referenced file
# is downloaded directly only if it is itself small (``.json``).
_PROBE_FILES = ("config.json", "model_index.json", "README.md")


def _parse_hf_url(url: str) -> tuple[str, str, str] | None:
    """Return ``(repo_id, filename, revision)`` for HF blob/resolve URLs."""
    parsed = urlparse(url)
    if parsed.netloc.lower().removeprefix("www.") != "huggingface.co":
        return None
    parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 5 or parts[2] not in ("blob", "resolve"):
        return None
    return f"{parts[0]}/{parts[1]}", "/".join(parts[4:]), parts[3]


def _download(*, repo_id: str, filename: str, revision: str, token: str) -> str:
    return hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision, token=token
    )


def _probe(repo_id: str, revision: str, *, token: str) -> str:
    """Download the first existing ``_PROBE_FILES`` entry; raise otherwise."""
    last: Exception | None = None
    for name in _PROBE_FILES:
        try:
            return _download(
                repo_id=repo_id, filename=name, revision=revision, token=token
            )
        except EntryNotFoundError as exc:
            last = exc
    raise AssertionError(
        f"no probe file ({', '.join(_PROBE_FILES)}) downloadable "
        f"from {repo_id}@{revision}: {last}"
    )


def _verify(label: str, url_or_id: str, *, token: str, seen: set[str]) -> None:
    """Verify HF access for ``url_or_id``; dedupes via ``seen``."""
    parsed = _parse_hf_url(url_or_id)
    if parsed is not None:
        repo_id, filename, revision = parsed
        key = f"{repo_id}@{revision}"
        if key in seen:
            print(f"  {label}: {key} (already verified)")
            return
        seen.add(key)
        if filename.endswith(".json"):
            path = _download(
                repo_id=repo_id, filename=filename, revision=revision, token=token
            )
        else:
            path = _probe(repo_id, revision, token=token)
    elif "/" in url_or_id and not url_or_id.startswith(("http", "s3", "/")):
        key = f"{url_or_id}@main"
        if key in seen:
            print(f"  {label}: {key} (already verified)")
            return
        seen.add(key)
        path = _probe(url_or_id, "main", token=token)
    else:
        print(f"  {label}: skipped non-HF path {url_or_id!r}")
        return
    print(f"  {label}: ok -> {path}")


def _pipeline_urls(pipeline: object) -> list[tuple[str, str]]:
    """Collect ``(label, url)`` pairs for every HF asset on ``pipeline``."""
    seen_labels: set[str] = set()
    urls: list[tuple[str, str]] = []
    for component, attr in _COMPONENTS:
        if component in seen_labels:
            continue
        cfg = getattr(pipeline, component, None)
        if cfg is None:
            continue
        url = getattr(cfg, attr, None)
        if isinstance(url, str) and url:
            urls.append((component, url))
            seen_labels.add(component)
    diffusion = getattr(pipeline, "diffusion_model", None)
    transformer = getattr(diffusion, "transformer", None) if diffusion else None
    url = getattr(transformer, "checkpoint_path", None) if transformer else None
    if isinstance(url, str) and url:
        urls.append(("transformer", url))
    return urls


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required", file=sys.stderr)
        return 1

    seen: set[str] = set()
    recipes: tuple[tuple[str, object], ...] = (
        (OMNIDREAMS_SLUG, OMNIDREAMS_RUNNERS[OMNIDREAMS_SLUG].pipeline),
        ("wan21-i2v-14b-480p", RUNNER_WAN21_I2V_14B_480P.pipeline),
    )
    for slug, pipeline in recipes:
        print(f"recipe: {slug}")
        for label, url in _pipeline_urls(pipeline):
            _verify(label, url, token=token, seen=seen)

    print("HF inference asset verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
