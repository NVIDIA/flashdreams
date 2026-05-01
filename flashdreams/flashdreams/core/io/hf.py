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

"""Hugging Face helpers shared across encoders.

The main goal is to decide when ``from_pretrained`` should be called with
``local_files_only=True`` so multi-rank loads of fully cached repos do not
trip HF's per-IP 429 rate limit.
"""

from __future__ import annotations

import os


def _str2bool(v: str | bool) -> bool:
    """Parse the usual yes/no/true/false/1/0 strings into a bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError(f"Boolean value expected, got {v!r}")


def _hf_repo_is_cached(repo_id: str) -> bool:
    """Return True if ``repo_id`` is already in the HF hub cache.

    Probes a few common root-level files (``config.json``,
    ``model_index.json``, ``tokenizer_config.json``,
    ``preprocessor_config.json``); the right one depends on whether the
    repo is a plain transformers model, a Diffusers pipeline, or a
    tokenizer/processor-only repo. If any one is present, sibling assets
    downloaded with it also load fine in offline mode.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False

    candidates = (
        "config.json",
        "model_index.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    )
    for filename in candidates:
        try:
            if try_to_load_from_cache(repo_id, filename) is not None:
                return True
        except Exception:
            continue
    return False


def should_use_local_files_only(repo_id_or_path: str) -> bool:
    """Decide whether to pass ``local_files_only=True`` to ``from_pretrained``.

    True if any of the following hold:

    - ``repo_id_or_path`` is a local directory.
    - ``HF_HUB_OFFLINE`` is truthy.
    - ``LOCAL_FILES_ONLY`` is truthy (legacy).
    - The repo is already cached locally.
    """
    return (
        os.path.isdir(repo_id_or_path)
        or _str2bool(os.getenv("HF_HUB_OFFLINE", "false"))
        or _str2bool(os.getenv("LOCAL_FILES_ONLY", "false"))
        or _hf_repo_is_cached(repo_id_or_path)
    )
