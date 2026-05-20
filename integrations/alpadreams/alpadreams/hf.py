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

"""Hugging Face repository helpers for Omni Dreams assets."""

from __future__ import annotations

import os

OMNI_DREAMS_HF_ORG_ENV_VAR = "OMNI_DREAMS_HF_ORG"
"""Set before import/run to choose the HF org for Omni Dreams assets."""

DEFAULT_OMNI_DREAMS_HF_ORG = "nvidia"


def omni_dreams_hf_org() -> str:
    """Return the configured HF org for Omni Dreams assets."""
    return (
        os.getenv(OMNI_DREAMS_HF_ORG_ENV_VAR, DEFAULT_OMNI_DREAMS_HF_ORG).strip("/")
        or DEFAULT_OMNI_DREAMS_HF_ORG
    )


def omni_dreams_hf_repo(repo_name: str) -> str:
    """Return an Omni Dreams HF repo id under the configured org."""
    return f"{omni_dreams_hf_org()}/{repo_name}"


def omni_dreams_hf_url(
    repo_name: str,
    path: str,
    *,
    repo_type: str | None = None,
) -> str:
    """Return a browser/download URL for an Omni Dreams HF repo path."""
    base = "https://huggingface.co"
    if repo_type == "dataset":
        base += "/datasets"
    return f"{base}/{omni_dreams_hf_repo(repo_name)}/{path.lstrip('/')}"
