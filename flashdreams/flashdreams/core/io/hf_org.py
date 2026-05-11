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

"""Route ``nvidia/omni-dreams-*`` Hugging Face URLs to a configurable mirror.

flashdreams hardcodes the canonical ``nvidia/omni-dreams-models`` org for the
omni-dreams alpadreams DiT URLs. Lighthouse adopters who only have access to
``nvidia-omni-dreams-lha/omni-dreams-*`` set
``OMNI_DREAMS_HF_ORG=nvidia-omni-dreams-lha`` and the rewrite below substitutes
the org segment at download time.

The rewriter is a no-op when the env var is unset, equal to ``"nvidia"``, or
when the URL does not contain a ``nvidia/omni-dreams-*`` substring -- so it is
safe to apply unconditionally on every HF URL flashdreams resolves. The same
env var is honoured by omni-dreams's own scene fetcher, so users only set it
once.
"""

from __future__ import annotations

import os
import re
from typing import Final

OMNI_DREAMS_HF_ORG_ENV_VAR: Final[str] = "OMNI_DREAMS_HF_ORG"
DEFAULT_OMNI_DREAMS_HF_ORG: Final[str] = "nvidia"

# Match the canonical ``nvidia/omni-dreams-{models,samples,scenes}`` repo id
# wherever it appears (full HF URLs and bare repo ids both work). Anchored on
# word boundaries so unrelated nvidia/* repos (e.g. ``nvidia/Cosmos-Reason1-7B``)
# pass through untouched.
_OMNI_DREAMS_REPO_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bnvidia/omni-dreams-(models|samples|scenes)\b"
)


def resolve_omni_dreams_hf_org() -> str:
    """Return the configured omni-dreams HF org, defaulting to ``"nvidia"``."""
    return os.environ.get(OMNI_DREAMS_HF_ORG_ENV_VAR, DEFAULT_OMNI_DREAMS_HF_ORG)


def rewrite_omni_dreams_hf_url(url: str) -> str:
    """Substitute ``nvidia/omni-dreams-{models,samples,scenes}`` with
    ``<org>/omni-dreams-{...}``.

    ``<org>`` is read from the ``OMNI_DREAMS_HF_ORG`` env var. No-op when the
    env var is unset, equal to ``"nvidia"``, or when ``url`` contains no
    matching omni-dreams substring.
    """
    org = resolve_omni_dreams_hf_org()
    if org == DEFAULT_OMNI_DREAMS_HF_ORG:
        return url
    return _OMNI_DREAMS_REPO_PATTERN.sub(
        lambda m: f"{org}/omni-dreams-{m.group(1)}", url
    )
