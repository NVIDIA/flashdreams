# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve which Hugging Face org hosts the omni-dreams repos.

Both demo paths (``interactive_drive`` and ``webrtc.server``) fetch
``<org>/omni-dreams-scenes`` from here. Org defaults to ``"nvidia"`` and is
overridable via ``--hf-org`` (CLI) or ``OMNI_DREAMS_HF_ORG``; entry points
stamp the env var in ``main()`` so resolution stays centralised here.
"""

from __future__ import annotations

import os
import re
from typing import Final, Literal

DEFAULT_HF_ORG: Final[str] = "nvidia"
ENV_VAR: Final[str] = "OMNI_DREAMS_HF_ORG"

RepoKind = Literal["scenes"]

# Internal: every kind of omni-dreams repo we expose. Keep this sorted so the
# rewriter regex below stays deterministic.
_KINDS: Final[tuple[RepoKind, ...]] = ("scenes",)

# Match NVIDIA-org omni-dreams scene URLs. Anchored on the
# ``nvidia/omni-dreams-`` prefix so unrelated HF URLs pass through untouched.
_NVIDIA_OMNI_DREAMS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bnvidia/omni-dreams-(scenes)\b"
)


def resolve_hf_org(cli_value: str | None = None) -> str:
    """Pick the HF org by precedence: ``cli_value`` > ``OMNI_DREAMS_HF_ORG`` > :data:`DEFAULT_HF_ORG`."""
    if cli_value:
        return cli_value
    return os.environ.get(ENV_VAR, DEFAULT_HF_ORG)


def hf_repo(*, kind: RepoKind, org: str | None = None) -> str:
    """Fully-qualified repo id for an omni-dreams component (``org`` defaults to :func:`resolve_hf_org`)."""
    if kind not in _KINDS:
        raise ValueError(
            f"unknown omni-dreams repo kind {kind!r}; expected one of {list(_KINDS)}"
        )
    actual_org = org or resolve_hf_org()
    return f"{actual_org}/omni-dreams-{kind}"


def rewrite_omni_dreams_urls(text: str, org: str | None = None) -> str:
    """Rewrite ``nvidia/omni-dreams-scenes`` substrings in ``text`` to ``<org>/...``.

    No-op when ``org`` is ``"nvidia"``; lets canonical docs keep NVIDIA URLs
    while a non-default org is rewritten on parse.
    """
    actual_org = org or resolve_hf_org()
    if actual_org == DEFAULT_HF_ORG:
        return text
    return _NVIDIA_OMNI_DREAMS_PATTERN.sub(
        lambda match: f"{actual_org}/omni-dreams-{match.group(1)}", text
    )


def apply_cli_to_env(cli_value: str | None) -> str:
    """Stamp the resolved org into the env var so later imports see it; returns the org.

    Call once in ``main()`` after argparse, before the fetching module graph runs.
    """
    org = resolve_hf_org(cli_value)
    os.environ[ENV_VAR] = org
    return org


def describe_hf_access_state() -> str:
    """Summarise the HF auth + org env vars for 401/403/404 error messages.

    Surfaces the two usual culprits: ``HF_TOKEN`` set under the wrong name and
    ``OMNI_DREAMS_HF_ORG`` left at the default when another org is needed.
    """
    # huggingface_hub honours HF_TOKEN (current) and HUGGING_FACE_HUB_TOKEN
    # (legacy); report the legacy name only when it's the only one set.
    if os.environ.get("HF_TOKEN"):
        token_line = "  HF_TOKEN: set"
    elif os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        token_line = "  HUGGING_FACE_HUB_TOKEN: set"
    else:
        token_line = "  HF_TOKEN: NOT SET"
    org_env = os.environ.get(ENV_VAR)
    resolved = resolve_hf_org()
    if org_env is None:
        org_line = f"  {ENV_VAR}: not set (scene fetches default to org {resolved!r})"
    else:
        org_line = f"  {ENV_VAR}: {org_env!r} (scene fetches go to org {resolved!r})"
    return "\n".join(
        [
            "Detected environment:",
            token_line,
            org_line,
        ]
    )


def hf_access_hint(repo_id: str, url: str | None = None) -> str:
    """Multi-line diagnostic for an HF auth/access failure.

    Combines :func:`describe_hf_access_state` with README setup steps.
    ``repo_id`` shows which org the fetch hit; ``url`` is the optional file URL.
    """
    header = f"Hugging Face refused access to repo {repo_id!r}" + (
        f" while fetching {url}." if url else "."
    )
    return "\n".join(
        [
            header,
            "",
            describe_hf_access_state(),
            "",
            "Most common fixes:",
            "  - If you use a non-default authorized org, set OMNI_DREAMS_HF_ORG",
            "    or pass --hf-org so scene URLs are routed away from the",
            "    canonical nvidia/* repos.",
            "  - For direct nvidia access, export HF_TOKEN and request access to",
            "    https://huggingface.co/datasets/nvidia/omni-dreams-scenes first.",
            "",
            "See integrations/omnidreams/README.md and "
            "integrations/omnidreams/omnidreams/interactive_drive/README.md "
            "for the full setup flow.",
        ]
    )
