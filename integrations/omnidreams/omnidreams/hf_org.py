# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve which Hugging Face organisation hosts the omni-dreams repos.

Used by both demo paths in this package:

  * ``omnidreams.interactive_drive`` (the desktop demo) fetches
    ``<org>/omni-dreams-scenes`` USDZ scenes via
    ``omnidreams-prepare`` / first-launch auto-stage.
  * ``omnidreams.webrtc.server`` fetches the same dataset for its
    session-managed scene cache under ``FLASHDREAMS_CACHE_DIR``.

The flashdreams world-model backend owns its own model/checkpoint fetches.

The default org is ``"nvidia"``. Environments that use another authorized
Hugging Face org can flip every fetch to that org by either:

  * passing ``--hf-org <org>`` on the CLI (interactive-drive only), **or**
  * exporting ``OMNI_DREAMS_HF_ORG=<org>``.

Every entry point that adds the CLI flag pokes the env var in ``main()``
before the rest of the module graph imports anything that fetches, which
keeps the resolution centralised here -- no thread-the-arg-through plumbing.
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
    """Pick the HF org for the current process.

    Precedence (highest first):

    1. an explicit ``cli_value`` (e.g. from ``--hf-org`` on the command
       line; ``None`` means the flag wasn't passed),
    2. the ``OMNI_DREAMS_HF_ORG`` environment variable,
    3. the built-in :data:`DEFAULT_HF_ORG`.
    """
    if cli_value:
        return cli_value
    return os.environ.get(ENV_VAR, DEFAULT_HF_ORG)


def hf_repo(*, kind: RepoKind, org: str | None = None) -> str:
    """Return the fully-qualified repo id for an omni-dreams component.

    ``kind`` is the logical role (currently only ``"scenes"``); ``org``
    defaults to :func:`resolve_hf_org`'s result if not supplied.
    """
    if kind not in _KINDS:
        raise ValueError(
            f"unknown omni-dreams repo kind {kind!r}; expected one of {list(_KINDS)}"
        )
    actual_org = org or resolve_hf_org()
    return f"{actual_org}/omni-dreams-{kind}"


def rewrite_omni_dreams_urls(text: str, org: str | None = None) -> str:
    """Rewrite every ``nvidia/omni-dreams-scenes`` substring
    inside ``text`` to ``<org>/omni-dreams-{...}``.

    A no-op when ``org`` resolves to ``"nvidia"`` (the canonical default).
    Used by the scene setup flow so a caller who flips
    ``OMNI_DREAMS_HF_ORG`` doesn't have to ship a parallel yaml file --
    canonical docs/paths keep the NVIDIA URLs and setup rewrites them on parse.
    """
    actual_org = org or resolve_hf_org()
    if actual_org == DEFAULT_HF_ORG:
        return text
    return _NVIDIA_OMNI_DREAMS_PATTERN.sub(
        lambda match: f"{actual_org}/omni-dreams-{match.group(1)}", text
    )


def apply_cli_to_env(cli_value: str | None) -> str:
    """Stamp the resolved org into the env var so subsequent imports see it.

    Entry-point ``main()`` functions call this once after argparse, before
    the rest of the module graph (manifest loader, hf_repo lookups, etc.)
    runs. Returns the resolved org for any caller that wants to log it.
    """
    org = resolve_hf_org(cli_value)
    os.environ[ENV_VAR] = org
    return org


def describe_hf_access_state() -> str:
    """Render a short, human-readable summary of the HF auth + org env vars.

    Used by error wrappers around HF fetches so that 401 / 403 / 404 failures
    surface the two knobs that almost always cause them — ``HF_TOKEN`` set
    under the wrong name, and ``OMNI_DREAMS_HF_ORG`` left at the default
    when the user is actually entitled to a different authorized org.
    """
    # ``huggingface_hub`` honours both ``HF_TOKEN`` (current) and
    # ``HUGGING_FACE_HUB_TOKEN`` (legacy). Report ``HF_TOKEN`` by default;
    # only mention the legacy name when that's the only one set, so users
    # who picked it see "set" instead of a misleading "NOT SET".
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
    """Build a multi-line diagnostic message for an HF auth/access failure.

    Combines :func:`describe_hf_access_state` with actionable next steps
    pointing at the README's setup instructions. ``url`` is optional context
    for the line item (the file URL that failed); ``repo_id`` is the resolved
    repo string so the user can tell at a glance whether the fetch went to
    the canonical ``nvidia/*`` org or to a mirror.
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
