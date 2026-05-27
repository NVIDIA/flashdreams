#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One-shot setup helper for the interactive_drive demo.

Pulls a sample scene USDZ from the ``nvidia/omni-dreams-scenes``
Hugging Face dataset and optionally pre-warms the Cosmos-Reason1 text
encoder used by the flashdreams world-model path.
Re-running is safe: any asset already present on disk is skipped.

Scene staging goes through Hugging Face; set ``HF_TOKEN`` with access to
``nvidia/omni-dreams-scenes`` before running this helper.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from interactive_drive.cli import _SAMPLE_ROOT as _REPO_ROOT
from interactive_drive.hf_org import (
    DEFAULT_HF_ORG,
    apply_cli_to_env,
    hf_repo,
)


def scenes_repo() -> str:
    """Resolve the scenes HF dataset repo from the env var (set by
    ``apply_cli_to_env`` in :func:`main`). Lifted to a function so the
    value picks up any CLI override applied after this module is imported.
    """
    return hf_repo(kind="scenes")


def hf_prewarm_urls() -> tuple[str, ...]:
    """Hugging Face files the flashdreams-backed runtime lazily downloads."""
    return ()


# Full HF repos snapshot-downloaded up front. The Cosmos-Reason1 runtime
# text encoder is ~14 GB across several safetensors shards; we pre-fetch
# the whole repo here so the demo's first launch doesn't block on it.
# Users can skip this pre-warm with ``--skip-text-encoder`` and let flashdreams
# pull the repo lazily on first use.
HF_PREWARM_REPOS: tuple[str, ...] = ("nvidia/Cosmos-Reason1-7B",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch demo scenes and pre-warm the Hugging Face cache.",
    )
    parser.add_argument(
        "--scene-uuid",
        default=None,
        help=(
            "Stage only this specific scene UUID from the scenes dataset. "
            "When omitted, every scene currently published is staged "
            "(~1 GiB across all clips). The exact dataset depends on "
            "--hf-org; for the default 'nvidia' org see "
            "https://huggingface.co/datasets/nvidia/omni-dreams-scenes/tree/main/scenes."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download staged scenes even if they already exist on disk.",
    )
    parser.add_argument(
        "--skip-scene",
        action="store_true",
        help="Don't stage any scene USDZ. Use when you already have one locally.",
    )
    parser.add_argument(
        "--skip-hf-prewarm",
        action="store_true",
        help="Skip pre-warming Hugging Face model repos. Assets will still be pulled lazily at runtime.",
    )
    parser.add_argument(
        "--skip-text-encoder",
        action="store_true",
        help=(
            "Skip pre-warming the Cosmos-Reason1 runtime text encoder (~14 GB). "
            "The runtime will download it lazily on first use."
        ),
    )
    parser.add_argument(
        "--hf-org",
        default=None,
        metavar="ORG",
        help=(
            "Hugging Face org that hosts the omni-dreams repos (models /"
            f" samples / scenes). Defaults to {DEFAULT_HF_ORG!r}."
            " Equivalent to setting OMNI_DREAMS_HF_ORG; the flag wins"
            " when both are present."
        ),
    )
    return parser.parse_args()


def repo_root() -> Path:
    return _REPO_ROOT


def info(message: str) -> None:
    print(f"[prepare] {message}")


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def scene_path(root: Path, uuid: str) -> Path:
    """Absolute path the demo expects the USDZ to live at."""
    return root / "assets" / "scenes" / f"{uuid}.usdz"


def prewarm_huggingface_cache(
    urls: tuple[str, ...],
    repos: tuple[str, ...] = (),
) -> None:
    """Pre-download the HF files + full repos referenced by the default manifest.

    File URLs go through ``WorldModelManifest``'s parser (same code path used at
    runtime); full repo IDs are materialised via ``snapshot_download`` so that
    ``from_pretrained(repo_id)`` calls at runtime don't touch the network.
    """
    try:
        from interactive_drive.world_model.manifest import download_hf_file
    except Exception as exc:  # pragma: no cover - interactive_drive must be importable
        raise RuntimeError(
            "Unable to import interactive_drive.world_model.manifest; run "
            "`uv sync --package omnidreams-interactive-drive` from the "
            "flashdreams workspace root first."
        ) from exc

    for url in urls:
        info(f"Pre-warming HF cache: {url}")
        local = download_hf_file(url)
        info(f"  \u2192 {local}")

    if not repos:
        return

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Unable to import huggingface_hub.snapshot_download; run "
            "`uv sync --package omnidreams-interactive-drive` from the "
            "flashdreams workspace root first."
        ) from exc

    for repo_id in repos:
        info(f"Pre-warming HF repo snapshot: {repo_id}")
        local = snapshot_download(repo_id=repo_id)
        info(f"  \u2192 {local}")


def list_available_scene_uuids() -> list[str]:
    """Enumerate every ``scenes/<uuid>.usdz`` file published to the scenes
    dataset. The exact repo id depends on the resolved HF org; see
    :func:`scenes_repo`.

    Returns a sorted list of UUID strings (without the ``scenes/`` prefix or
    ``.usdz`` suffix). Requires ``HF_TOKEN`` to be set because the dataset is
    private.
    """
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Unable to import huggingface_hub.HfApi; run "
            "`uv sync --package omnidreams-interactive-drive` from the "
            "flashdreams workspace root first."
        ) from exc

    files = HfApi().list_repo_files(repo_id=scenes_repo(), repo_type="dataset")
    prefix = "scenes/"
    suffix = ".usdz"
    uuids = [
        path[len(prefix) : -len(suffix)]
        for path in files
        if path.startswith(prefix) and path.endswith(suffix)
    ]
    return sorted(uuids)


def stage_scene(root: Path, uuid: str, *, force: bool) -> Path:
    """Download the scene USDZ from the HF dataset and materialise it at
    ``assets/scenes/<uuid>.usdz`` so the ``interactive_drive --scene ...`` argument
    can find it via its usual relative path.

    ``hf_hub_download`` is used internally, so the file is also cached under
    ``~/.cache/huggingface/hub/datasets--<org>--omni-dreams-scenes`` for the
    resolved org and any subsequent pulls of the same UUID are no-ops.
    """
    dest = scene_path(root, uuid)

    if dest.exists() and not force:
        info(f"Scene already staged at {dest} ({human_bytes(dest.stat().st_size)}).")
        return dest

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Unable to import huggingface_hub; run "
            "`uv sync --package omnidreams-interactive-drive` from the "
            "flashdreams workspace root first."
        ) from exc

    repo = scenes_repo()
    info(f"Downloading scene from {repo}: {uuid}.usdz")
    cached = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=f"scenes/{uuid}.usdz",
    )
    # Copy (not symlink) into assets/scenes/ so the path referenced by the
    # demo command line is a real file robust to the HF cache moving.
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    info(f"Staged scene at {dest} ({human_bytes(dest.stat().st_size)}).")
    return dest


def main() -> int:
    args = parse_args()
    root = repo_root()

    # Stamp the resolved HF org into the env var BEFORE the first call to
    # ``scenes_repo()`` / ``hf_prewarm_urls()`` -- those are lazy and read
    # from the env, so this single call routes every fetch below to the
    # right org without explicit threading.
    resolved_org = apply_cli_to_env(args.hf_org)
    if resolved_org != DEFAULT_HF_ORG:
        info(f"Using HF org '{resolved_org}' for omni-dreams repos.")

    # Pre-warm optional HF repos first. If HF_TOKEN is missing we skip
    # everything HF -- without it we can't reach the private scenes repo.
    if args.skip_hf_prewarm:
        info("Skipping Hugging Face cache pre-warm per --skip-hf-prewarm.")
    elif not os.environ.get("HF_TOKEN"):
        info(
            "HF_TOKEN is not set; skipping Hugging Face cache pre-warm. "
            "Export HF_TOKEN and rerun to stage text-encoder assets ahead of time, or "
            "pass --skip-hf-prewarm to silence this message. The runtime "
            "will fetch assets lazily on first use once HF_TOKEN is set."
        )
    else:
        repos_to_prewarm = () if args.skip_text_encoder else HF_PREWARM_REPOS
        if args.skip_text_encoder:
            info("Skipping Cosmos-Reason1 runtime text-encoder pre-warm per --skip-text-encoder.")
        prewarm_huggingface_cache(hf_prewarm_urls(), repos_to_prewarm)

    # Scene USDZ -- required at demo launch time, no lazy fallback.
    if args.skip_scene:
        info("Skipping scene staging per --skip-scene.")
    elif not os.environ.get("HF_TOKEN"):
        info(
            "HF_TOKEN is not set; skipping scene download. Export HF_TOKEN "
            "and rerun, or pass --skip-scene and provide your own USDZ via "
            "the --scene flag to interactive_drive."
        )
    elif args.scene_uuid is not None:
        stage_scene(root, args.scene_uuid, force=args.force)
    else:
        uuids = list_available_scene_uuids()
        info(f"Staging all {len(uuids)} scene(s) from {scenes_repo()}.")
        for i, uuid in enumerate(uuids, start=1):
            info(f"  [{i}/{len(uuids)}] {uuid}")
            stage_scene(root, uuid, force=args.force)

    info("Workspace assets are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
