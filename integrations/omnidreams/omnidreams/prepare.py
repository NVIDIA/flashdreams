#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One-shot setup helper for every ``flashdreams-omnidreams`` demo.

Stages the resources both demo paths share:

* ``nvidia/omni-dreams-scenes`` USDZ archives -> consumed sealed by the
  desktop ``interactive-drive`` demo and unpacked on demand by
  ``omnidreams.webrtc.server`` (both read from the shared cache at
  ``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/``; see
  :mod:`omnidreams.scenes`).
* The Cosmos-Reason1 text encoder used by the flashdreams world-model
  pipeline -- pinned to the same commit as
  :class:`CosmosReason1TextEncoderConfig` so the prewarm files satisfy
  the runtime ``from_pretrained(revision=...)`` call (otherwise the
  ~14 GB warm-up downloads HEAD and the runtime re-fetches at launch).
* The world-model DiT checkpoint(s) the runtime loads from
  ``nvidia/omni-dreams-models`` (see :func:`hf_prewarm_urls`).

Re-running is safe: any asset already present on disk is skipped.
Staging goes through Hugging Face; set ``HF_TOKEN`` with access to both
``nvidia/omni-dreams-scenes`` and ``nvidia/omni-dreams-models`` first.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

from omnidreams.hf_org import DEFAULT_HF_ORG, apply_cli_to_env
from omnidreams.scenes import (
    SCENE_VARIANT_DEFAULT,
    hf_hub_download_scene,
    hf_scenes_repo_id,
    list_available_scene_files,
    local_scene_archive_path,
    normalise_scene_uuid,
)


def hf_prewarm_urls() -> tuple[str, ...]:
    """World-model DiT checkpoint URLs the runtime would otherwise fetch lazily.

    Pulled live off :data:`omnidreams.config.AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS`
    so they track whatever the runtime loads. Pre-warming them here keeps the
    first launch off the multi-GB download path and surfaces gated-repo access
    errors at setup time. ``MISSING`` sentinels (internal-only checkpoints) and
    non-Hugging-Face values are skipped.
    """
    try:
        from omnidreams.config import AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS
    except ImportError as exc:  # pragma: no cover - config must be importable
        raise RuntimeError(
            "Unable to import omnidreams.config to resolve world-model "
            "checkpoint URLs; run `uv sync --package flashdreams-omnidreams` "
            "from the flashdreams workspace root first."
        ) from exc

    urls: list[str] = []
    seen: set[str] = set()
    for value in AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS.values():
        if not value.startswith("https://huggingface.co/"):
            continue
        if value not in seen:
            seen.add(value)
            urls.append(value)
    return tuple(urls)


def _cosmos_reason1_prewarm_targets() -> tuple[tuple[str, str], ...]:
    """``(repo_id, revision)`` tuples for the runtime text encoder.

    Pulled live off :class:`CosmosReason1TextEncoderConfig` so the prewarm
    pins the same commit the runtime loads. The encoder config's
    ``revision`` default is a specific Cosmos-Reason1.1 SFT commit
    (not ``main`` HEAD); without passing it through to
    ``snapshot_download`` the prewarm fetches HEAD and the runtime then
    re-downloads the pinned revision on first launch -- ~14 GB of
    wasted bandwidth. The import is lazy because the cosmos_reason1
    module pulls in torch + transformers.
    """
    try:
        from flashdreams.infra.encoder.text.cosmos_reason1 import (
            CosmosReason1TextEncoderConfig,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Unable to import CosmosReason1TextEncoderConfig; run "
            "`uv sync --package flashdreams-omnidreams` from the "
            "flashdreams workspace root first."
        ) from exc

    config = CosmosReason1TextEncoderConfig()
    return ((config.model_name, config.revision),)


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
            "(all weather variants of each clip). The exact dataset depends "
            "on --hf-org; for the default 'nvidia' org see "
            "https://huggingface.co/datasets/nvidia/omni-dreams-scenes/tree/main/scenes."
        ),
    )
    parser.add_argument(
        "--scene-variant",
        default=None,
        help=(
            "With --scene-uuid, stage only this weather variant "
            "('default', 'rain', 'snow'). When omitted, every published "
            "variant of the selected scene (or of all scenes) is staged."
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
        "--perf",
        action="store_true",
        help=(
            "Sync the pinned third-party CUDA sources required by the perf "
            "manifest's native acceleration (example_world_model_perf.yaml). "
            "Clones them into omnidreams_singleview/3rdparty/; the extension "
            "itself compiles on first run. Requires a source checkout and git."
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


def scene_path(scene_uuid: str, variant: str = SCENE_VARIANT_DEFAULT) -> Path:
    """Absolute path the demo expects a staged USDZ scene variant to live at.

    Shared cache layout under ``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/``;
    see :func:`omnidreams.scenes.local_scene_archive_path` for the exact
    convention. Accepts either a bare UUID or a ``clipgt-<uuid>`` stem.
    """
    return local_scene_archive_path(scene_uuid, variant)


def prewarm_huggingface_cache(
    urls: tuple[str, ...],
    repos: tuple[tuple[str, str], ...] = (),
) -> None:
    """Pre-download the HF files + full repos referenced by the default manifest.

    File URLs go through ``WorldModelManifest``'s parser (same code path used at
    runtime); ``(repo_id, revision)`` pairs are materialised via
    ``snapshot_download`` so that ``from_pretrained(repo_id, revision=...)``
    calls at runtime don't touch the network. ``revision`` must match the
    commit the runtime loads -- a HEAD prewarm with a pinned runtime
    revision ends up re-downloading at first launch.
    """
    try:
        from omnidreams.interactive_drive.world_model.manifest import download_hf_file
    except Exception as exc:  # pragma: no cover - interactive_drive must be importable
        raise RuntimeError(
            "Unable to import omnidreams.interactive_drive.world_model.manifest; run "
            "`uv sync --package flashdreams-omnidreams` from the "
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
            "`uv sync --package flashdreams-omnidreams` from the "
            "flashdreams workspace root first."
        ) from exc

    for repo_id, revision in repos:
        info(f"Pre-warming HF repo snapshot: {repo_id}@{revision[:12]}")
        local = snapshot_download(repo_id=repo_id, revision=revision)
        info(f"  \u2192 {local}")


def stage_scene(
    scene_uuid: str, *, variant: str = SCENE_VARIANT_DEFAULT, force: bool
) -> Path:
    """Download a scene variant's USDZ and copy it into the shared cache.

    Materialises ``clipgt-<uuid>[-<variant>].usdz`` under the scenes cache so
    the demo's ``--scene`` arg points at a stable file. Accepts a bare UUID or
    a ``clipgt-<uuid>`` stem.
    """
    bare_uuid = normalise_scene_uuid(scene_uuid)
    dest = scene_path(bare_uuid, variant)
    archive_name = dest.name

    if dest.exists() and not force:
        info(f"Scene already staged at {dest} ({human_bytes(dest.stat().st_size)}).")
        return dest

    info(f"Downloading scene from {hf_scenes_repo_id()}: {archive_name}")
    cached = hf_hub_download_scene(bare_uuid, variant)
    # Copy (not symlink) into the cache root so the path referenced by
    # the demo command line is a real file robust to the HF cache moving
    # (e.g. user sets HF_HOME between runs).
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    info(f"Staged scene at {dest} ({human_bytes(dest.stat().st_size)}).")
    return dest


def _sync_thirdparty_module() -> ModuleType:
    """Load the standalone ``sync_thirdparty`` tool by path.

    It lives in the ``omnidreams_singleview`` source tree (a sibling of the
    ``omnidreams`` package, not an installed module), so it's loaded from its
    file rather than imported.
    """
    tool_path = (
        Path(__file__).resolve().parents[1]
        / "omnidreams_singleview"
        / "tools"
        / "sync_thirdparty.py"
    )
    if not tool_path.is_file():
        raise SystemExit(
            "--perf requires the omnidreams_singleview sources, which ship "
            f"only in a source checkout; not found at {tool_path}."
        )
    spec = importlib.util.spec_from_file_location(
        "omnidreams_sync_thirdparty", tool_path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {tool_path}.")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the tool's frozen dataclasses can resolve their
    # own module via sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sync_perf_thirdparty(*, force: bool) -> None:
    """Sync the pinned native CUDA sources the perf manifest builds against.

    ``example_world_model_perf.yaml`` (``native_dit_acceleration: required``)
    compiles against CUTLASS, SageAttention, SpargeAttn, and cudnn-frontend.
    Clones them into ``omnidreams_singleview/3rdparty/``; the extension itself
    builds on first run. Needs git + network.
    """
    sync = _sync_thirdparty_module()
    try:
        sources = sync.load_manifest()
        info(f"Syncing {len(sources)} native third-party source(s) for --perf.")
        for result in sync.sync_sources(sources, force=force):
            info(f"  {result.source.name}: {result.commit[:12]} -> {result.path}")
    except sync.ThirdPartySyncError as exc:
        raise SystemExit(f"--perf third-party sync failed: {exc}") from exc


def main() -> int:
    args = parse_args()

    # Stamp the resolved HF org into the env var BEFORE the first call to
    # ``hf_scenes_repo_id()`` / ``hf_prewarm_urls()`` -- those are lazy
    # and read from the env, so this single call routes every fetch
    # below to the right org without explicit threading.
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
        if args.skip_text_encoder:
            info(
                "Skipping Cosmos-Reason1 runtime text-encoder pre-warm per --skip-text-encoder."
            )
            repos_to_prewarm: tuple[tuple[str, str], ...] = ()
        else:
            repos_to_prewarm = _cosmos_reason1_prewarm_targets()
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
    elif args.scene_uuid is not None and args.scene_variant is not None:
        stage_scene(args.scene_uuid, variant=args.scene_variant, force=args.force)
    elif args.scene_uuid is not None:
        # All published variants of the requested scene.
        bare_uuid = normalise_scene_uuid(args.scene_uuid)
        variants = [
            variant
            for uuid, variant in list_available_scene_files()
            if uuid == bare_uuid
        ] or [SCENE_VARIANT_DEFAULT]
        info(
            f"Staging {len(variants)} variant(s) of {bare_uuid} "
            f"from {hf_scenes_repo_id()}: {', '.join(variants)}."
        )
        for variant in variants:
            stage_scene(bare_uuid, variant=variant, force=args.force)
    elif args.scene_variant is not None:
        # That variant across every scene that publishes it.
        scene_files = [
            (uuid, variant)
            for uuid, variant in list_available_scene_files()
            if variant == args.scene_variant
        ]
        info(
            f"Staging {len(scene_files)} scene(s) (variant {args.scene_variant}) "
            f"from {hf_scenes_repo_id()}."
        )
        for i, (uuid, variant) in enumerate(scene_files, start=1):
            info(f"  [{i}/{len(scene_files)}] {uuid} ({variant})")
            stage_scene(uuid, variant=variant, force=args.force)
    else:
        scene_files = list_available_scene_files()
        info(
            f"Staging all {len(scene_files)} scene archive(s) "
            f"from {hf_scenes_repo_id()}."
        )
        for i, (uuid, variant) in enumerate(scene_files, start=1):
            info(f"  [{i}/{len(scene_files)}] {uuid} ({variant})")
            stage_scene(uuid, variant=variant, force=args.force)

    # Independent of Hugging Face (git clones, no HF_TOKEN).
    if args.perf:
        sync_perf_thirdparty(force=args.force)

    info("Workspace assets are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
