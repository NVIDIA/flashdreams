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

Re-running is safe: any asset already present on disk is skipped.
Scene staging goes through Hugging Face; set ``HF_TOKEN`` with access
to ``nvidia/omni-dreams-scenes`` before running this helper.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import zipfile
from pathlib import Path

from omnidreams.hf_org import DEFAULT_HF_ORG, apply_cli_to_env
from omnidreams.scenes import (
    hf_hub_download_scene,
    hf_scenes_repo_id,
    list_available_scene_uuids,
    local_scene_archive_path,
    normalise_scene_uuid,
)

# ---------------------------------------------------------------------------
# PBSS source for the release-candidate "larger map" anonymized USDZs
# ---------------------------------------------------------------------------
# Side-channel for grabbing the four 26.02-based anonymized USDZs Guillermo
# uploaded to ``s3://guillermog/PAI-900/valid-736-with-packed-anonymized-usdz-v3/``
# on 2026-05-28. These are the scenes planned for the OmniDreams OSS release
# but not yet republished to ``nvidia/omni-dreams-scenes``; this is the only
# place to get them today. The 5th interactive-drive demo UUID
# (``065dcac9-...``) is not in the v3 set and is intentionally omitted.
PBSS_ENDPOINT_URL = "https://pdx.s8k.io"
PBSS_REGION = "us-east-1"
PBSS_ACCESS_KEY_ID = "team-alpamayo"
PBSS_SECRET_ENV_VAR = "ALPAMAYO_S3_SECRET"
PBSS_BUCKET = "guillermog"
PBSS_PREFIX = "PAI-900/valid-736-with-packed-anonymized-usdz-v3"
PBSS_SCENE_UUIDS: tuple[str, ...] = (
    "01d503d4-449b-46fc-8d78-9085e70d3554",
    "0b10bce8-61f1-4350-8577-cf3c9493ffc3",
    "0d1fcd2c-ed47-4c72-b756-8e24bce0b9f4",
    "0d76134f-350d-44b5-a694-208e9dab9600",
)


def hf_prewarm_urls() -> tuple[str, ...]:
    """Hugging Face files the flashdreams-backed runtime lazily downloads."""
    return ()


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
    parser.add_argument(
        "--from-pbss-anon-v3",
        action="store_true",
        help=(
            "Stage release-candidate scenes (anonymized, 26.02-base, larger "
            f"maps) from PBSS s3://{PBSS_BUCKET}/{PBSS_PREFIX}/ instead of "
            f"Hugging Face. Requires {PBSS_SECRET_ENV_VAR}. Without "
            "--pbss-count, stages just the 4 original demo UUIDs (the 5th, "
            "065dcac9..., isn't in the v3 set). With --pbss-count N, "
            "auto-discovers and stages the first N UUIDs from the prefix. "
            "With --scene-uuid, stages just that one UUID."
        ),
    )
    parser.add_argument(
        "--pbss-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Only meaningful with --from-pbss-anon-v3. Auto-discover and "
            "stage the first N UUIDs (alphabetical) from the PBSS prefix "
            "instead of the hardcoded 4 demo UUIDs. Each USDZ is ~1.8 GiB; "
            "e.g. N=25 ~= 45 GiB total."
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


def scene_path(scene_uuid: str) -> Path:
    """Absolute path the demo expects a staged USDZ scene to live at.

    Shared cache layout under ``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/``;
    see :func:`omnidreams.scenes.local_scene_archive_path` for the exact
    convention. Accepts either a bare UUID or a ``clipgt-<uuid>`` stem.
    """
    return local_scene_archive_path(scene_uuid)


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


def stage_scene(scene_uuid: str, *, force: bool) -> Path:
    """Download the scene USDZ from the HF dataset and materialise it under
    ``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/clipgt-<uuid>.usdz`` so the
    desktop demo's ``--scene`` arg points at a stable on-disk file.

    The HF download itself is content-addressed by ``huggingface_hub``,
    so subsequent calls with the same UUID -- including the webrtc
    server's ``_ensure_hf_webrtc_scene_synced`` -- are cache hits.

    Accepts either a bare UUID or a ``clipgt-<uuid>`` stem; both
    normalise to the bare form for consistent path / URL building.
    """
    bare_uuid = normalise_scene_uuid(scene_uuid)
    dest = scene_path(bare_uuid)

    if dest.exists() and not force:
        info(f"Scene already staged at {dest} ({human_bytes(dest.stat().st_size)}).")
        return dest

    info(f"Downloading scene from {hf_scenes_repo_id()}: clipgt-{bare_uuid}.usdz")
    cached = hf_hub_download_scene(bare_uuid)
    # Copy (not symlink) into the cache root so the path referenced by
    # the demo command line is a real file robust to the HF cache moving
    # (e.g. user sets HF_HOME between runs).
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    info(f"Staged scene at {dest} ({human_bytes(dest.stat().st_size)}).")
    return dest


# Camera whose first frame doubles as the world-model's ``first_image`` seed.
# Front-wide matches the camera the demo renders by default; cross/tele frames
# are also packed in ``frames/`` but aren't useful as the seed image.
_PBSS_FIRST_IMAGE_CAMERA = "camera_front_wide_120fov"

# Stand-in prompt for repacked PBSS USDZs. The release pipeline uses VLM-
# generated prompts (per Slack); until those land in the bucket alongside
# the USDZs, this placeholder keeps the world-model side from crashing on
# a missing ``clipgt/prompt.txt`` and roughly matches the short caption
# the original demo scenes shipped with.
_PBSS_PLACEHOLDER_PROMPT = "A dashcam view looking forward on a driving scene.\n"


def _repack_pbss_usdz(usdz_path: Path) -> None:
    """Inject ``clipgt/first_image.jpeg`` + ``clipgt/prompt.txt`` into a PBSS USDZ.

    The release-candidate USDZs in ``guillermog/...-anonymized-usdz-v3``
    bundle per-camera first frames under ``frames/<camera>/<ts>.jpeg`` and
    no prompt at all, but the interactive-drive scene loader looks for
    ``clipgt/first_image.*`` + ``clipgt/prompt.txt`` (the layout the
    existing ``omni-dreams-scenes`` HF dataset uses). We bridge that here
    by appending the missing entries to the zip in place -- zip's central
    directory just gets rewritten at the end of the file, the existing
    NRE/mesh/parquet entries are untouched, and the USDZ stays valid.

    Idempotent: if ``clipgt/first_image.*`` already exists in the archive
    we skip everything (a previous repack succeeded, or the archive was
    already in the demo-ready shape).
    """
    with zipfile.ZipFile(usdz_path, "r") as zf:
        names = zf.namelist()
        if any(n.startswith("clipgt/first_image") for n in names):
            return  # already repacked

        # Pick the front-wide first frame; fall back to any frames/* entry
        # so a future PBSS layout change with a different camera key still
        # produces a usable first_image rather than silently leaving the
        # world model with no seed.
        front_wide_frames = sorted(
            n for n in names
            if n.startswith(f"frames/{_PBSS_FIRST_IMAGE_CAMERA}/")
            and n.lower().endswith((".jpeg", ".jpg", ".png"))
        )
        any_frames = sorted(
            n for n in names
            if n.startswith("frames/")
            and n.lower().endswith((".jpeg", ".jpg", ".png"))
        )
        chosen = front_wide_frames[0] if front_wide_frames else (
            any_frames[0] if any_frames else None
        )
        if chosen is None:
            info(
                f"  warning: {usdz_path.name} has no frames/<camera>/*.jpeg; "
                "cannot inject clipgt/first_image. The HD-map rasterizer "
                "will work but the world model won't have a seed image."
            )
            return

        first_image_bytes = zf.read(chosen)
        first_image_ext = Path(chosen).suffix.lower().lstrip(".")
        has_existing_prompt = any(
            n == "clipgt/prompt.txt" or n.startswith("clipgt/prompt_")
            for n in names
        )

    # Reopen in append mode so we don't have to rewrite the 1.8 GiB body.
    with zipfile.ZipFile(usdz_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"clipgt/first_image.{first_image_ext}", first_image_bytes)
        if not has_existing_prompt:
            zf.writestr("clipgt/prompt.txt", _PBSS_PLACEHOLDER_PROMPT)

    info(
        f"  Repacked {usdz_path.name}: added clipgt/first_image.{first_image_ext} "
        f"(from {chosen})"
        + ("" if has_existing_prompt else " and placeholder clipgt/prompt.txt")
    )


def _pbss_client():
    """Boto3 S3 client for PBSS Portland against the team-alpamayo account.

    ``boto3`` is imported lazily so the HF-only path doesn't pay for it
    and so users without the package only hit the error if they actually
    use ``--from-pbss-anon-v3``.
    """
    try:
        import boto3  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - boto3 must be installed
        raise RuntimeError(
            "boto3 is required for --from-pbss-anon-v3 but is not installed."
        ) from exc

    secret = os.environ.get(PBSS_SECRET_ENV_VAR)
    if not secret:
        raise RuntimeError(
            f"{PBSS_SECRET_ENV_VAR} is required to read from PBSS "
            f"({PBSS_ACCESS_KEY_ID}). Export it before re-running."
        )

    return boto3.client(
        "s3",
        endpoint_url=PBSS_ENDPOINT_URL,
        region_name=PBSS_REGION,
        aws_access_key_id=PBSS_ACCESS_KEY_ID,
        aws_secret_access_key=secret,
    )


def stage_scene_from_pbss(scene_uuid: str, *, force: bool, _client=None) -> Path:
    """Download one anonymized-v3 USDZ from Guillermo's PBSS bucket.

    Writes to the same on-disk location as :func:`stage_scene` so the
    rest of the demo (which reads from :func:`local_scene_archive_path`)
    finds it without code changes.

    These USDZs already contain the demo-required ``clipgt/*.parquet``
    layout, but bundle the first frame as
    ``frames/<camera>/<timestamp>.jpeg`` rather than
    ``clipgt/first_image.*`` and ship no ``clipgt/prompt.txt``. The HD-map
    rasterizer works as-is; the world-model side may need a follow-up
    repackaging step (move a first frame into ``clipgt/``, drop in a
    ``clipgt/prompt.txt``) before it can be served end-to-end.
    """
    bare_uuid = normalise_scene_uuid(scene_uuid)
    dest = scene_path(bare_uuid)

    s3 = _client if _client is not None else _pbss_client()
    key = f"{PBSS_PREFIX}/{bare_uuid}/{bare_uuid}.usdz"
    try:
        remote_size = s3.head_object(Bucket=PBSS_BUCKET, Key=key)["ContentLength"]
    except Exception as exc:
        raise RuntimeError(
            f"Scene {bare_uuid} not found on PBSS at "
            f"s3://{PBSS_BUCKET}/{key} ({exc})."
        ) from exc

    if dest.exists() and not force:
        local_size = dest.stat().st_size
        # Tolerance covers the repack step (which appends ~few MB of
        # first_image + prompt.txt to the original PBSS USDZ). Anything
        # bigger than that means the on-disk file is a different
        # archive entirely (e.g. the much smaller HF-sourced "small map"
        # version from a prior prepare.py run) and should be replaced.
        if abs(local_size - remote_size) <= 16 * 1024 * 1024:
            info(
                f"Scene already staged at {dest} "
                f"({human_bytes(local_size)}; PBSS {human_bytes(remote_size)})."
            )
            _repack_pbss_usdz(dest)
            return dest
        info(
            f"Re-downloading {dest.name}: local {human_bytes(local_size)} "
            f"differs from PBSS {human_bytes(remote_size)} "
            "(probably stale small-map HF version)."
        )

    info(
        f"Downloading PBSS scene s3://{PBSS_BUCKET}/{key} "
        f"({human_bytes(remote_size)}) -> {dest}"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling .partial then atomically rename so an interrupted
    # download never leaves a half-file at the canonical path (which the
    # idempotency check above treats as "already staged").
    tmp = dest.with_suffix(".usdz.partial")
    s3.download_file(PBSS_BUCKET, key, str(tmp))
    tmp.rename(dest)
    info(f"Staged scene at {dest} ({human_bytes(dest.stat().st_size)}).")
    _repack_pbss_usdz(dest)
    return dest


def _list_pbss_scene_uuids(client, *, limit: int | None = None) -> list[str]:
    """Enumerate UUIDs at ``s3://{PBSS_BUCKET}/{PBSS_PREFIX}/`` (sorted).

    Skips any path that isn't ``<prefix>/<uuid>/<uuid>.usdz``, so partial
    uploads / sibling MP4s don't sneak in. ``limit`` truncates the list
    after sorting so the slice is deterministic across runs.
    """
    uuids: list[str] = []
    prefix_with_slash = f"{PBSS_PREFIX}/"
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=PBSS_BUCKET, Prefix=prefix_with_slash
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".usdz"):
                continue
            tail = key[len(prefix_with_slash):]
            parts = tail.split("/")
            if len(parts) != 2 or parts[1] != f"{parts[0]}.usdz":
                continue
            uuids.append(parts[0])
    uuids.sort()
    if limit is not None:
        uuids = uuids[:limit]
    return uuids


def stage_scenes_from_pbss(
    *, force: bool, uuids: tuple[str, ...] | list[str] | None = None
) -> None:
    """Stage a set of PBSS-anon-v3 scenes concurrently.

    ``uuids`` defaults to :data:`PBSS_SCENE_UUIDS` (the 4 hardcoded demo
    scenes). Pass an explicit list -- e.g. from
    :func:`_list_pbss_scene_uuids` -- to stage a larger batch.
    """
    client = _pbss_client()  # also smoke-tests creds before fan-out
    scene_uuids = list(uuids) if uuids is not None else list(PBSS_SCENE_UUIDS)
    # Average ~1.85 GiB/scene observed in the v3 prefix; close enough for
    # a heads-up totals print so the user can ^C before committing to
    # tens of GiB of GETs.
    approx_gib = len(scene_uuids) * 1.85
    info(
        f"Staging {len(scene_uuids)} scene(s) from "
        f"s3://{PBSS_BUCKET}/{PBSS_PREFIX}/ (~{approx_gib:.0f} GiB total)."
    )
    # 4 concurrent GETs is the sweet spot: enough to saturate a 10 GbE
    # link but small enough not to hammer the PBSS frontend.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                stage_scene_from_pbss, uuid, force=force, _client=client
            ): uuid
            for uuid in scene_uuids
        }
        failures: list[tuple[str, BaseException]] = []
        for fut in concurrent.futures.as_completed(futures):
            uuid = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                failures.append((uuid, exc))
                info(f"  FAILED {uuid}: {exc}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(scene_uuids)} PBSS scenes failed to "
            f"stage; see preceding [prepare] FAILED lines."
        )


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
    elif args.from_pbss_anon_v3:
        # PBSS branch is auth'd by ALPAMAYO_S3_SECRET, not HF_TOKEN, so it
        # short-circuits the HF gating below.
        if args.scene_uuid is not None:
            stage_scene_from_pbss(args.scene_uuid, force=args.force)
        elif args.pbss_count is not None:
            client = _pbss_client()
            discovered = _list_pbss_scene_uuids(client, limit=args.pbss_count)
            info(
                f"Discovered {len(discovered)} UUID(s) under "
                f"s3://{PBSS_BUCKET}/{PBSS_PREFIX}/ "
                f"(limited to first {args.pbss_count})."
            )
            stage_scenes_from_pbss(force=args.force, uuids=discovered)
        else:
            stage_scenes_from_pbss(force=args.force)
    elif not os.environ.get("HF_TOKEN"):
        info(
            "HF_TOKEN is not set; skipping scene download. Export HF_TOKEN "
            "and rerun, or pass --skip-scene and provide your own USDZ via "
            "the --scene flag to interactive_drive."
        )
    elif args.scene_uuid is not None:
        stage_scene(args.scene_uuid, force=args.force)
    else:
        uuids = list_available_scene_uuids()
        info(f"Staging all {len(uuids)} scene(s) from {hf_scenes_repo_id()}.")
        for i, uuid in enumerate(uuids, start=1):
            info(f"  [{i}/{len(uuids)}] {uuid}")
            stage_scene(uuid, force=args.force)

    info("Workspace assets are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
