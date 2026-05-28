# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared metadata + helpers for the ``omni-dreams-scenes`` Hugging Face dataset.

Both demo paths in this package consume the same source of scene data
(the ``omni-dreams-scenes`` HF dataset at ``scenes/clipgt-<uuid>.usdz``)
and now also share an on-disk cache layout under
:data:`FLASHDREAMS_CACHE_DIR`/``omnidreams-scenes/``. They still differ
in *what they do* with the cached archive:

* ``omnidreams.interactive_drive`` (desktop demo) keeps the USDZ
  archive intact (its scene loader reads prompts / first-images out of
  the zip via ``zipfile.ZipFile``). ``omnidreams-prepare`` / the
  demo's first-launch auto-stage copies the HF-cached archive to
  ``<scenes_cache_root>/clipgt-<uuid>.usdz``.
* ``omnidreams.webrtc.session`` extracts the USDZ payload into
  ``<scenes_cache_root>/<uuid>/clipgt/`` and reads
  ``clipgt/first_image.*`` + ``clipgt/prompt.txt`` directly from disk.

The two layouts coexist in the same root: an archive lives at
``<root>/clipgt-<uuid>.usdz`` (a file) while the extracted directory
lives at ``<root>/<uuid>/`` (a directory). No name conflict, and
neither side downloads twice because both call
:func:`hf_hub_download_scene` which goes through ``huggingface_hub``'s
content-addressed HF cache.

This module owns:

* the dataset name and HF org resolver,
* the archive filename template and bare-UUID convention,
* the on-disk cache root (env-overridable via ``FLASHDREAMS_CACHE_DIR``),
* the file-suffix conventions and the variant-suffix parser used by
  the interactive-drive scene loader,
* a ``list_available_scene_uuids`` helper that walks the HF dataset.

Centralising these keeps the two demos in lock-step on what a "clipgt
scene" looks like, where to fetch one from, and where one ends up on
disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from omnidreams.hf_org import hf_repo

# ---------------------------------------------------------------------------
# Hugging Face dataset metadata
# ---------------------------------------------------------------------------

# Filename suffixes accepted as the scene's first-frame image. Both demo
# paths normalise to lowercase before comparison.
SCENE_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
)

# Canonical filename for the per-scene prompt inside an extracted clipgt
# bundle. The interactive-drive demo also supports ``prompt_<N>.txt`` for
# multiple variants of the same scene (parsed via ``variant_from_stem``
# below); the webrtc session pipeline only uses the canonical name.
SCENE_PROMPT_FILENAME: Final[str] = "prompt.txt"

# Conventional subdirectory under which a USDZ archive's payload is
# unpacked by the webrtc session pipeline (``<scenes_cache_root>/<uuid>/
# clipgt/``).
SCENE_CLIPGT_DIRNAME: Final[str] = "clipgt"

# Convenience link to the canonical NVIDIA-hosted dataset browser. The
# resolver below honours OMNI_DREAMS_HF_ORG when picking the actual repo
# id; this URL is intentionally fixed at ``nvidia/`` because the public
# docs always point there.
HF_DATASET_BROWSER_URL: Final[str] = (
    "https://huggingface.co/datasets/nvidia/omni-dreams-scenes/tree/main/scenes"
)


def hf_scenes_repo_id(org: str | None = None) -> str:
    """Return ``<resolved-org>/omni-dreams-scenes`` for HF lookups.

    Delegates to :func:`omnidreams.hf_org.hf_repo` so the
    ``OMNI_DREAMS_HF_ORG`` env var / ``--hf-org`` CLI flag flow through
    here too, keeping the webrtc server in lock-step with
    interactive-drive once the env var is set.
    """
    return hf_repo(kind="scenes", org=org)


def normalise_scene_uuid(scene_uuid: str) -> str:
    """Coerce either ``clipgt-<uuid>`` stems or bare ``<uuid>`` to the bare form.

    The omni-dreams-scenes dataset stores files as
    ``scenes/clipgt-<uuid>.usdz``, and the demo's default ``--scene`` path
    uses the same ``clipgt-<uuid>.usdz`` filename locally. Users (and
    internal callers using ``Path.stem`` on a local file) sometimes pass
    the ``clipgt-`` prefix in; others (the webrtc server, the
    ``--scene-uuid`` flag) pass the bare UUID. The downstream HF + local
    path helpers below all assume the **bare** UUID form, so this
    function normalises at the boundary.
    """
    return scene_uuid.strip().removeprefix("clipgt-")


def scene_archive_filename(scene_uuid: str) -> str:
    """HF-dataset path for one scene's USDZ archive.

    Accepts either a bare UUID or a ``clipgt-<uuid>`` stem (see
    :func:`normalise_scene_uuid`).
    """
    return f"scenes/clipgt-{normalise_scene_uuid(scene_uuid)}.usdz"


# ---------------------------------------------------------------------------
# On-disk cache layout
# ---------------------------------------------------------------------------

# Root of every flashdreams-managed cache directory. Honours an opt-in
# ``FLASHDREAMS_CACHE_DIR`` env var so users who keep dot-caches on a
# separate volume can redirect everything in one place. Defined as a
# module-level constant rather than a function so monkeypatching it in
# tests is straightforward; helpers below read it on every call so a
# late re-assignment still takes effect.
FLASHDREAMS_CACHE_DIR: Path = Path(
    os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams"))
)


def scenes_cache_root() -> Path:
    """Shared cache root for staged scenes (both archive and extracted forms).

    Resolves to ``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes``. Read on
    every call so tests that monkeypatch :data:`FLASHDREAMS_CACHE_DIR`
    immediately see the redirect.
    """
    return FLASHDREAMS_CACHE_DIR / "omnidreams-scenes"


def local_scene_archive_path(scene_uuid: str) -> Path:
    """Where the desktop demo expects a staged scene archive to live.

    ``<scenes_cache_root>/clipgt-<uuid>.usdz``. Matches the
    ``clipgt-<uuid>.usdz`` naming the HF dataset uses so a user staring
    at the cache dir sees the same filenames as on Hugging Face.
    """
    return scenes_cache_root() / f"clipgt-{normalise_scene_uuid(scene_uuid)}.usdz"


# ---------------------------------------------------------------------------
# Filename convention helpers
# ---------------------------------------------------------------------------


def variant_from_stem(stem: str, prefix: str) -> str | None:
    """Canonical scene-variant name parser.

    Maps a file *stem* (no extension) to the variant slug used by
    ``--variant`` / the HUD's variant selector. The convention, matching
    what ``nvidia/omni-dreams-scenes`` ships:

    * ``<prefix>``           -> ``"default"``  (e.g. ``prompt.txt``, ``first_image.png``)
    * ``<prefix>_<X>``       -> ``<X>``        (e.g. ``prompt_1.txt`` -> ``"1"``)
    * anything else          -> ``None``       (rejected; caller skips it)

    The trailing-suffix-without-underscore form (``prompt1.txt``,
    ``first_image1.png``) is **rejected** so the HUD's selector and the
    scene-loader's prompt dict agree on the variant key. Previously a
    naive ``stem.replace(prefix, "")`` quietly mapped ``prompt_1`` to
    ``_1`` while the HUD displayed ``1``, so the selector silently fell
    back to the default prompt on real scenes.

    Used by every discovery path that walks clipgt asset names:

    * ``omnidreams.interactive_drive.scene_loader._discover_prompts``
      and ``._discover_first_images`` (USDZ archive entries).
    * ``omnidreams.interactive_drive.demo._discover_variants``
      (HUD variant-selector dropdown).
    * ``omnidreams.interactive_drive.assets.scene_bundle._discover_prompts``
      and ``._discover_first_frames`` (unpacked scene directories).
    """
    if stem == prefix:
        return "default"
    if stem.startswith(prefix + "_"):
        return stem[len(prefix) + 1 :]
    return None


# ---------------------------------------------------------------------------
# Dataset enumeration
# ---------------------------------------------------------------------------


def list_available_scene_uuids() -> list[str]:
    """Enumerate every ``scenes/clipgt-<uuid>.usdz`` file in the HF dataset.

    Returns a sorted list of **bare** UUID strings (no ``clipgt-``
    prefix, no ``scenes/`` path, no ``.usdz`` suffix). The bare form
    matches what :func:`scene_archive_filename`,
    :func:`local_scene_archive_path`, and
    :func:`hf_hub_download_scene` expect as input.

    Requires ``HF_TOKEN`` to be set because the dataset is gated. The
    exact repo id is resolved via :func:`hf_scenes_repo_id`, so the
    function honours ``OMNI_DREAMS_HF_ORG`` / the ``--hf-org`` CLI flag.

    Imported lazily by callers (e.g. ``omnidreams-prepare``) so
    the ``huggingface_hub`` dependency only matters when this function
    is actually used.
    """
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - huggingface_hub must be installed
        raise RuntimeError(
            "Unable to import huggingface_hub.HfApi; run "
            "`uv sync --package flashdreams-omnidreams` from the flashdreams "
            "workspace root first."
        ) from exc

    repo_id = hf_scenes_repo_id()
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    path_prefix = "scenes/clipgt-"
    suffix = ".usdz"
    uuids = [
        path[len(path_prefix) : -len(suffix)]
        for path in files
        if path.startswith(path_prefix) and path.endswith(suffix)
    ]
    return sorted(uuids)


def hf_hub_download_scene(scene_uuid: str) -> Path:
    """Download one scene's USDZ archive from the resolved HF dataset.

    Returns the local path inside ``huggingface_hub``'s content-addressed
    cache (typically ``~/.cache/huggingface/hub/...``). Both demo paths
    call this; the second caller for the same UUID gets a cache HIT and
    no network traffic.

    Callers own what to do *after* download:

    * ``omnidreams.prepare.stage_scene`` copies the cached
      archive to :func:`local_scene_archive_path` so the demo's
      ``--scene`` path is a stable real file.
    * ``omnidreams.webrtc.session._ensure_hf_webrtc_scene_synced``
      extracts the archive under :func:`scenes_cache_root` /
      ``<uuid>/clipgt/`` for filesystem access.

    Accepts either a bare UUID or a ``clipgt-<uuid>`` stem.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Unable to import huggingface_hub; run "
            "`uv sync --package flashdreams-omnidreams` from the flashdreams "
            "workspace root first."
        ) from exc

    cached = hf_hub_download(
        repo_id=hf_scenes_repo_id(),
        repo_type="dataset",
        filename=scene_archive_filename(scene_uuid),
    )
    return Path(cached)
