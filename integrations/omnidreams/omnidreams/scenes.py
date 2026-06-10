# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared metadata + helpers for the ``omni-dreams-scenes`` HF dataset.

Keeps the desktop ``interactive_drive`` demo (which uses the USDZ archive
intact) and ``webrtc.session`` (which extracts it) in lock-step on scene
naming, the HF org resolver, the variant-suffix parser, and the shared
on-disk cache layout under :data:`FLASHDREAMS_CACHE_DIR`/``omnidreams-scenes/``.
The archive (``<root>/clipgt-<uuid>.usdz``) and extracted dir
(``<root>/<uuid>/``) coexist without name conflict.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

from omnidreams.hf_org import hf_repo

# First-frame image suffixes; both demo paths lowercase before comparison.
SCENE_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
)

# Per-scene prompt filename. interactive-drive also supports ``prompt_<N>.txt``
# variants (via ``variant_from_stem``); webrtc uses only this canonical name.
SCENE_PROMPT_FILENAME: Final[str] = "prompt.txt"

# Subdir webrtc unpacks a USDZ payload into (``<scenes_cache_root>/<uuid>/clipgt/``).
SCENE_CLIPGT_DIRNAME: Final[str] = "clipgt"

# Per-camera ground-truth frames live at ``frames/<camera>/<ts_us>.jpeg``;
# scenes seed generation from the first frame instead of ``first_image.png``.
SCENE_FRAMES_DIRNAME: Final[str] = "frames"
SCENE_FRAME_SUFFIXES: Final[frozenset[str]] = frozenset({".jpeg", ".jpg", ".png"})

# Slug for the base (no-suffix) scene archive.
SCENE_VARIANT_DEFAULT: Final[str] = "default"

# Weather variant -> 1-based prompt index inside the archive
# (prompt1=clear, prompt2=snow, prompt3=rain). Unknown variants -> prompt 1.
SCENE_VARIANT_PROMPT_INDEX: Final[dict[str, int]] = {
    SCENE_VARIANT_DEFAULT: 1,
    "snow": 2,
    "rain": 3,
}

# Parses ``clipgt-<uuid>[-<variant>]``. Anchored on the canonical UUID shape
# so the variant split doesn't bite into the UUID's hyphens; prefix optional.
_CLIPGT_STEM_RE: Final = re.compile(
    r"^(?:clipgt-)?"
    r"(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"(?:-(?P<variant>.+))?$"
)

# Canonical NVIDIA dataset browser URL; intentionally fixed at ``nvidia/``
# (public docs always point there) even when OMNI_DREAMS_HF_ORG overrides the repo.
HF_DATASET_BROWSER_URL: Final[str] = (
    "https://huggingface.co/datasets/nvidia/omni-dreams-scenes/tree/main/scenes"
)


def hf_scenes_repo_id(org: str | None = None) -> str:
    """Return ``<resolved-org>/omni-dreams-scenes`` for HF lookups.

    Delegates to :func:`omnidreams.hf_org.hf_repo` so ``OMNI_DREAMS_HF_ORG``
    / ``--hf-org`` flow through here too.
    """
    return hf_repo(kind="scenes", org=org)


def parse_scene_stem(stem: str) -> tuple[str, str]:
    """Split a ``clipgt-<uuid>[-<variant>]`` stem into ``(bare_uuid, variant)``.

    Variant defaults to :data:`SCENE_VARIANT_DEFAULT` when there's no suffix.
    Non-UUID inputs just get the ``clipgt-`` prefix stripped (so synthetic /
    non-clipgt names still yield a sane bare id).
    """
    match = _CLIPGT_STEM_RE.match(stem.strip())
    if match is not None:
        return match.group("uuid"), (match.group("variant") or SCENE_VARIANT_DEFAULT)
    return stem.strip().removeprefix("clipgt-"), SCENE_VARIANT_DEFAULT


def normalise_scene_uuid(scene_uuid: str) -> str:
    """Coerce a ``clipgt-<uuid>[-<variant>]`` stem or bare ``<uuid>`` to the bare UUID.

    Strips both the ``clipgt-`` prefix and any variant suffix; downstream HF /
    local path helpers all assume the bare form.
    """
    return parse_scene_stem(scene_uuid)[0]


def _variant_suffix(variant: str | None) -> str:
    """Filename suffix for ``variant`` (``""`` for the default/base archive)."""
    slug = (variant or SCENE_VARIANT_DEFAULT).strip()
    return "" if slug in ("", SCENE_VARIANT_DEFAULT) else f"-{slug}"


def scene_archive_filename(
    scene_uuid: str, variant: str = SCENE_VARIANT_DEFAULT
) -> str:
    """HF-dataset path for one scene variant's USDZ archive.

    ``variant`` selects a weather sibling (``-rain`` / ``-snow``); the default
    maps to the base ``scenes/clipgt-<uuid>.usdz``.
    """
    return f"scenes/clipgt-{normalise_scene_uuid(scene_uuid)}{_variant_suffix(variant)}.usdz"


def prompt_variant_for_scene_variant(variant: str) -> str:
    """Map a scene variant slug to the in-archive prompt key (``"1"``/``"2"``/``"3"``).

    Weather variants map via :data:`SCENE_VARIANT_PROMPT_INDEX` so the seed
    prompt matches the imagery; a numeric variant is returned as-is (legacy
    in-archive selection), and unknown slugs fall back to ``"1"``.
    """
    slug = (variant or SCENE_VARIANT_DEFAULT).strip()
    if slug.isdecimal():
        return slug
    return str(SCENE_VARIANT_PROMPT_INDEX.get(slug, 1))


def resolve_variant_archive(scene_path: Path, variant: str) -> Path:
    """Return the sibling USDZ for ``variant`` next to ``scene_path``.

    Returns the matching ``clipgt-<uuid>[-<variant>].usdz`` sibling when it
    exists on disk, else ``scene_path`` unchanged (legacy single-archive
    scenes have no sibling; the loader picks the variant from within).
    """
    scene_path = Path(scene_path)
    uuid, _current = parse_scene_stem(scene_path.stem)
    candidate = scene_path.with_name(f"clipgt-{uuid}{_variant_suffix(variant)}.usdz")
    if candidate != scene_path and candidate.exists():
        return candidate
    return scene_path


# Root of every flashdreams-managed cache dir (override via
# ``FLASHDREAMS_CACHE_DIR``). Module-level constant read on every call so tests
# can monkeypatch it and a late re-assignment still takes effect.
FLASHDREAMS_CACHE_DIR: Path = Path(
    os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams"))
)


def scenes_cache_root() -> Path:
    """Shared cache root for staged scenes: ``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes``."""
    return FLASHDREAMS_CACHE_DIR / "omnidreams-scenes"


def local_scene_archive_path(
    scene_uuid: str, variant: str = SCENE_VARIANT_DEFAULT
) -> Path:
    """Staged archive path ``<scenes_cache_root>/clipgt-<uuid>[-<variant>].usdz``.

    Mirrors the HF dataset's filenames so the cache dir matches Hugging Face.
    """
    return (
        scenes_cache_root()
        / f"clipgt-{normalise_scene_uuid(scene_uuid)}{_variant_suffix(variant)}.usdz"
    )


def variant_from_stem(stem: str, prefix: str) -> str | None:
    """Map a file *stem* to its variant slug (``--variant`` / HUD selector).

    * ``<prefix>``      -> ``"default"`` (e.g. ``prompt.txt``)
    * ``<prefix>_<X>``  -> ``<X>``       (e.g. ``prompt_1.txt`` -> ``"1"``)
    * ``<prefix><N>``   -> ``<N>``       (numeric only, e.g. ``prompt1.txt`` -> ``"1"``)
    * anything else     -> ``None``      (rejected; caller skips it)
    """
    if stem == prefix:
        return "default"
    if stem.startswith(prefix + "_"):
        return stem[len(prefix) + 1 :]
    if stem.startswith(prefix):
        suffix = stem[len(prefix) :]
        if suffix.isdecimal():
            return suffix
    return None


def _list_repo_scene_files() -> list[str]:
    """Return every ``scenes/clipgt-*.usdz`` repo path in the HF dataset."""
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
    return [
        path for path in files if path.startswith(path_prefix) and path.endswith(suffix)
    ]


def list_available_scene_files() -> list[tuple[str, str]]:
    """Enumerate every scene archive in the HF dataset as ``(uuid, variant)``.

    Sorted so each scene's base archive comes first. Requires ``HF_TOKEN``
    (gated dataset); honours ``OMNI_DREAMS_HF_ORG`` / ``--hf-org``.
    """
    pairs = {parse_scene_stem(Path(path).stem) for path in _list_repo_scene_files()}
    return sorted(
        pairs,
        key=lambda pair: (pair[0], "" if pair[1] == SCENE_VARIANT_DEFAULT else pair[1]),
    )


def list_available_scene_uuids() -> list[str]:
    """Sorted unique bare scene UUIDs in the HF dataset (one per scene).

    Use :func:`list_available_scene_files` for the per-variant breakdown.
    """
    return sorted({uuid for uuid, _variant in list_available_scene_files()})


def hf_hub_download_scene(
    scene_uuid: str, variant: str = SCENE_VARIANT_DEFAULT
) -> Path:
    """Download one scene variant's USDZ from the HF dataset into the HF cache.

    Returns the cached local path; repeat calls for the same UUID + variant
    are cache hits. ``variant`` selects a weather sibling (``rain`` / ``snow``).
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
        filename=scene_archive_filename(scene_uuid, variant),
    )
    return Path(cached)
