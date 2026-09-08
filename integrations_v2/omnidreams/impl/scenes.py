# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared discovery and staging helpers for the ``omni-dreams-scenes`` dataset.

The desktop demo consumes USDZ archives intact, while realtime demos extract
them into a normalized ClipGT layout. Both paths share scene naming, variant
selection, Hugging Face lookup, and the cache rooted at
``FLASHDREAMS_CACHE_DIR/omnidreams-scenes``.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Set as AbstractSet
from pathlib import Path, PurePosixPath
from typing import Final

from filelock import FileLock
from loguru import logger
from omnidreams.impl.hf_org import hf_repo

# First-frame image suffixes; both demo paths lowercase before comparison.
SCENE_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
)

# Per-scene prompt filename. Interactive Drive also supports ``prompt_<N>.txt``
# variants through ``variant_from_stem``.
SCENE_PROMPT_FILENAME: Final[str] = "prompt.txt"

# Subdirectory used for extracted USDZ payloads.
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

    Delegates to :func:`omnidreams.impl.hf_org.hf_repo` so ``OMNI_DREAMS_HF_ORG``
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


def scene_variant_suffix(variant: str | None) -> str:
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
    return (
        f"scenes/clipgt-{normalise_scene_uuid(scene_uuid)}"
        f"{scene_variant_suffix(variant)}.usdz"
    )


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
    candidate = scene_path.with_name(
        f"clipgt-{uuid}{scene_variant_suffix(variant)}.usdz"
    )
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
        / f"clipgt-{normalise_scene_uuid(scene_uuid)}{scene_variant_suffix(variant)}.usdz"
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


def _choose_existing_asset(
    directory: Path,
    *,
    exact_name: str | None = None,
    fallback_stems: tuple[str, ...] = (),
    fallback_prefixes: tuple[str, ...] = (),
    allowed_suffixes: AbstractSet[str] | None = None,
    preferred_stems: tuple[str, ...] = (),
) -> Path | None:
    if not directory.is_dir():
        return None

    if exact_name is not None:
        exact_path = directory / exact_name
        if exact_path.is_file() and (
            allowed_suffixes is None or exact_path.suffix.lower() in allowed_suffixes
        ):
            return exact_path

    candidates = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
            continue
        if (
            path.stem in preferred_stems
            or path.stem in fallback_stems
            or any(path.stem.startswith(f"{prefix}-") for prefix in fallback_prefixes)
        ):
            candidates.append(path)

    if not candidates:
        return None

    preferred_order = {stem: index for index, stem in enumerate(preferred_stems)}
    return sorted(
        candidates,
        key=lambda path: (
            preferred_order.get(path.stem, len(preferred_order)),
            path.name,
        ),
    )[0]


def _camera_name_candidates(camera_name: str) -> tuple[str, ...]:
    underscore = camera_name.replace(":", "_")
    colon = camera_name.replace("_", ":")
    return tuple(dict.fromkeys((camera_name, underscore, colon)))


def _first_frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), path.name) if stem.isdigit() else (2**63 - 1, path.name)


def _resolve_first_frame(clipgt_dir: Path, camera_name: str) -> Path | None:
    frames_root = clipgt_dir / SCENE_FRAMES_DIRNAME
    if not frames_root.is_dir():
        return None
    candidate_dirs = [
        frames_root / name
        for name in _camera_name_candidates(camera_name)
        if (frames_root / name).is_dir()
    ]
    if not candidate_dirs:
        candidate_dirs = [
            path for path in sorted(frames_root.iterdir()) if path.is_dir()
        ]
    for directory in candidate_dirs:
        frames = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SCENE_FRAME_SUFFIXES
        ]
        if frames:
            return sorted(frames, key=_first_frame_sort_key)[0]
    return None


def resolve_scene_assets(
    scene_dir: Path,
    *,
    prompt_filename: str,
    clipgt_dirname: str,
    camera_name: str = "camera_front_wide_120fov",
    variant: str = SCENE_VARIANT_DEFAULT,
) -> tuple[Path, Path, Path]:
    """Resolve the ClipGT root, first frame, and prompt for a scene."""
    missing_assets = []
    clipgt_dir = scene_dir / clipgt_dirname
    if not clipgt_dir.is_dir():
        missing_assets.append(str(clipgt_dir))
        resolved_clipgt_dir = None
    else:
        resolved_clipgt_dir = clipgt_dir

    first_frame_path = (
        None
        if resolved_clipgt_dir is None
        else _resolve_first_frame(resolved_clipgt_dir, camera_name)
    )
    if first_frame_path is None and resolved_clipgt_dir is not None:
        first_frame_path = _choose_existing_asset(
            resolved_clipgt_dir,
            fallback_stems=("first_image_1",),
            allowed_suffixes=SCENE_IMAGE_SUFFIXES,
            preferred_stems=("first_image",),
        )
    if first_frame_path is None:
        missing_assets.append(
            f"frames/<camera>/*.jpeg or first_image.* under {resolved_clipgt_dir}/"
        )

    weather_prompt_stem = f"prompt{prompt_variant_for_scene_variant(variant)}"
    prompt_path = (
        None
        if resolved_clipgt_dir is None
        else _choose_existing_asset(
            resolved_clipgt_dir,
            fallback_stems=("prompt1", "prompt2", "prompt3", "prompt"),
            allowed_suffixes={".txt"},
            preferred_stems=(weather_prompt_stem, "prompt"),
        )
    )
    if prompt_path is None:
        missing_assets.append(f"{prompt_filename} under {resolved_clipgt_dir}/")

    if missing_assets:
        raise FileNotFoundError(
            "Missing Omnidreams scene assets: " + ", ".join(missing_assets)
        )

    assert resolved_clipgt_dir is not None
    assert first_frame_path is not None
    assert prompt_path is not None
    return resolved_clipgt_dir, first_frame_path, prompt_path


def _safe_extract_zip(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_file() or destination.is_symlink():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(source) as zf:
        for member in zf.infolist():
            member_path = PurePosixPath(member.filename)
            if (
                member_path.is_absolute()
                or not member_path.parts
                or any(part in {"", ".", ".."} for part in member_path.parts)
            ):
                raise ValueError(
                    f"Unsafe archive member in {source}: {member.filename}"
                )
            target = destination / Path(*member_path.parts)
            target_resolved = target.resolve()
            if destination_root != target_resolved and destination_root not in (
                target_resolved.parents
            ):
                raise ValueError(
                    f"Archive member escapes destination: {member.filename}"
                )
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def extract_local_scene(
    scene_dir: Path,
    *,
    scene_uuid: str | None,
    variant: str = SCENE_VARIANT_DEFAULT,
    clipgt_dirname: str,
) -> Path:
    """Extract a local scene archive into the normalized scene layout."""
    if scene_uuid is None:
        return scene_dir

    scene_uuid = scene_uuid.strip()
    assert scene_uuid, "scene_uuid must be non-empty when provided."
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene_dir does not exist: {scene_dir}")

    suffix = scene_variant_suffix(variant)
    expected_names = (
        f"clipgt-{scene_uuid}{suffix}.usdz",
        f"{scene_uuid}{suffix}.usdz",
    )
    archive_path = _choose_existing_asset(scene_dir, exact_name=expected_names[0]) or (
        _choose_existing_asset(scene_dir, exact_name=expected_names[1])
    )
    if archive_path is None:
        archive_path = _choose_existing_asset(
            scene_dir,
            fallback_prefixes=(
                f"clipgt-{scene_uuid}{suffix}",
                f"{scene_uuid}{suffix}",
                f"clipgt-{scene_uuid}",
                scene_uuid,
            ),
            allowed_suffixes={".usdz"},
            preferred_stems=(
                f"clipgt-{scene_uuid}{suffix}",
                f"{scene_uuid}{suffix}",
                f"clipgt-{scene_uuid}",
                scene_uuid,
            ),
        )
    if archive_path is None:
        raise FileNotFoundError(
            "scene_uuid is set but no local USDZ archive was found in "
            f"{scene_dir}. Expected one of: {', '.join(expected_names)}."
        )

    normalized_scene_dir = scene_dir / f"{scene_uuid}{suffix}"
    _safe_extract_zip(archive_path, normalized_scene_dir / clipgt_dirname)
    return normalized_scene_dir


def ensure_hf_scene_synced(
    scene_uuid: str,
    *,
    variant: str = SCENE_VARIANT_DEFAULT,
    clipgt_dirname: str = SCENE_CLIPGT_DIRNAME,
) -> Path:
    """Download and extract a Hugging Face scene into the shared cache."""
    scene_uuid = scene_uuid.strip()
    assert scene_uuid, "scene_uuid must be set."
    suffix = scene_variant_suffix(variant)
    cache_root = scenes_cache_root()
    scene_dir = cache_root / f"{scene_uuid}{suffix}"
    lock_path = cache_root / ".locks" / f"{scene_uuid}{suffix}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path)):
        archive_path = hf_hub_download_scene(scene_uuid, variant)
        _safe_extract_zip(archive_path, scene_dir / clipgt_dirname)

    logger.info(
        "Synced Omnidreams scene {} (variant {}) from Hugging Face ({}) to {}",
        scene_uuid,
        variant,
        hf_scenes_repo_id(),
        scene_dir,
    )
    return scene_dir


def _link_or_copy_file(source: Path, target: Path) -> None:
    try:
        os.symlink(source, target)
        return
    except OSError:
        pass

    try:
        os.link(source, target)
        return
    except OSError:
        shutil.copy2(source, target)


def prepare_clipgt_dir(
    clipgt_dir: Path,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Normalize supported ClipGT parquet layouts for the scene loader."""

    def has_prefixed_parquets(path: Path) -> bool:
        return any(path.glob("*.calibration_estimate.parquet"))

    def has_unprefixed_parquets(path: Path) -> bool:
        return (path / "calibration_estimate.parquet").exists()

    if has_prefixed_parquets(clipgt_dir):
        return clipgt_dir, None

    parquet_source_dir: Path | None = None
    if has_unprefixed_parquets(clipgt_dir):
        parquet_source_dir = clipgt_dir
    else:
        for candidate in (child for child in clipgt_dir.iterdir() if child.is_dir()):
            if has_prefixed_parquets(candidate):
                return candidate, None
            if has_unprefixed_parquets(candidate):
                parquet_source_dir = candidate
                break

    if parquet_source_dir is None:
        return clipgt_dir, None

    temp_dir = tempfile.TemporaryDirectory(prefix="omnidreams-clipgt-")
    staged = Path(temp_dir.name)
    for source in parquet_source_dir.glob("*.parquet"):
        target = staged / f"clip.{source.name}"
        _link_or_copy_file(source.resolve(), target)
    return staged, temp_dir
