# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml
from omnidreams.hf_org import (
    DEFAULT_HF_ORG,
    hf_access_hint,
    resolve_hf_org,
    rewrite_omni_dreams_urls,
)

_HF_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/[^/]+/[^/]+/(?:blob|resolve)/[^/]+/.+$",
    re.IGNORECASE,
)
_DEFAULT_RESOLUTION_WH = (1280, 704)
_RESOLUTION_ALIGNMENT_PX = 16


def _is_hf_url(raw: str) -> bool:
    """Return True for ``https://huggingface.co/<ns>/<repo>/blob|resolve/<rev>/<file>`` URLs."""
    return bool(_HF_URL_PATTERN.match(raw))


def _parse_hf_url(url: str) -> tuple[str, str, str | None, str]:
    """Parse an HF file URL into ``(repo_id, filename, subfolder, revision)``.

    Callers are expected to have validated the URL with ``_is_hf_url`` first;
    the only checks here are on the URL's path structure.
    """
    parsed = urlparse(url)
    parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 5 or parts[2] not in {"blob", "resolve"}:
        raise ValueError(
            f"Invalid Hugging Face file URL: {url}. "
            "Expected /<namespace>/<repo>/blob|resolve/<revision>/<path/to/file>."
        )
    namespace, repo, _route, revision, *rest = parts
    filename = rest[-1]
    subfolder = "/".join(rest[:-1]) or None
    return f"{namespace}/{repo}", filename, subfolder, revision


def download_hf_file(url: str) -> Path:
    """Resolve an HF file URL to a local cached path via ``hf_hub_download``.

    Also used by ``prepare.py`` to pre-warm the Hugging Face cache so that
    the first demo run does not block on network downloads.
    """
    # ``huggingface_hub`` is declared under the ``world-model`` extra. Import
    # lazily so manifests that only reference local paths stay import-safe.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    repo_id, filename, subfolder, revision = _parse_hf_url(url)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
        )
    except RepositoryNotFoundError as exc:
        # 401 / 403 / 404 from HF all surface as RepositoryNotFoundError and
        # almost always come from one of two misconfigurations: HF_TOKEN
        # missing/misnamed, or OMNI_DREAMS_HF_ORG left at the default when
        # the caller is entitled to a different org. Replace the stock HF
        # message with a diagnostic that names both knobs explicitly.
        raise RuntimeError(hf_access_hint(repo_id, url)) from exc
    return Path(local_path)


def _resolve_manifest_path(raw_path: str | None, *, manifest_dir: Path) -> Path | None:
    """Resolve a manifest path entry.

    Accepts:
      - ``None`` / empty → returns ``None``
      - An absolute or manifest-relative local filesystem path
      - A Hugging Face file URL (``https://huggingface.co/.../resolve/<rev>/<file>``),
        which is materialised into the local HF cache and resolved to its
        on-disk path
    """
    if not raw_path:
        return None
    if _is_hf_url(raw_path):
        return download_hf_file(raw_path)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    return path


def _parse_resolution_wh(raw: object) -> tuple[int, int]:
    if raw is None:
        return _DEFAULT_RESOLUTION_WH
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"resolution_wh must be [width, height], got {raw!r}")
    width, height = int(raw[0]), int(raw[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution_wh must be positive, got {(width, height)!r}")
    if width % _RESOLUTION_ALIGNMENT_PX or height % _RESOLUTION_ALIGNMENT_PX:
        raise ValueError(
            "resolution_wh must be divisible by "
            f"{_RESOLUTION_ALIGNMENT_PX}, got {(width, height)!r}"
        )
    return (width, height)


@dataclass(frozen=True)
class WorldModelManifest:
    debug_condition_frame_dir: Path | None = None
    resolution_wh: tuple[int, int] = _DEFAULT_RESOLUTION_WH
    fps: int = 30
    num_frames_per_block: int = 8
    compile_net: bool = True
    light_vae: bool = True
    encode_with_pixel_shuffle: bool = False
    local_attn_size: int = 6
    skip_finalize_kv_cache: bool = False
    sink_size: int = 0
    denoising_steps: list[int] = field(default_factory=lambda: [1000, 500])
    upsampling_enabled: bool = False
    upsampling_scale: int = 4
    device: str = "cuda:0"
    seed_for_every_rollout: int | None = None


def load_world_model_manifest(path: str | Path) -> WorldModelManifest:
    manifest_path = Path(path)
    manifest_dir = manifest_path.resolve().parent
    raw_yaml = manifest_path.read_text(encoding="utf-8")
    # Some deployments set ``OMNI_DREAMS_HF_ORG`` (or the equivalent
    # ``--hf-org`` CLI flag, which is stamped into the env var early in
    # ``main()``). The canonical example yaml ships with NVIDIA URLs; rewrite
    # ``nvidia/omni-dreams-scenes`` to the resolved org here so callers don't
    # have to maintain a parallel yaml file. Other HF URLs in the manifest
    # (lightx2v Autoencoders, Cosmos-Reason1, ...) are not OmniDreams scene
    # repos and pass through unchanged.
    resolved_org = resolve_hf_org()
    if resolved_org != DEFAULT_HF_ORG:
        rewritten = rewrite_omni_dreams_urls(raw_yaml, org=resolved_org)
        if rewritten != raw_yaml:
            print(
                f"[manifest] rewrote {DEFAULT_HF_ORG}/omni-dreams-* URLs to "
                f"{resolved_org}/omni-dreams-* per OMNI_DREAMS_HF_ORG",
                flush=True,
            )
        raw_yaml = rewritten
    data = yaml.safe_load(raw_yaml) or {}
    resolution = _parse_resolution_wh(data.get("resolution_wh"))
    return WorldModelManifest(
        debug_condition_frame_dir=_resolve_manifest_path(
            data.get("debug_condition_frame_dir"),
            manifest_dir=manifest_dir,
        ),
        resolution_wh=resolution,
        fps=int(data.get("fps", 30)),
        num_frames_per_block=int(data.get("num_frames_per_block", 8)),
        compile_net=bool(data.get("compile_net", True)),
        light_vae=bool(data.get("light_vae", True)),
        encode_with_pixel_shuffle=bool(data.get("encode_with_pixel_shuffle", False)),
        local_attn_size=int(data.get("local_attn_size", 6)),
        skip_finalize_kv_cache=bool(data.get("skip_finalize_kv_cache", False)),
        sink_size=int(data.get("sink_size", 0)),
        denoising_steps=[int(x) for x in data.get("denoising_steps", [1000, 500])],
        upsampling_enabled=bool(data.get("upsampling_enabled", False)),
        upsampling_scale=int(data.get("upsampling_scale", 4)),
        device=str(data.get("device", "cuda:0")),
        seed_for_every_rollout=(
            int(data["seed_for_every_rollout"])
            if data.get("seed_for_every_rollout") is not None
            else None
        ),
    )
