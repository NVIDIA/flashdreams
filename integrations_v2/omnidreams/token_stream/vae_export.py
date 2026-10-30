# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config-driven export of the launched VAE decoder for token streaming.

The server generates latents with the decoder named in its pipeline config, and
the browser must decode them with the *same* decoder. This module exports that
exact decoder (not a hardwired one) and stamps its identity into the spec, so a
mismatch is detectable rather than silently producing wrong frames. Shared by
the ``export_vae`` CLI and the server's auto-export at launch.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from omnidreams.token_stream import vae_artifacts

from flashdreams.runtime_v2.serving.token_stream.ort_runtime import ensure_ort_runtime

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}

#: Per-precision actions decided by :func:`plan_vae_exports`.
EXPORT = "export"
REUSE = "reuse"


class TokenStreamExportError(RuntimeError):
    """Token streaming could not be provisioned at launch.

    Raised when a matching decoder artifact is missing and cannot be produced --
    auto-export is disabled, or the export / post-export validation failed. The
    server treats this (and
    :class:`~omnidreams.token_stream.vae_artifacts.TokenStreamUnsupportedDecoder`)
    as a fatal launch error when ``--enable-token-stream`` was requested.
    """


def resolve_pipeline_config(name: str) -> Any:
    """Look up a shipped OmniDreams pipeline config by ``name``.

    Raises ``ValueError`` listing the available names (instead of a bare
    ``KeyError``) so the CLI/server fail with guidance.
    """
    from omnidreams.config import OMNIDREAMS_CONFIGS

    try:
        return OMNIDREAMS_CONFIGS[name]
    except KeyError:
        available = ", ".join(sorted(OMNIDREAMS_CONFIGS))
        raise ValueError(
            f"unknown pipeline config {name!r}; available: {available}"
        ) from None


def export_decoder_from_config(
    pipeline_config: Any,
    *,
    video_width: int,
    video_height: int,
    precision: str,
    device: str = "cuda:0",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Export the pipeline's VAE decoder to a cache-as-IO ONNX + spec sidecar.

    The decoder, its weights, and ``len_t`` come from ``pipeline_config``, so the
    artifact matches what the server generates with. Raises
    :class:`~omnidreams.token_stream.vae_artifacts.TokenStreamUnsupportedDecoder`
    for a non-TAEHV decoder *before* any weights are loaded.
    """
    from flashdreams.recipes.taehv.export import export_streaming_decoder

    decoder_cfg = pipeline_config.decoder
    len_t = int(pipeline_config.diffusion_model.transformer.len_t)

    # Family check first: a non-TAEHV decoder has no cache-IO export, so bail out
    # here rather than after paying to instantiate the decoder.
    identity = vae_artifacts.identity_for_config(
        decoder_cfg, len_t, video_width, video_height
    )

    dtype = _DTYPES[precision]
    # torch.onnx needs a traceable eager module: drop the perf config's compile /
    # CUDA-graph wrappers (used for generation, not traceable for export).
    eager_cfg = dataclasses.replace(
        decoder_cfg, use_compile=False, use_cuda_graph=False
    )
    decoder = eager_cfg.setup().to(device, dtype).eval()

    # Shapes are consequences of the checkpoint, read off the instantiated module.
    spatial = int(getattr(decoder, "spatial_compression_ratio", 8))
    latent_channels = int(getattr(decoder.taehv, "latent_channels", 16))

    out_dir = Path(out_dir) if out_dir is not None else vae_artifacts.cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / vae_artifacts.onnx_path(precision).name

    spec = export_streaming_decoder(
        decoder,
        out_path=out_path,
        latent_frames=len_t,
        latent_channels=latent_channels,
        latent_height=video_height // spatial,
        latent_width=video_width // spatial,
        device=device,
        dtype=dtype,
        version=vae_artifacts.identity_hash(identity),
    )
    # export_streaming_decoder knows only the shape spec; add the launched-decoder
    # identity so the server handshake (and client) can verify the match, and the
    # cold-start convention so the client does not hardcode it. The TAEHV cache's
    # native cold start is the all-zeros state.
    spec["identity"] = identity
    spec["cold_start"] = "zeros"
    out_path.with_suffix(".spec.json").write_text(json.dumps(spec, indent=2))
    return spec


def plan_vae_exports(
    launched_identity: dict[str, Any],
    precisions: Sequence[str],
    cache_state: Mapping[str, dict[str, Any] | None],
) -> dict[str, str]:
    """Per precision: ``REUSE`` a cached artifact whose identity matches the
    launched decoder, else ``EXPORT``. Pure (no GPU / no I/O) so the whole
    decision matrix is unit-testable; ``cache_state`` maps precision ->
    cached identity (or ``None`` if absent).
    """
    return {
        p: REUSE if cache_state.get(p) == launched_identity else EXPORT
        for p in precisions
    }


def _validate_exported_onnx(precision: str) -> None:
    """Structurally check a freshly exported ONNX; raise and delete it on failure.

    Uses ``onnx.checker`` only: ``onnxruntime`` is not a server dependency, and
    the real runtime (the browser's onnxruntime-web / WebGPU EP) can only be
    exercised client-side, so this catches a corrupt / malformed graph, not
    WebGPU-EP op support.
    """
    import onnx  # ty:ignore[unresolved-import]

    path = vae_artifacts.onnx_path(precision)
    try:
        onnx.checker.check_model(str(path))
    except Exception as exc:
        path.unlink(missing_ok=True)  # don't leave a bad artifact to be advertised
        raise TokenStreamExportError(
            f"exported ONNX for {precision} failed validation: {exc}"
        ) from exc


def ensure_vae_artifacts(
    pipeline_config: Any,
    *,
    video_width: int,
    video_height: int,
    precisions: Sequence[str],
    device: str = "cuda:0",
    auto_export: bool = True,
) -> None:
    """Ensure a decoder artifact matching the launched model exists for each
    precision, exporting the missing/stale ones. Called at server launch.

    Raises
    :class:`~omnidreams.token_stream.vae_artifacts.TokenStreamUnsupportedDecoder`
    for a non-TAEHV decoder, or :class:`TokenStreamExportError` when a matching
    artifact is missing and cannot be produced (auto-export off, or export /
    validation failed). The caller aborts the launch on either.
    """
    launched = vae_artifacts.identity_for_config(
        pipeline_config.decoder,
        int(pipeline_config.diffusion_model.transformer.len_t),
        video_width,
        video_height,
    )
    cache_state = {p: vae_artifacts.artifact_identity(p) for p in precisions}
    plan = plan_vae_exports(launched, precisions, cache_state)

    reused = sorted(p for p, action in plan.items() if action == REUSE)
    exported = sorted(p for p, action in plan.items() if action == EXPORT)
    for precision in exported:
        if not auto_export:
            raise TokenStreamExportError(
                f"no VAE decoder artifact matching the launched model for "
                f"{precision} and auto-export is disabled; run "
                f"`python -m omnidreams.token_stream.export_vae "
                f"--precision {precision}`."
            )
        export_decoder_from_config(
            pipeline_config,
            video_width=video_width,
            video_height=video_height,
            precision=precision,
            device=device,
        )
        _validate_exported_onnx(precision)

    logger.info(
        "Token streaming ready: decoder version={} reused={} exported={}.",
        vae_artifacts.identity_hash(launched),
        reused or "[]",
        exported or "[]",
    )


def prepare_token_stream(
    pipeline_config: Any,
    *,
    video_width: int,
    video_height: int,
    device: str,
    precisions: Sequence[str] = ("fp32",),
) -> dict[str, Any]:
    """Provision the client-side VAE decoder and return the session metadata.

    This is the ``token_stream_preparer`` hook an application runs in ``init``:
    it exports (or reuses) the launched decoder's ONNX for ``precisions``, then
    returns the ``SessionDesc.metadata`` entries the server advertises (the
    ``vae_model`` descriptor) and serves the artifacts from
    (``token_stream_assets``). Raises
    :class:`~omnidreams.token_stream.vae_artifacts.TokenStreamUnsupportedDecoder`
    or :class:`TokenStreamExportError` so an un-provisionable token stream aborts
    the launch rather than advertising a decoder the browser cannot run.

    ``precisions`` defaults to fp32 only; fp16 stays opt-in until its GPU export
    path is validated.
    """
    ensure_vae_artifacts(
        pipeline_config,
        video_width=video_width,
        video_height=video_height,
        precisions=precisions,
        device=device,
        auto_export=True,
    )
    launched = vae_artifacts.identity_for_config(
        pipeline_config.decoder,
        int(pipeline_config.diffusion_model.transformer.len_t),
        video_width,
        video_height,
    )
    # Advertise only the requested precisions whose on-disk artifact matches the
    # launched decoder -- never a stale artifact left at a different identity
    # (the v1 exporter wrote the same cache layout, so a leftover fp16 could be
    # present at a different identity).
    matched = [p for p in precisions if vae_artifacts.artifact_identity(p) == launched]
    if not matched:
        raise TokenStreamExportError(
            f"no provisioned token-stream decoder matches the launched model {launched}"
        )
    descriptor = vae_artifacts.build_descriptor(matched)
    # The browser decodes with onnxruntime-web; prefetch that runtime and serve
    # it same-origin so the page carries no CDN dependency (locked decision C).
    ort_descriptor, ort_routes = ensure_ort_runtime()
    return {
        "token_stream_extra_header": {
            "vae_model": descriptor,
            "ort_runtime": ort_descriptor,
        },
        "token_stream_assets": {
            **vae_artifacts.build_asset_routes(matched),
            **ort_routes,
        },
    }
