# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline synthetic assets for interactive-drive world-model latency runs."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.taehv.impl import TAEHV
from flashdreams.recipes.wan.autoencoder import vae as wan_vae_module
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)

_DEFAULT_SYNTHETIC_SEED = 20260611
_CACHE_SUBDIR = "synthetic_world_model"


@dataclass(frozen=True)
class SyntheticWorldModelAssets:
    """Local checkpoint-like files used by a synthetic world-model config."""

    encoder_checkpoint_path: Path | None
    decoder_checkpoint_path: Path
    native_vae_fp8_state_path: Path | None = None


def default_synthetic_asset_dir(*, config_name: str) -> Path:
    """Return the cache directory for synthetic assets for a config slug."""

    cache_root = Path(
        os.environ.get("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")
    ).expanduser()
    return cache_root / _CACHE_SUBDIR / config_name


def build_synthetic_world_model_assets(
    out_dir: Path,
    *,
    encoder_cfg: Any,
    decoder_cfg: Any,
    native_vae_fp8: bool = False,
    pixel_height: int | None = None,
    pixel_width: int | None = None,
    device: torch.device | str = "cpu",
    seed: int = _DEFAULT_SYNTHETIC_SEED,
) -> SyntheticWorldModelAssets:
    """Materialize complete offline weights matching the selected recipe.

    The generated files are local single-file checkpoints consumed by the
    normal production configs. They intentionally change only weight values,
    not runtime flags such as compile, CUDA graph, or native acceleration.
    """

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = _maybe_build_wan_encoder_checkpoint(
        out_dir,
        encoder_cfg=encoder_cfg,
        seed=seed,
    )
    decoder_path = _build_decoder_checkpoint(
        out_dir,
        decoder_cfg=decoder_cfg,
        seed=seed + 1,
    )
    native_fp8_path = None
    if native_vae_fp8:
        if encoder_path is None:
            raise TypeError("native VAE fp8 synthetic state requires a Wan VAE encoder")
        if pixel_height is None or pixel_width is None:
            raise ValueError("pixel_height and pixel_width are required for fp8 export")
        native_fp8_path = _build_native_vae_fp8_state(
            out_dir,
            encoder_cfg=encoder_cfg,
            encoder_checkpoint_path=encoder_path,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            device=torch.device(device),
        )

    return SyntheticWorldModelAssets(
        encoder_checkpoint_path=encoder_path,
        decoder_checkpoint_path=decoder_path,
        native_vae_fp8_state_path=native_fp8_path,
    )


def _config_uses_lightvae(config: Any) -> bool:
    """Return whether ``config`` selects the lightvae VAE variant.

    This intentionally mirrors production's own detection: ``WanVAEEncoder``
    and ``WanVAEDecoder`` both choose the architecture with
    ``"lightvae" in config.checkpoint_path`` (see
    ``flashdreams/recipes/wan/autoencoder/vae.py``). Matching that exact rule
    is the invariant that guarantees the synthetic checkpoint is generated for
    the same architecture the production loader will reconstruct, so their keys
    and shapes line up. The ``or ""`` is only None-safety -- production assumes
    ``checkpoint_path`` is set -- and does not change which configs are treated
    as lightvae. If production ever changes its detection, change it here too.
    """

    return "lightvae" in (getattr(config, "checkpoint_path", None) or "")


_ARCH_FINGERPRINT_FIELDS = (
    "base_dim",
    "decoder_base_dim",
    "z_dim",
    "patch_size",
    "is_residual",
    "latent_mean",
    "latent_std",
    "checkpoint_path",
)


def _arch_fingerprint(*configs: Any, **extra: Any) -> str:
    """Short hash of the shape-determining config fields.

    Synthetic assets are cached per recipe slug and the cache persists across
    CI runs (it lives under ``FLASHDREAMS_CACHE_DIR``). Embedding this
    fingerprint in each filename means an in-place architectural change to a
    recipe -- one that keeps the same slug but alters tensor shapes -- yields a
    new filename and rebuilds, instead of the production loader silently failing
    to load a stale checkpoint at warmup.
    """

    payload: list[Any] = []
    for config in configs:
        payload.append(type(config).__name__)
        payload.extend(
            (name, repr(getattr(config, name)))
            for name in _ARCH_FINGERPRINT_FIELDS
            if hasattr(config, name)
        )
    payload.extend(sorted(extra.items()))
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return digest[:12]


# Serializes the global ``load_checkpoint`` swap below. ``WanVAE.__init__`` has
# no skip-load option, so the swap is unavoidable; the lock is held across the
# whole build so two concurrent builders cannot interleave the set/restore and
# leave the no-op installed permanently (which would silently hand every later
# real WanVAE an empty state dict and random weights).
_LOAD_CHECKPOINT_SWAP_LOCK = threading.Lock()


@contextlib.contextmanager
def _suppressed_checkpoint_load() -> Iterator[None]:
    """Build ``WanVAE`` without reading a checkpoint.

    The VAE is constructed only to capture its default-initialized architecture
    before we overwrite the weights, so the module-level ``load_checkpoint``
    (which would fail -- the synthetic file does not exist yet) is swapped for a
    no-op returning an empty state dict. The swap mutates process-global state,
    so it runs under ``_LOAD_CHECKPOINT_SWAP_LOCK``. A local set/restore keeps
    ``unittest.mock`` out of this production module.
    """

    with _LOAD_CHECKPOINT_SWAP_LOCK:
        original = wan_vae_module.load_checkpoint
        wan_vae_module.load_checkpoint = lambda *args, **kwargs: {}
        try:
            yield
        finally:
            wan_vae_module.load_checkpoint = original


def _maybe_build_wan_encoder_checkpoint(
    out_dir: Path,
    *,
    encoder_cfg: Any,
    seed: int,
) -> Path | None:
    if not isinstance(encoder_cfg, WanVAEEncoderConfig):
        return None
    use_lightvae = _config_uses_lightvae(encoder_cfg)
    fingerprint = _arch_fingerprint(encoder_cfg)
    stem = "synthetic_lightvae_encoder" if use_lightvae else "synthetic_vae_encoder"
    path = out_dir / f"{stem}_{fingerprint}.safetensors"
    if path.exists():
        return path

    vae = _new_wan_vae_from_encoder_config(encoder_cfg)
    _materialize_and_initialize(vae, seed=seed)
    _save_state_dict(path, vae.state_dict())
    return path


def _build_decoder_checkpoint(
    out_dir: Path,
    *,
    decoder_cfg: Any,
    seed: int,
) -> Path:
    if isinstance(decoder_cfg, TeahvVAEDecoderConfig):
        fingerprint = _arch_fingerprint(decoder_cfg)
        path = out_dir / f"synthetic_lighttae_decoder_{fingerprint}.safetensors"
        if not path.exists():
            taehv = TAEHV(
                checkpoint_path=None,
                use_cuda_graph=False,
                use_compile=False,
            )
            _materialize_and_initialize(taehv, seed=seed)
            _save_state_dict(path, taehv.state_dict())
        return path

    if isinstance(decoder_cfg, WanVAEDecoderConfig):
        fingerprint = _arch_fingerprint(decoder_cfg)
        path = out_dir / f"synthetic_vae_decoder_{fingerprint}.safetensors"
        if not path.exists():
            vae = _new_wan_vae_from_decoder_config(decoder_cfg)
            _materialize_and_initialize(vae, seed=seed)
            _save_state_dict(path, vae.state_dict())
        return path

    raise TypeError(
        f"unsupported synthetic decoder config: {type(decoder_cfg).__name__}"
    )


def _new_wan_vae_from_encoder_config(config: WanVAEEncoderConfig) -> Any:
    use_lightvae = _config_uses_lightvae(config)
    with _suppressed_checkpoint_load():
        return wan_vae_module.WanVAE(
            vae_path=(
                "synthetic_lightvae_encoder.safetensors"
                if use_lightvae
                else "synthetic_vae_encoder.safetensors"
            ),
            use_lightvae=use_lightvae,
            use_cuda_graph=False,
            use_compile=False,
            enable_encoder=True,
            enable_decoder=False,
            base_dim=config.base_dim,
            z_dim=config.z_dim,
            patch_size=config.patch_size,
            is_residual=config.is_residual,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            state_dict_transform=None,
        )


def _new_wan_vae_from_decoder_config(config: WanVAEDecoderConfig) -> Any:
    use_lightvae = _config_uses_lightvae(config)
    with _suppressed_checkpoint_load():
        return wan_vae_module.WanVAE(
            vae_path="synthetic_vae_decoder.safetensors",
            use_lightvae=use_lightvae,
            use_cuda_graph=False,
            use_compile=False,
            enable_encoder=False,
            enable_decoder=True,
            base_dim=config.base_dim,
            decoder_base_dim=config.decoder_base_dim,
            z_dim=config.z_dim,
            patch_size=config.patch_size,
            is_residual=config.is_residual,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            state_dict_transform=None,
        )


def _materialize_and_initialize(module: torch.nn.Module, *, seed: int) -> None:
    """Allocate meta parameters and buffers on CPU and fill them deterministically.

    ``to_empty`` leaves both parameters and buffers backed by uninitialized
    storage, so zero the buffers too; otherwise any persistent buffer (e.g. a
    BatchNorm running stat) would be serialized with garbage.
    """

    module.to_empty(device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
            else:
                parameter.zero_()
        for buffer in module.buffers():
            buffer.zero_()
    module.eval().requires_grad_(False)


def _save_state_dict(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state_dict.items()
        if not name.startswith(("_encoder_call.", "_decoder_call."))
        if not tensor.is_complex() and not tensor.is_quantized
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(tmp_path))
    tmp_path.replace(path)


def _build_native_vae_fp8_state(
    out_dir: Path,
    *,
    encoder_cfg: WanVAEEncoderConfig,
    encoder_checkpoint_path: Path,
    pixel_height: int,
    pixel_width: int,
    device: torch.device,
) -> Path:
    fingerprint = _arch_fingerprint(
        encoder_cfg,
        pixel_height=pixel_height,
        pixel_width=pixel_width,
    )
    path = out_dir / f"synthetic_lightvae_fp8_state_{fingerprint}.pt"
    if path.exists():
        return path

    export = _load_lightvae_fp8_export_module()
    from flashdreams.infra.config import derive_config

    cfg = derive_config(
        encoder_cfg,
        checkpoint_path=str(encoder_checkpoint_path),
        dtype=torch.float16,
        use_compile=False,
        use_cuda_graph=False,
        native_vae_acceleration="disabled",
        native_vae_fp8_state_path=None,
    )
    encoder = cfg.setup().to(device).eval()
    video = torch.zeros(
        (1, 3, 13, pixel_height, pixel_width),
        device=device,
        dtype=torch.float16,
    )
    amax = export._collect_activation_amax(encoder.vae, video)
    state = export._build_fp8_state(
        encoder.vae.state_dict(),
        export._activation_scales(amax, scale_max=24.0),
        scale_max=24.0,
    )
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)
    return path


_LIGHTVAE_FP8_EXPORT_SYMBOLS = (
    "_collect_activation_amax",
    "_build_fp8_state",
    "_activation_scales",
)


def _load_lightvae_fp8_export_module() -> Any:
    script_path = (
        Path(__file__).resolve().parents[3] / "scripts" / "export_lightvae_fp8_state.py"
    )
    spec = importlib.util.spec_from_file_location(
        "omnidreams_synthetic_export_lightvae_fp8_state",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [
        name for name in _LIGHTVAE_FP8_EXPORT_SYMBOLS if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            f"{script_path} is missing expected helpers {missing}; the synthetic "
            "fp8 VAE path depends on these symbols staying in sync with the script."
        )
    return module


__all__ = [
    "SyntheticWorldModelAssets",
    "build_synthetic_world_model_assets",
    "default_synthetic_asset_dir",
]
