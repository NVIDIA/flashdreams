# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OmniDreams VAE native acceleration wrappers."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import Literal, get_args

import torch
from omnidreams.native import omnidreams_singleview
from omnidreams.native.acceleration import (
    NativeAccelerationConfig,
    NativeAccelerationMode,
    NativeAccelerationUnavailable,
    NativeBackendSelection,
    require_extension_symbols,
)
from omnidreams.native.primitives import (
    NativePrepError,
    NativeTensorSpec,
    prepare_tensor_for_native,
)
from torch import Tensor

from flashdreams.recipes.wan.autoencoder.vae import (
    TEMPORAL_WINDOW,
    WanVAECache,
    WanVAEEncoder,
    WanVAEEncoderConfig,
)

NativeVAEBackend = Literal["fp8"]
NATIVE_LIGHTVAE_FP8_STATE_ENV = "OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH"


@dataclass(kw_only=True)
class OmnidreamsWanVAEEncoderConfig(WanVAEEncoderConfig):
    """Wan VAE encoder config with OmniDreams native acceleration knobs."""

    _target: type = field(default_factory=lambda: OmnidreamsWanVAEEncoder)

    native_vae_acceleration: NativeAccelerationMode = "disabled"
    native_vae_build_root: str | None = None
    native_vae_max_jobs: int | str | None = None
    native_vae_verbose_build: bool = False
    native_vae_backend: NativeVAEBackend = "fp8"
    native_vae_fp8_state_path: str | None = None


def _native_acceleration_config(
    config: OmnidreamsWanVAEEncoderConfig,
) -> NativeAccelerationConfig:
    return NativeAccelerationConfig(
        mode=config.native_vae_acceleration,
        build_root=config.native_vae_build_root,
        max_jobs=config.native_vae_max_jobs,
        verbose_build=config.native_vae_verbose_build,
    )


def _native_vae_fp8_state_path(
    config: OmnidreamsWanVAEEncoderConfig,
) -> str | None:
    return config.native_vae_fp8_state_path or os.environ.get(
        NATIVE_LIGHTVAE_FP8_STATE_ENV
    )


def _native_vae_availability_check(
    *,
    component: str,
    backend: NativeVAEBackend,
    run_symbol: str,
    setup_symbols: tuple[str, ...] = (),
) -> Callable[[ModuleType], tuple[bool, str]]:
    symbol_check = require_extension_symbols(
        "omnidreams_vae_backend_status",
        run_symbol,
        *setup_symbols,
    )

    def check(extension: ModuleType) -> tuple[bool, str]:
        ok, reason = symbol_check(extension)
        if not ok:
            return ok, reason

        status = extension.omnidreams_vae_backend_status(component, backend)
        if not isinstance(status, Mapping):
            return False, "omnidreams_vae_backend_status did not return a mapping"

        available = bool(status.get("available", False))
        status_reason = str(
            status.get(
                "reason",
                "native VAE backend is available"
                if available
                else "native VAE backend is unavailable",
            )
        )
        return available, status_reason

    return check


def _native_vae_preflight_reason(
    *,
    component: str,
    config: OmnidreamsWanVAEEncoderConfig,
) -> str | None:
    if config.native_vae_backend not in get_args(NativeVAEBackend):
        return (
            f"native VAE backend must be one of {get_args(NativeVAEBackend)}, "
            f"got {config.native_vae_backend!r}"
        )
    if (
        component == "vae_encoder"
        and config.native_vae_backend == "fp8"
        and not _native_vae_fp8_state_path(config)
    ):
        return (
            "native LightVAE fp8 backend requires native_vae_fp8_state_path "
            f"or {NATIVE_LIGHTVAE_FP8_STATE_ENV}"
        )
    return None


def _unavailable_native_vae_selection(
    *,
    component: str,
    config: OmnidreamsWanVAEEncoderConfig,
    reason: str,
) -> NativeBackendSelection:
    if config.native_vae_acceleration == "required":
        raise NativeAccelerationUnavailable(reason)
    return NativeBackendSelection(
        component=component,
        mode=config.native_vae_acceleration,
        enabled=False,
        reason=reason,
    )


def _select_native_vae_backend(
    *,
    component: str,
    config: OmnidreamsWanVAEEncoderConfig,
    run_symbol: str,
    setup_symbols: tuple[str, ...] = (),
) -> NativeBackendSelection:
    reason = _native_vae_preflight_reason(component=component, config=config)
    if reason is not None:
        return _unavailable_native_vae_selection(
            component=component,
            config=config,
            reason=reason,
        )
    return omnidreams_singleview.select_backend(
        component,
        _native_acceleration_config(config),
        availability_check=_native_vae_availability_check(
            component=component,
            backend=config.native_vae_backend,
            run_symbol=run_symbol,
            setup_symbols=setup_symbols,
        ),
    )


def _wan_encoder_symbols(backend: NativeVAEBackend) -> tuple[str, tuple[str, ...]]:
    suffix = backend
    return (
        f"omnidreams_vae_encode_wan_{suffix}",
        (
            f"omnidreams_vae_create_wan_encoder_{suffix}",
            f"omnidreams_vae_reset_wan_encoder_{suffix}",
        ),
    )


class _NativeWanVAEEncoderExecutor:
    _FP8_INPUT_SPEC = NativeTensorSpec(
        name="wan_vae_encoder_input",
        layout="B C T H W",
        shape=(1, 3, None, None, None),
        dtypes=(torch.float16,),
        axis_divisibility=(("H", 8), ("W", 8)),
    )

    def __init__(
        self,
        *,
        selection: NativeBackendSelection,
        backend: NativeVAEBackend,
    ) -> None:
        self.selection = selection
        self.extension = selection.require_extension()
        self.backend = backend
        self._helper: ModuleType | None = None
        self._native_encoder: object | None = None
        self._native_encoder_model_id: int | None = None
        self._native_encoder_device: torch.device | None = None
        self._fp8_state_path: str | None = None
        self._fp8_state: Mapping[str, Tensor] | None = None
        self._fp8_state_loaded_path: str | None = None
        self._cache_id: int | None = None
        self._cache_is_empty = True

    @property
    def _required(self) -> bool:
        return self.selection.mode == "required"

    def _fallback_or_raise(self, exc: Exception) -> None:
        if self._required:
            raise NativeAccelerationUnavailable(str(exc)) from exc

    @classmethod
    def from_config(
        cls,
        config: OmnidreamsWanVAEEncoderConfig,
    ) -> "_NativeWanVAEEncoderExecutor | None":
        run_symbol, setup_symbols = _wan_encoder_symbols(config.native_vae_backend)
        selection = _select_native_vae_backend(
            component="vae_encoder",
            config=config,
            run_symbol=run_symbol,
            setup_symbols=setup_symbols,
        )
        if not selection.enabled:
            return None
        executor = cls(selection=selection, backend=config.native_vae_backend)
        executor._fp8_state_path = _native_vae_fp8_state_path(config)
        return executor

    def _helper_module(self) -> ModuleType:
        if self._helper is None:
            self._helper = omnidreams_singleview.load_python_module("vae_weights")
        return self._helper

    def _load_fp8_state(self, helper: ModuleType) -> Mapping[str, Tensor]:
        if self._fp8_state_path is None:
            raise NativeAccelerationUnavailable(
                "native LightVAE fp8 backend requires native_vae_fp8_state_path "
                f"or {NATIVE_LIGHTVAE_FP8_STATE_ENV}"
            )
        if (
            self._fp8_state is not None
            and self._fp8_state_loaded_path == self._fp8_state_path
        ):
            return self._fp8_state
        state = helper.load_lightvae_fp8_state(self._fp8_state_path)
        self._fp8_state = state
        self._fp8_state_loaded_path = self._fp8_state_path
        return state

    def _get_native_encoder(self, model: object, device: torch.device) -> object | None:
        model_id = id(model)
        if (
            self._native_encoder is not None
            and self._native_encoder_model_id == model_id
            and self._native_encoder_device == device
        ):
            return self._native_encoder

        try:
            helper = self._helper_module()
            if self.backend == "fp8":
                fp8_state = self._load_fp8_state(helper)
                state = helper.build_lightvae_encoder_fp8_staged_state(
                    model,
                    fp8_state,
                    self.extension,
                )
                native_encoder = self.extension.omnidreams_vae_create_wan_encoder_fp8(
                    state,
                )
            else:
                raise NativeAccelerationUnavailable(
                    f"unsupported native LightVAE encoder backend {self.backend!r}"
                )
        except Exception as exc:
            self._fallback_or_raise(exc)
            return None

        self._native_encoder = native_encoder
        self._native_encoder_model_id = model_id
        self._native_encoder_device = device
        self._cache_is_empty = True
        return native_encoder

    def _bind_cache(self, cache: WanVAECache) -> None:
        cache_id = id(cache)
        if self._cache_id == cache_id:
            return
        self._cache_id = cache_id
        self._cache_is_empty = True
        if self._native_encoder is not None:
            self._reset_native_encoder()

    def after_initialize_autoregressive_cache(self, cache: object) -> None:
        if self._native_encoder is not None:
            self._reset_native_encoder()
        self._cache_id = id(cache)
        self._cache_is_empty = True

    def _reset_native_encoder(self) -> None:
        reset = getattr(
            self.extension,
            f"omnidreams_vae_reset_wan_encoder_{self.backend}",
        )
        reset(self._native_encoder)

    def _encode_chunk(self, native_encoder: object, input_bcthw: Tensor) -> Tensor:
        encode = getattr(
            self.extension,
            f"omnidreams_vae_encode_wan_{self.backend}",
        )
        return encode(
            native_encoder,
            input_bcthw,
            True,
        )

    def try_encode(
        self,
        model: object,
        input_bcthw: Tensor,
        cache: WanVAECache,
    ) -> Tensor | None:
        if cache.enc_state and self._cache_id != id(cache):
            exc = NativeAccelerationUnavailable(
                "native LightVAE encoder cannot attach to a WanVAE cache that "
                "was already advanced by the Python encoder"
            )
            if self._required:
                raise exc
            return None

        try:
            prepared = prepare_tensor_for_native(input_bcthw, self._FP8_INPUT_SPEC)
        except NativePrepError as exc:
            if self._cache_id == id(cache) and not self._cache_is_empty:
                raise NativeAccelerationUnavailable(
                    "native LightVAE encoder cannot fall back after its internal "
                    "streaming cache has advanced"
                ) from exc
            self._fallback_or_raise(exc)
            return None

        self._bind_cache(cache)
        native_encoder = self._get_native_encoder(model, prepared.tensor.device)
        if native_encoder is None:
            return None

        x = prepared.tensor
        outs: list[Tensor] = []
        if self._cache_is_empty:
            outs.append(self._encode_chunk(native_encoder, x[:, :, :1]))
            x = x[:, :, 1:]
        else:
            try:
                if x.shape[2] % TEMPORAL_WINDOW != 0:
                    raise NativePrepError(
                        "Streaming LightVAE encode after the first chunk requires "
                        f"T % {TEMPORAL_WINDOW} == 0; got T={x.shape[2]}"
                    )
            except NativePrepError as exc:
                if not self._cache_is_empty:
                    raise NativeAccelerationUnavailable(
                        "native LightVAE encoder cannot fall back after its "
                        "internal streaming cache has advanced"
                    ) from exc
                self._fallback_or_raise(exc)
                return None

        t = x.shape[2]
        body = (t // TEMPORAL_WINDOW) * TEMPORAL_WINDOW
        for i in range(0, body, TEMPORAL_WINDOW):
            outs.append(
                self._encode_chunk(native_encoder, x[:, :, i : i + TEMPORAL_WINDOW])
            )
        if body < t:
            outs.append(self._encode_chunk(native_encoder, x[:, :, body:]))

        self._cache_is_empty = False
        if len(outs) == 1:
            return outs[0]
        return torch.cat(outs, dim=2)


class OmnidreamsWanVAEEncoder(WanVAEEncoder):
    """Wan VAE encoder with optional OmniDreams native dispatch."""

    config: OmnidreamsWanVAEEncoderConfig

    def __init__(self, config: OmnidreamsWanVAEEncoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self._native_vae_selection: NativeBackendSelection | None = None
        self._native_vae_executor: _NativeWanVAEEncoderExecutor | None = None
        if config.native_vae_acceleration != "disabled":
            run_symbol, setup_symbols = _wan_encoder_symbols(config.native_vae_backend)
            self._native_vae_selection = _select_native_vae_backend(
                component="vae_encoder",
                config=config,
                run_symbol=run_symbol,
                setup_symbols=setup_symbols,
            )
            if self._native_vae_selection.enabled:
                self._native_vae_executor = _NativeWanVAEEncoderExecutor(
                    selection=self._native_vae_selection,
                    backend=config.native_vae_backend,
                )
                self._native_vae_executor._fp8_state_path = _native_vae_fp8_state_path(
                    config
                )

    @property
    def native_vae_selection(self) -> NativeBackendSelection | None:
        return self._native_vae_selection

    def initialize_autoregressive_cache(self) -> WanVAECache:
        cache = self.vae.prepare_cache()
        if self._native_vae_executor is not None:
            self._native_vae_executor.after_initialize_autoregressive_cache(cache)
        return cache

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: WanVAECache | None = None,
    ) -> Tensor:
        if cache is None:
            cache = self.initialize_autoregressive_cache()

        assert input.ndim >= 4, "Expected input to have shape [..., T, C, H, W]"

        *batch_shape, _t, _c, _h, _w = input.shape
        batch_size = math.prod(batch_shape)
        x = input.reshape(batch_size, *input.shape[-4:]).to(dtype=self.config.dtype)
        input_bcthw = x.transpose(1, 2)

        if self._native_vae_executor is not None:
            native_z = self._native_vae_executor.try_encode(
                self.vae,
                input_bcthw,
                cache,
            )
            if native_z is not None:
                z = native_z.transpose(1, 2)
                return z.reshape(*batch_shape, *z.shape[1:])

        z = self.vae.encode(input_bcthw, cache=cache).transpose(1, 2)
        return z.reshape(*batch_shape, *z.shape[1:])


__all__ = [
    "NativeVAEBackend",
    "OmnidreamsWanVAEEncoder",
    "OmnidreamsWanVAEEncoderConfig",
]
