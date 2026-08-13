# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility bridge from ``flashdreams-run t2v`` to integration apps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

from t2v import (
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    FIELD_PROMPT,
    FIELD_TOTAL_BLOCKS,
    T2VDemoAdapter,
    T2VInputProvider,
    T2VModelConfig,
    T2VRuntime,
    T2VScenario,
    T2VSession,
    model_config_from_runner,
)

from flashdreams.demo import Application, DemoAdapterApplication


@dataclass(frozen=True, slots=True)
class T2VBackendBridge:
    """One legacy backend key routed to one integration-owned app."""

    key: str
    label: str
    app_slug: str
    app_module: str
    config_module: str


_BACKENDS: dict[str, T2VBackendBridge] = {
    "causal-forcing": T2VBackendBridge(
        key="causal-forcing",
        label="Causal-Forcing (Wan 2.1)",
        app_slug="causal-forcing-t2v",
        app_module="causal_forcing.t2v.app",
        config_module="causal_forcing.config",
    ),
    "cosmos-predict2": T2VBackendBridge(
        key="cosmos-predict2",
        label="Cosmos Predict2",
        app_slug="cosmos-predict2-t2v",
        app_module="cosmos_predict2.t2v.app",
        config_module="cosmos_predict2.config",
    ),
    "self-forcing": T2VBackendBridge(
        key="self-forcing",
        label="Self-Forcing (Wan 2.1)",
        app_slug="self-forcing-t2v",
        app_module="self_forcing.t2v.app",
        config_module="self_forcing.config",
    ),
}


def backend_choices() -> tuple[str, ...]:
    """Return stable legacy CLI backend choices."""
    return tuple(_BACKENDS)


def backend_metadata() -> list[dict[str, Any]]:
    """Return browser-safe backend metadata derived from integration configs."""
    metadata: list[dict[str, Any]] = []
    for bridge in _BACKENDS.values():
        model = model_from_backend(bridge.key)
        metadata.append(
            {
                "key": bridge.key,
                "label": bridge.label,
                "default_preset": model.preset_id,
                "presets": _t2v_preset_ids(bridge),
                "application": bridge.app_slug,
            }
        )
    return metadata


def default_pipeline() -> Any:
    """Return the legacy runner's suppressed default pipeline value."""
    return model_from_backend("causal-forcing").pipeline


def model_from_backend(
    backend: str,
    preset_id: str | None = None,
) -> T2VModelConfig:
    """Resolve a legacy backend/preset selector to an integration-owned model."""
    bridge = _resolve_backend(backend)
    if preset_id is None:
        return _with_legacy_backend_option(
            _default_model_from_application(bridge), bridge
        )
    return _with_legacy_backend_option(
        _model_from_runner_config(bridge, preset_id), bridge
    )


def make_adapter(
    backend: str,
    preset_id: str | None = None,
    *,
    write_download_artifact: bool = False,
) -> T2VDemoAdapter:
    """Build an adapter from a legacy CLI/UI backend key."""
    return T2VDemoAdapter(
        model=model_from_backend(backend, preset_id),
        write_download_artifact=write_download_artifact,
    )


def _resolve_backend(value: str) -> T2VBackendBridge:
    try:
        return _BACKENDS[value]
    except KeyError as exc:
        raise ValueError(
            f"Unknown backend {value!r}. Available backends: {', '.join(_BACKENDS)}."
        ) from exc


def _default_model_from_application(bridge: T2VBackendBridge) -> T2VModelConfig:
    app = _load_application(bridge)
    if not isinstance(app, DemoAdapterApplication):
        raise TypeError(
            f"T2V application {bridge.app_slug!r} must return "
            f"DemoAdapterApplication, got {type(app).__name__}."
        )
    adapter = app.adapter
    model = getattr(adapter, "model", None)
    if not isinstance(model, T2VModelConfig):
        raise TypeError(f"T2V application {bridge.app_slug!r} must use T2VDemoAdapter.")
    return model


def _load_application(bridge: T2VBackendBridge) -> Application:
    for entry_point in entry_points(group="flashdreams.applications"):
        if entry_point.name == bridge.app_slug:
            factory = entry_point.load()
            return _coerce_application(
                factory() if callable(factory) else factory,
                app_slug=bridge.app_slug,
            )
    factory = getattr(import_module(bridge.app_module), "create_app")
    return _coerce_application(factory(), app_slug=bridge.app_slug)


def _coerce_application(value: object, *, app_slug: str) -> Application:
    if not isinstance(value, Application):
        raise TypeError(
            f"T2V application {app_slug!r} must return Application, "
            f"got {type(value).__name__}."
        )
    return value


def _model_from_runner_config(
    bridge: T2VBackendBridge, preset_id: str
) -> T2VModelConfig:
    runner_configs = getattr(import_module(bridge.config_module), "RUNNER_CONFIGS")
    try:
        runner = runner_configs[preset_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown {bridge.key} preset {preset_id!r}. Available presets: "
            f"{', '.join(_t2v_preset_ids(bridge))}."
        ) from exc
    if not _is_t2v_preset(runner.runner_name):
        raise ValueError(
            f"Preset {preset_id!r} is not a T2V preset for backend {bridge.key!r}."
        )
    default_model = _default_model_from_application(bridge)
    return model_config_from_runner(model_id=default_model.model_id, runner=runner)


def _t2v_preset_ids(bridge: T2VBackendBridge) -> tuple[str, ...]:
    runner_configs = getattr(import_module(bridge.config_module), "RUNNER_CONFIGS")
    return tuple(name for name in runner_configs if _is_t2v_preset(name))


def _is_t2v_preset(name: str) -> bool:
    return "-t2v-" in name


def _with_legacy_backend_option(
    model: T2VModelConfig,
    bridge: T2VBackendBridge,
) -> T2VModelConfig:
    return replace(
        model,
        runtime_options={
            **model.runtime_options,
            "backend": bridge.key,
            "application": bridge.app_slug,
        },
    )


__all__ = [
    "FIELD_FPS",
    "FIELD_PIXEL_HEIGHT",
    "FIELD_PIXEL_WIDTH",
    "FIELD_PROMPT",
    "FIELD_TOTAL_BLOCKS",
    "T2VDemoAdapter",
    "T2VBackendBridge",
    "T2VInputProvider",
    "T2VModelConfig",
    "T2VRuntime",
    "T2VScenario",
    "T2VSession",
    "backend_choices",
    "backend_metadata",
    "default_pipeline",
    "make_adapter",
    "model_from_backend",
]
