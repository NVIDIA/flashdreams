# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal-Forcing configuration adapter for the shared T2V application."""

from typing import Any

from t2v import T2VApplication, T2VApplicationDefaults

from ...config import T2V_APPLICATION_DEFAULTS, T2V_APPLICATION_HOOKS


def load_config() -> T2VApplicationDefaults:
    """Load Causal-Forcing defaults for the shared T2V application."""
    return T2V_APPLICATION_DEFAULTS


def create_app(
    pipeline_config: Any | None = None,
    *,
    ui_renderer_factory: Any | None = None,
) -> T2VApplication:
    """Create the shared T2V application with Causal-Forcing configured.

    Args:
        pipeline_config: Pipeline override used by tests; ``None`` loads the
            integration's model config.
        ui_renderer_factory: Optional renderer factory for UI tests.
    """
    return T2VApplication(
        defaults=load_config(),
        pipeline_config=pipeline_config,
        hooks=T2V_APPLICATION_HOOKS,
        ui_renderer_factory=ui_renderer_factory,
    )


__all__ = ["create_app", "load_config"]
