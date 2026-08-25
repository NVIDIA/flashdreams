# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.2 TI2V adapter for the shared T2V application."""

from collections.abc import Callable
from typing import Any

from t2v import T2VApplication, T2VApplicationDefaults

from ...config import (
    TI2V_APPLICATION_DEFAULTS,
    TI2V_APPLICATION_HOOKS,
    create_ti2v_application_hooks,
)


def load_config() -> T2VApplicationDefaults:
    """Load Wan 2.2 TI2V defaults from the model config."""
    return TI2V_APPLICATION_DEFAULTS


def create_app(
    pipeline_config: Any | None = None,
    *,
    image_loader: Callable[..., Any] | None = None,
    ui_renderer_factory: Callable[[int, int], Any] | None = None,
) -> T2VApplication:
    """Create the shared T2V application with Wan 2.2 TI2V configured.

    Args:
        pipeline_config: Pipeline override used by tests; ``None`` loads the
            integration's pipeline config.
        image_loader: Optional first-frame tensor loader override.
        ui_renderer_factory: Optional renderer factory for UI tests.

    Returns:
        Shared T2V application configured for Wan 2.2 TI2V.
    """
    hooks = (
        TI2V_APPLICATION_HOOKS
        if image_loader is None
        else create_ti2v_application_hooks(image_loader)
    )
    return T2VApplication(
        defaults=load_config(),
        pipeline_config=pipeline_config,
        hooks=hooks,
        ui_renderer_factory=ui_renderer_factory,
    )


__all__ = ["create_app", "load_config"]
