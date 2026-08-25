# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HY-WorldPlay binding for the reusable Action2V application."""

from __future__ import annotations

from action2v import Action2VApplication

from flashdreams.api_v2.application import IApplication

from ...config import (
    HY_WORLDPLAY_APPLICATION_DEFAULTS,
    HY_WORLDPLAY_APPLICATION_HOOKS,
)


def create_app() -> IApplication:
    """Create the HY-WorldPlay Action2V application."""
    return Action2VApplication(
        defaults=HY_WORLDPLAY_APPLICATION_DEFAULTS,
        hooks=HY_WORLDPLAY_APPLICATION_HOOKS,
    )


__all__ = ["create_app"]
