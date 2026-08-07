# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility re-export; this module now lives in the app package.

Kept so existing imports resolve while the engine moves to
``apps/interactive-drive``. Import from :mod:`interactive_drive_app.input`
in new code.
"""

from interactive_drive_app.input.backend import (
    InputBackend,
    SampledInput,
)

__all__ = [
    "InputBackend",
    "SampledInput",
]
