# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned public API of the game-engine layer.

Import a version explicitly, ``from omnidreams_game_engine.api import v0``;
its names are stable within that version (``STABILITY.md`` beside this file).
Core modules of the package never import from ``api``.
"""

from __future__ import annotations

from omnidreams_game_engine.api import v0
from omnidreams_game_engine.api.v0 import API_VERSION

__all__ = ["API_VERSION", "v0"]
