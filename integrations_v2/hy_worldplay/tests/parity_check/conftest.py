# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest collection guard: keep main discovery out of the cloned vendor tree and sub-venv."""

from __future__ import annotations

collect_ignore_glob = [
    "HY-WorldPlay/**",
    ".venv/**",
]
