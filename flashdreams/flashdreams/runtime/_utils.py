# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared by the experimental runtime API."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeVar

ValueT = TypeVar("ValueT")


def freeze_mapping(value: Mapping[str, ValueT]) -> Mapping[str, ValueT]:
    """Return a read-only shallow copy of ``value``."""
    return MappingProxyType(dict(value))
