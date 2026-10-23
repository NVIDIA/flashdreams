# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HandDreams on the FlashDreams V2 API."""

__all__ = ["HandDreamsV2Application"]


def __getattr__(name: str):
    """Load application dependencies only when the application is requested."""
    if name in __all__:
        from . import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
