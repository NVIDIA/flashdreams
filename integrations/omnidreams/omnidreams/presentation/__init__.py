# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model-agnostic presentation backends: presenter/overlay contracts, canvas, interop.

Importing this package pulls in PIL and numpy only. SlangPy is imported by
:mod:`omnidreams.presentation.local_window` and torch by
:mod:`omnidreams.presentation.cuda_interop`, both at construction time, so
contracts and geometry stay usable on hosts with no GPU or display.
"""

from omnidreams.presentation.base import (
    HudOverlay,
    InputSink,
    KeyAction,
    KeyEvent,
    PointerAction,
    PointerEvent,
    PresenterBackend,
    Rect,
    SupportsPrepareFrame,
)
from omnidreams.presentation.canvas import (
    LRUCache,
    allocate_canvas,
    draw_status_overlay,
    fit_rect,
    measure_text,
    resolve_font,
    truncate_text_to_width,
)
from omnidreams.presentation.frame import (
    DisplayFrame,
    as_rgb_host_uint8,
    has_cuda_tensor,
    prefetch_frame,
    rgb_source_size,
)
from omnidreams.presentation.local_window import LocalWindowPresenter, WindowConfig

__all__ = [
    "DisplayFrame",
    "HudOverlay",
    "InputSink",
    "KeyAction",
    "KeyEvent",
    "LRUCache",
    "LocalWindowPresenter",
    "PointerAction",
    "PointerEvent",
    "PresenterBackend",
    "Rect",
    "SupportsPrepareFrame",
    "WindowConfig",
    "allocate_canvas",
    "as_rgb_host_uint8",
    "draw_status_overlay",
    "fit_rect",
    "has_cuda_tensor",
    "measure_text",
    "prefetch_frame",
    "resolve_font",
    "rgb_source_size",
    "truncate_text_to_width",
]
