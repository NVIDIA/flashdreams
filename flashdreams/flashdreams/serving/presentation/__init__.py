# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model-agnostic presentation backends: presenter/overlay contracts, canvas, interop.

The core an integration builds a native demo on: it owns window creation,
device and swapchain setup, resize recovery, and the camera composite, and
delegates demo-owned chrome and local interaction to a
:class:`~flashdreams.serving.presentation.base.HudOverlay`.

Requires the ``local-window`` extra for Pillow and SlangPy. Importing the
package needs only Pillow and numpy; SlangPy is imported by
:mod:`~flashdreams.serving.presentation.local_window` and torch by
:mod:`~flashdreams.serving.presentation.cuda_interop`, both at construction
time, so contracts and geometry stay usable on hosts with no GPU or display.
"""

from flashdreams.serving.presentation.base import (
    HudOverlay,
    KeyAction,
    KeyEvent,
    NullOverlay,
    PointerAction,
    PointerEvent,
    PresenterBackend,
    Rect,
    SupportsPrepareFrame,
)
from flashdreams.serving.presentation.canvas import (
    LRUCache,
    allocate_canvas,
    draw_status_overlay,
    fit_rect,
    measure_text,
    resolve_font,
    truncate_text_to_width,
)
from flashdreams.serving.presentation.composite import CompositeOverlay
from flashdreams.serving.presentation.compositor import (
    CameraMode,
    FrameCompositor,
)
from flashdreams.serving.presentation.frame import (
    DisplayFrame,
    as_rgb_host_uint8,
    has_cuda_tensor,
    prefetch_frame,
    rgb_source_size,
)
from flashdreams.serving.presentation.local_window import (
    LocalWindowPresenter,
    WindowConfig,
)
from flashdreams.serving.presentation.panel import PanelOverlay, PanelWidget

__all__ = [
    "CameraMode",
    "CompositeOverlay",
    "DisplayFrame",
    "FrameCompositor",
    "HudOverlay",
    "KeyAction",
    "KeyEvent",
    "LRUCache",
    "LocalWindowPresenter",
    "NullOverlay",
    "PanelOverlay",
    "PanelWidget",
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
