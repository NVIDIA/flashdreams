# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared ``Loading...`` overlay used while the backend is warming up.

:class:`omnidreams.interactive_drive.app.InteractiveDriveApp` pre-renders this once at
startup into a :class:`~omnidreams.interactive_drive.types.PresentedFrame` and seeds
:func:`~omnidreams.interactive_drive.runtime.loop.run_main_loop` with it as the
``initial_presented_frame``. The loop re-presents that frame on every
display tick until the pipeline produces its first chunk (~60 s for the
world-model backend due to HF download + torch.compile), so the user
sees a high-contrast "Loading world model..." box instead of a frozen
initial camera frame and reasonably assuming the demo has hung.

PIL + ImageDraw.textbbox isn't cheap but it's called once per session.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_loading_overlay(
    base_rgb_host_uint8: np.ndarray,
    message: str = "Loading world model...",
) -> np.ndarray:
    """Return a new RGB uint8 array with ``message`` drawn centered on top
    of ``base_rgb_host_uint8``.

    The input is assumed to be ``(H, W, 3)`` uint8 RGB (matching
    :attr:`omnidreams.interactive_drive.types.PresentedFrame.rgb_host_uint8`). Output has
    the same shape and dtype. A slight full-frame dim is applied behind
    the text box so the message stays legible regardless of scene
    contents.
    """
    base = Image.fromarray(base_rgb_host_uint8).convert("RGBA")
    width, height = base.size

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Dim everything slightly so the scene doesn't compete with the box.
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0, 110))

    # Pillow 10.1+ accepts ``size=`` on load_default; older versions
    # return a tiny 7x10 bitmap, which still works just looks dinky.
    try:
        font = ImageFont.load_default(size=max(28, height // 20))
    except TypeError:
        font = ImageFont.load_default()

    left, top, right, bottom = draw.textbbox((0, 0), message, font=font)
    text_w = right - left
    text_h = bottom - top
    cx, cy = width // 2, height // 2
    pad = max(12, text_h // 2)

    # Rounded-ish callout box: a dark solid fill plus a 2 px bright border.
    box = (
        cx - text_w // 2 - pad,
        cy - text_h // 2 - pad,
        cx + text_w // 2 + pad,
        cy + text_h // 2 + pad,
    )
    draw.rectangle(box, fill=(20, 20, 20, 230), outline=(240, 240, 240, 240), width=2)
    draw.text(
        (cx - text_w // 2 - left, cy - text_h // 2 - top),
        message,
        fill=(255, 255, 255, 255),
        font=font,
    )

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(composed, dtype=np.uint8)
