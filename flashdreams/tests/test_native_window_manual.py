# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual Vulkan/display smoke coverage for the SlangPy presenter."""

from __future__ import annotations

import numpy as np
import pytest
from flashdreams.serving.native_window.presenter import SlangPyNativePresenter

pytestmark = pytest.mark.manual


def test_slangpy_native_presenter_smoke() -> None:
    presenter = SlangPyNativePresenter(
        width=64,
        height=64,
        title="FlashDreams local-window smoke",
        on_key=lambda _event, _key: None,
    )
    try:
        presenter.process_events()
        presenter.present_frame(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        presenter.close()
