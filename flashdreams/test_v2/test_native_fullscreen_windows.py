# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Windows fullscreen round trip using a hidden, GPU-free window."""

import ctypes
import sys
import uuid
from ctypes import wintypes

import pytest

from flashdreams.runtime_v2.native_window_client_window import (
    _SlangPyNativeWindowPresenter,
)

pytestmark = pytest.mark.manual


@pytest.mark.skipif(sys.platform != "win32", reason="Requires a Windows desktop")
def test_native_fullscreen_restores_window_style_and_position() -> None:
    spy = pytest.importorskip("slangpy")
    title = f"FlashDreams fullscreen check {uuid.uuid4()}"
    window = spy.Window(
        width=320,
        height=240,
        title=title,
        mode=spy.WindowMode.minimized,
        resizable=False,
    )
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    get_style = user32.GetWindowLongPtrW
    get_style.argtypes = [wintypes.HWND, ctypes.c_int]
    get_style.restype = ctypes.c_ssize_t
    hwnd = user32.FindWindowW(None, title)
    assert hwnd
    original = wintypes.RECT()
    assert user32.GetWindowRect(hwnd, ctypes.byref(original))
    original_style = get_style(hwnd, -16)
    presenter = object.__new__(_SlangPyNativeWindowPresenter)
    presenter._native_hwnd = hwnd
    presenter._closed = False
    presenter._windowed_placement = None
    presenter._fullscreen_enabled = False
    presenter._optimal_size = None
    try:
        assert presenter.request_fullscreen(True)
        size = presenter.fullscreen_size(True)
        assert size is not None and size[0] >= 320 and size[1] >= 240
        assert get_style(hwnd, -16) & 0x00CF0000 == 0
        fullscreen_rect = wintypes.RECT()
        assert user32.GetWindowRect(hwnd, ctypes.byref(fullscreen_rect))
        assert (
            fullscreen_rect.right - fullscreen_rect.left,
            fullscreen_rect.bottom - fullscreen_rect.top,
        ) == size
        assert presenter.request_fullscreen(False)
        assert get_style(hwnd, -16) == original_style
        restored = wintypes.RECT()
        assert user32.GetWindowRect(hwnd, ctypes.byref(restored))
        assert bytes(restored) == bytes(original)
        assert presenter.fullscreen_size(False) is not None
        assert presenter.request_fullscreen(False)
        assert user32.GetWindowRect(hwnd, ctypes.byref(restored))
        assert bytes(restored) == bytes(original)
    finally:
        window.close()
