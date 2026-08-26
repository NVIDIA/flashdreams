# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exception compatibility helpers shared across FlashDreams."""


def add_exception_note(error: BaseException, note: str) -> None:
    """Add diagnostic context when the running Python supports exception notes.

    ``BaseException.add_note`` was added in Python 3.11, while FlashDreams
    supports Python 3.10. Cleanup must never replace the primary failure merely
    because that compatibility method is absent.
    """
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


__all__ = ["add_exception_note"]
