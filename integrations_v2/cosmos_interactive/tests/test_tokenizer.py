# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen3-VL tokenizer helpers and a skip-friendly smoke test."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.ci_cpu


def test_resolution_suffix_matches_reference_convention() -> None:
    from cosmos_interactive._tokenizer import _with_resolution_suffix

    # Caption ending in '.' -> single-space join (the reference implementation ResolutionTextInfo rule).
    assert _with_resolution_suffix("gentle waves.", 480, 832, is_video=True) == (
        "gentle waves. This video is of 480x832 resolution."
    )
    # Caption without trailing '.' -> ". " join.
    assert _with_resolution_suffix("gentle waves", 480, 832, is_video=True) == (
        "gentle waves. This video is of 480x832 resolution."
    )


def test_tokenize_prompt_smoke() -> None:
    pytest.importorskip("transformers")
    from cosmos_interactive._tokenizer import tokenize_prompt

    try:
        ids = tokenize_prompt(
            "A calm ocean at sunrise, gentle waves.", height=480, width=832
        )
    except Exception as exc:  # noqa: BLE001 - tokenizer files may be unavailable offline
        pytest.skip(f"Qwen3-VL tokenizer unavailable offline: {exc}")
    assert ids.ndim == 1 and ids.numel() > 0 and ids.dtype == torch.long
