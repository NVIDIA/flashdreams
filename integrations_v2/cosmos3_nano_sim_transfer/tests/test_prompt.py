# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer prompt text and token ids against cosmos-framework's."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from cosmos3_nano_sim_transfer.impl.prompt import (
    SYSTEM_PROMPT,
    prompt_token_ids,
    transfer_caption,
)

pytestmark = pytest.mark.ci_cpu

FIXTURE = Path(__file__).parent / "data" / "cosmos_framework_prompt_ids.json"


# Text cosmos-framework 86e3528 builds for these requests
# (``inference._format_prompt_with_template`` with the Transfer templates and
# ``fps=int(round(output_fps))``, as ``transfer.py`` calls it).
COSMOS_FRAMEWORK_TEXT = [
    (
        "A taxi drives down a city street",
        201,
        30.0,
        "A taxi drives down a city street. The video is 6.7 seconds long and is of "
        "30 FPS. This video is of 480x832 resolution.",
    ),
    (
        "A taxi drives down a city street...",
        201,
        30.0,
        "A taxi drives down a city street. The video is 6.7 seconds long and is of "
        "30 FPS. This video is of 480x832 resolution.",
    ),
    (
        "  A street.  ",
        201,
        30.0,
        "A street. The video is 6.7 seconds long and is of 30 FPS. "
        "This video is of 480x832 resolution.",
    ),
    (
        "A street.",
        601,
        29.97,
        "A street. The video is 20.0 seconds long and is of 30 FPS. "
        "This video is of 480x832 resolution.",
    ),
    (
        "A street",
        600,
        30.0,
        "A street. The video is 20.0 seconds long and is of 30 FPS. "
        "This video is of 480x832 resolution.",
    ),
    (
        "A street.",
        121,
        24.0,
        "A street. The video is 5.0 seconds long and is of 24 FPS. "
        "This video is of 480x832 resolution.",
    ),
    ("A street.", 1, 30.0, "A street. This video is of 480x832 resolution."),
]


@pytest.mark.parametrize(
    ("caption", "num_frames", "fps", "expected"), COSMOS_FRAMEWORK_TEXT
)
def test_text_is_formatted_as_cosmos_framework_formats_it(
    caption: str, num_frames: int, fps: float, expected: str
) -> None:
    assert transfer_caption(caption, num_frames=num_frames, fps=fps) == expected


def test_caption_guards() -> None:
    with pytest.raises(ValueError):
        transfer_caption("A street.", num_frames=0, fps=30.0)
    with pytest.raises(ValueError):
        transfer_caption("A street.", num_frames=201, fps=0.4)
    # JSON that is not an object stays plain text, as in cosmos-framework.
    assert transfer_caption("42", num_frames=1, fps=30.0) == (
        "42. This video is of 480x832 resolution."
    )


def test_structured_prompt_matches_framework_metadata_and_serialization() -> None:
    # Pinned public Framework inference.py _format_json_prompt_with_template;
    # insertion order, default separators and ASCII escaping are token-sensitive.
    caption = '{"scene":"café", "fps": 1, "duration":"99s", "resolution":{"H":1,"W":2}, "aspect_ratio":"1,1"}'
    assert transfer_caption(
        caption, num_frames=201, fps=29.97, aspect_ratio="16,9"
    ) == (
        '{"scene": "caf\\u00e9", "fps": 30.0, "duration": "6s", '
        '"resolution": {"H": 480, "W": 832}, "aspect_ratio": "16,9"}'
    )


def test_structured_image_removes_temporal_fields_and_preserves_optional_aspect() -> (
    None
):
    caption = '{"scene":"A street.","duration":"9s","fps":24,"aspect_ratio":"16,9"}'
    assert transfer_caption(caption, num_frames=1, fps=30.0) == (
        '{"scene": "A street.", "aspect_ratio": "16,9", '
        '"resolution": {"H": 480, "W": 832}}'
    )
    assert transfer_caption("{}", num_frames=121, fps=24) == (
        '{"duration": "5s", "fps": 24.0, "resolution": {"H": 480, "W": 832}}'
    )


def test_structured_prompt_emphasis_is_appended_after_json() -> None:
    result = transfer_caption(
        '{"scene":"A street."}',
        num_frames=121,
        fps=24,
        hint="seg",
        emphasize_control_in_prompt=True,
    )
    assert result.startswith(
        '{"scene": "A street.", "duration": "5s", "fps": 24.0, '
        '"resolution": {"H": 480, "W": 832}} Follow the seg'
    )


def test_system_prompt_is_the_transfer_one() -> None:
    assert SYSTEM_PROMPT.endswith(
        "control signals (edge maps, blur, depth, or segmentation)."
    )


@pytest.mark.parametrize("hint", ["depth", "seg", "edge", "blur"])
def test_control_emphasis_matches_framework_request_suffix(hint: str) -> None:
    # Framework transfer.py at 5950e7b4, lines 935–944. This is a request
    # option; the release metadata's true value must not change the default.
    plain = transfer_caption("A street.", num_frames=121, fps=24.0, hint=hint)
    emphasized = transfer_caption(
        "A street.",
        num_frames=121,
        fps=24.0,
        hint=hint,
        emphasize_control_in_prompt=True,
    )
    assert emphasized == plain + (
        f" Follow the {hint} control video precisely: shape, contour, silhouette,"
        f" position, and motion of every visible structure must align with the {hint}"
        " signal at every frame."
    )


@pytest.mark.parametrize("hint", [None, "depth, seg", "unknown"])
def test_emphasis_requires_one_known_control(hint: str | None) -> None:
    with pytest.raises(ValueError, match="one hint"):
        transfer_caption(
            "A street.",
            num_frames=1,
            fps=30.0,
            hint=hint,
            emphasize_control_in_prompt=True,
        )


def test_prompt_tokens_receive_the_emphasized_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from cosmos3_nano_sim_transfer.impl import prompt as prompt_module

    recorded = []

    def tokenize(text: str, **kwargs) -> torch.Tensor:
        recorded.append((text, kwargs))
        return torch.tensor([1, 2])

    monkeypatch.setattr(prompt_module, "tokenize_prompt", tokenize)
    prompt_token_ids(
        "A street.",
        num_frames=1,
        fps=30.0,
        tokenizer_path="unused",
        hint="seg",
        emphasize_control_in_prompt=True,
    )
    assert recorded[0][0].endswith("align with the seg signal at every frame.")
    assert recorded[0][1]["system_prompt"] == SYSTEM_PROMPT


@pytest.fixture(scope="module")
def tokenizer_path() -> str:
    """The release's ``text_tokenizer`` folder if named, else the Qwen3-VL tokenizer it matches.

    Only the Hugging Face fallback may skip, and only when it cannot be fetched
    (``transformers`` raises ``OSError`` offline or when the hub is unreachable).
    Any other error, or any error with ``COSMOS3_TRANSFER_TOKENIZER`` set, fails.
    """
    from cosmos_interactive._tokenizer import load_tokenizer

    named = os.environ.get("COSMOS3_TRANSFER_TOKENIZER")
    if named:
        load_tokenizer(named)
        return named
    path = "Qwen/Qwen3-VL-8B-Instruct"
    try:
        load_tokenizer(path)
    except OSError as exc:
        pytest.skip(f"{path} tokenizer cannot be fetched: {exc}")
    return path


def test_prompt_ids_match_cosmos_framework(tokenizer_path: str) -> None:
    """The 458 ids cosmos-framework packed for a 201-frame, 30 fps request."""
    reference = json.loads(FIXTURE.read_text())
    ids = prompt_token_ids(
        reference["caption"],
        num_frames=reference["num_frames"],
        fps=reference["fps"],
        tokenizer_path=tokenizer_path,
    ).tolist()
    assert len(ids) == len(reference["text_ids"]) == 458
    assert ids == reference["text_ids"]


def test_prompt_ids_end_with_the_generation_suffix_and_respect_max_tokens(
    tokenizer_path: str,
) -> None:
    reference = json.loads(FIXTURE.read_text())
    ids = prompt_token_ids(
        "A street.", num_frames=201, fps=30.0, tokenizer_path=tokenizer_path
    )
    assert ids.tolist()[-2:] == reference["text_ids"][-2:]
    with pytest.raises(ValueError, match="Shorten the caption"):
        prompt_token_ids(
            "A street.",
            num_frames=201,
            fps=30.0,
            tokenizer_path=tokenizer_path,
            max_tokens=ids.numel() - 1,
        )
