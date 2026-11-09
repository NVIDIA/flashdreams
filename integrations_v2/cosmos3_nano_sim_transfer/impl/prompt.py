# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer prompt: caption, duration and resolution sentences, system prompt.

The text is built the way cosmos-framework formats a plain-text Transfer
prompt: the caption without surrounding whitespace and trailing periods, then
``. The video is X seconds long and is of N FPS.`` (left out for a one-frame
clip) and ``. This video is of HxW resolution.``, where ``N`` is the frame rate
rounded to a whole number and ``X`` is the frame count divided by ``N``. The
text goes under the Transfer system prompt through the Qwen3-VL chat template,
followed by the end-of-turn and ``<|vision_start|>`` tokens.

JSON-object captions instead receive structured duration, FPS and resolution
fields, preserving Framework's JSON serialization and key order.

The default leaves cosmos-framework's ``emphasize_control_in_prompt``
off. Setting it explicitly appends the Framework control-adherence sentence.
The release's
``checkpoint.json`` and cosmos-framework's Transfer defaults turn it on, while
the requests this package was checked against (and the release README's
server examples) run with it off.
"""

from __future__ import annotations

import json
from pathlib import Path

from cosmos_interactive._tokenizer import tokenize_prompt
from torch import Tensor

from ..config import CONTROL_HINTS

SYSTEM_PROMPT = (
    "You are a helpful assistant that generates images or videos following the user's "
    "instructions and control signals (edge maps, blur, depth, or segmentation)."
)
"""System prompt the release was trained with (``config.SYSTEM_PROMPT_ID``)."""

DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
"""Duration sentence appended to the caption."""

RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
"""Resolution sentence appended after the duration sentence."""

HEIGHT = 480
"""Output height in pixels."""

WIDTH = 832
"""Output width in pixels."""


def prompt_fps(fps: float) -> int:
    """Frame rate the prompt states: ``fps`` rounded to a whole number.

    Raises:
        ValueError: ``fps`` rounds to less than one.
    """
    rounded = int(round(fps))
    if rounded < 1:
        raise ValueError(f"fps must round to at least 1, got {fps}")
    return rounded


def _parse_json_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        return None


def transfer_caption(
    caption: str,
    *,
    num_frames: int,
    fps: float,
    hint: str | None = None,
    emphasize_control_in_prompt: bool = False,
    aspect_ratio: str | None = None,
) -> str:
    """The user text of a Transfer prompt: caption, duration and resolution sentences.

    Args:
        caption: Scene description as plain text or a JSON object string.
        num_frames: Pixel frames of the clip the prompt describes.
        fps: Frame rate of the clip; the prompt states it rounded.
        hint: Active control modality; required when emphasizing the control.
        emphasize_control_in_prompt: Append Framework's control-adherence
            sentence. The parity baseline leaves this off.
        aspect_ratio: Optional nominal bucket label for a structured caption,
            e.g. ``16,9``. ``None`` preserves any existing JSON field.

    Raises:
        ValueError: ``num_frames`` is below one, ``fps`` rounds to less than
            one, or emphasis is requested without a supported control hint.
    """
    if num_frames < 1:
        raise ValueError(f"need num_frames >= 1, got {num_frames}")
    rate = prompt_fps(fps)
    structured = _parse_json_object(caption)
    if structured is not None:
        # Framework _format_json_prompt_with_template: overwrite metadata in
        # insertion order and retain the default json.dumps separators/escaping.
        if num_frames > 1:
            structured.update(duration=f"{int(num_frames / rate)}s", fps=float(rate))
        else:
            structured.pop("duration", None)
            structured.pop("fps", None)
        structured["resolution"] = {"H": HEIGHT, "W": WIDTH}
        if aspect_ratio is not None:
            structured["aspect_ratio"] = aspect_ratio
        text = json.dumps(structured)
    else:
        text = caption.strip()
        if num_frames > 1:
            duration = DURATION_TEMPLATE.format(duration=num_frames / rate, fps=rate)
            text = text.rstrip(".") + ". " + duration
        resolution = RESOLUTION_TEMPLATE.format(height=HEIGHT, width=WIDTH)
        text = text.strip().rstrip(".") + ". " + resolution
    if emphasize_control_in_prompt:
        if hint not in CONTROL_HINTS:
            raise ValueError(f"Control emphasis needs one hint from {CONTROL_HINTS}")
        text += (
            f" Follow the {hint} control video precisely: shape, contour, silhouette,"
            f" position, and motion of every visible structure must align with the {hint}"
            " signal at every frame."
        )
    return text


def prompt_token_ids(
    caption: str,
    *,
    num_frames: int,
    fps: float,
    tokenizer_path: str | Path,
    max_tokens: int | None = None,
    hint: str | None = None,
    emphasize_control_in_prompt: bool = False,
    aspect_ratio: str | None = None,
) -> Tensor:
    """Token ids of the full Transfer prompt, ``[N]`` int64.

    Args:
        caption: Scene description as plain text or a JSON object string.
        num_frames: Pixel frames of the clip; the duration sentence states them.
        fps: Frame rate of the clip.
        tokenizer_path: The release's ``text_tokenizer`` folder (or any
            Qwen3-VL tokenizer source ``transformers`` can load).
        max_tokens: Reject prompts longer than this; no limit when ``None``.
        hint: Active control modality, as in :func:`transfer_caption`.
        emphasize_control_in_prompt: Append the control-adherence sentence.
        aspect_ratio: Structured-caption bucket label; see :func:`transfer_caption`.

    Raises:
        ValueError: The prompt is longer than ``max_tokens``, or see
            :func:`transfer_caption`.
    """
    ids = tokenize_prompt(
        transfer_caption(
            caption,
            num_frames=num_frames,
            fps=fps,
            hint=hint,
            emphasize_control_in_prompt=emphasize_control_in_prompt,
            aspect_ratio=aspect_ratio,
        ),
        height=HEIGHT,
        width=WIDTH,
        tokenizer_path=str(tokenizer_path),
        system_prompt=SYSTEM_PROMPT,
        append_resolution=False,
    )
    if max_tokens is not None and ids.numel() > max_tokens:
        raise ValueError(
            f"the prompt is {ids.numel()} tokens, more than {max_tokens}. "
            "Shorten the caption."
        )
    return ids
