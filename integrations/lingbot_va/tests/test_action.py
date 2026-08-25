# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Robotwin action tensor boundary."""

import pytest
import torch

from lingbot_va.action import (
    LingbotVAActionProcessor,
    LingbotVAActionProcessorConfig,
)

pytestmark = pytest.mark.ci_cpu


def test_action_mask_selects_only_robotwin_channels() -> None:
    processor = LingbotVAActionProcessor()

    mask = processor.action_mask()

    assert mask.dtype == torch.bool
    assert mask.shape == (processor.config.action_dim,)
    assert mask.sum().item() == len(processor.config.used_action_channel_ids)


def test_action_preprocess_postprocess_round_trip() -> None:
    processor = LingbotVAActionProcessor()
    channel_count = len(processor.config.used_action_channel_ids)
    raw = torch.linspace(-0.25, 0.25, channel_count * 2 * 3).reshape(
        channel_count,
        2,
        3,
    )

    model_action = processor.preprocess(raw)
    restored = processor.postprocess(model_action)

    assert model_action.shape == (1, processor.config.action_dim, 2, 3, 1)
    assert restored.shape == (6, channel_count)
    torch.testing.assert_close(restored, raw.permute(1, 2, 0).reshape(6, channel_count))


def test_action_processor_rejects_inconsistent_schema() -> None:
    with pytest.raises(ValueError, match="q01 and q99"):
        LingbotVAActionProcessorConfig(action_dim=2, q01=(0.0,), q99=(1.0, 1.0))

    with pytest.raises(ValueError, match="unique"):
        LingbotVAActionProcessorConfig(used_action_channel_ids=(0, 0))


def test_action_processor_rejects_wrong_input_shape() -> None:
    processor = LingbotVAActionProcessor()

    with pytest.raises(ValueError, match="expected"):
        processor.preprocess(torch.zeros(15, 2, 3))

    with pytest.raises(ValueError, match="expected"):
        processor.postprocess(torch.zeros(2, 30, 2, 3, 1))
