# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engine-only pipeline exposes semantic pixels without a model."""

import pytest
import torch
from crazy_robotaxi.debug_engine import SemanticDebugPipeline

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("index,frames", [(0, 5), (1, 8), (8, 8)])
def test_semantic_pipeline_preserves_chunk_pixels_and_cadence(index, frames):
    pipeline = SemanticDebugPipeline("cpu")
    cache = pipeline.initialize_cache(
        image=object(), text=object(), view_names=object()
    )
    image = torch.linspace(-1, 1, frames * 3 * 4 * 6).reshape(1, 1, frames, 3, 4, 6)
    assert pipeline.get_num_output_frames(index) == frames
    output = pipeline.generate(autoregressive_index=index, cache=cache, input=image)
    assert output is image
    assert pipeline.finalize(autoregressive_index=index, cache=cache) is None
    pipeline.close()
    pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.generate(autoregressive_index=index, cache=cache, input=image)


@pytest.mark.parametrize(
    "shape", [(1, 1, 5, 3, 4, 6), (1, 2, 8, 3, 4, 6), (8, 3, 4, 6)]
)
def test_semantic_pipeline_rejects_misaligned_video(shape):
    pipeline = SemanticDebugPipeline("cpu")
    with pytest.raises(ValueError, match="BVTCHW"):
        pipeline.generate(autoregressive_index=1, cache=None, input=torch.zeros(shape))
