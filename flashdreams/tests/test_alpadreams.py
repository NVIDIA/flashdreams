# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from flashdreams.recipes.alpadreams.config import (
    build_sv_2steps_chunk2_loc6_lightvae_lighttae,
)


def test_alpadreams_streaming_inference():
    num_views = 1
    # Must match the alpadreams checkpoint training resolution
    height = 704
    width = 1280

    device = torch.device("cuda")
    dtype = torch.bfloat16

    image = torch.randn(1, num_views, 1, 3, height, width, device=device, dtype=dtype)
    text = [["Hello, world!"] * num_views]

    config = build_sv_2steps_chunk2_loc6_lightvae_lighttae()
    pipeline = config.setup().to(device)
    cache = pipeline.initialize_cache(text=text, image=image)  # ty:ignore[unknown-argument]

    autoregressive_index = 0
    num_frames = pipeline.get_num_frames(autoregressive_index)  # ty:ignore[call-non-callable]
    hdmap = torch.randn(
        1, num_views, num_frames, 3, height, width, device=device, dtype=dtype
    )
    decoded_video = pipeline.generate(autoregressive_index, hdmap=hdmap, cache=cache)  # ty:ignore[unknown-argument]
    pipeline.finalize(autoregressive_index, cache=cache)
    assert decoded_video.shape == hdmap.shape

    autoregressive_index = 1
    num_frames = pipeline.get_num_frames(autoregressive_index)  # ty:ignore[call-non-callable]
    hdmap = torch.randn(
        1, num_views, num_frames, 3, height, width, device=device, dtype=dtype
    )
    decoded_video = pipeline.generate(autoregressive_index, hdmap=hdmap, cache=cache)  # ty:ignore[unknown-argument]
    pipeline.finalize(autoregressive_index, cache=cache)
    assert decoded_video.shape == hdmap.shape


if __name__ == "__main__":
    test_alpadreams_streaming_inference()
