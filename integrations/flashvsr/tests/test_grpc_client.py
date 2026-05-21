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

from __future__ import annotations

import numpy as np
import pytest
from flashvsr.grpc import uplift_server as grpc_server
from flashvsr.grpc.protos import flashvsr_pb2 as pb2
from flashvsr.grpc.uplift_client import build_chunk_request, build_chunks

pytestmark = pytest.mark.ci_cpu


def test_build_chunks_drops_trailing_partial_chunk() -> None:
    assert build_chunks(total_frames=45, first_chunk=13, chunk_size=16) == [
        (0, 13),
        (13, 16),
        (29, 16),
    ]


def test_build_chunk_request_raw_display_only() -> None:
    frames = np.zeros((8, 4, 6, 3), dtype=np.uint8)
    request = build_chunk_request(
        chunk_idx=0,
        frame_data=frames,
        scale=2,
        sparse_ratio=0.0,
        input_format="raw",
        jpeg_quality=90,
        display_only=True,
    )

    assert request.frame_encoding == pb2.FRAME_ENCODING_RAW_RGB
    assert request.frames_rgb == frames.tobytes()
    assert request.num_frames == 8
    assert request.height == 4
    assert request.width == 6
    assert request.input_height == 4
    assert request.input_width == 6
    assert request.scale == 2
    assert request.display_only


def test_attention_mode_auto_falls_back_to_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grpc_server, "_block_sparse_attn_available", lambda: False)

    assert grpc_server._resolve_attention_mode("auto") == "full"


def test_attention_mode_sparse_requires_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grpc_server, "_block_sparse_attn_available", lambda: False)

    with pytest.raises(RuntimeError, match="block_sparse_attn"):
        grpc_server._resolve_attention_mode("sparse")
