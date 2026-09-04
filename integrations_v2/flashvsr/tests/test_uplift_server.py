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

"""CPU contract test for the standalone FlashVSR uplift server."""

from concurrent import futures

import pytest

grpc = pytest.importorskip("grpc")
uplift_server = pytest.importorskip("flashvsr.impl.uplift_server")

pytestmark = pytest.mark.ci_cpu


def test_uplift_protocol_and_service_registration() -> None:
    response = uplift_server.StartSessionResponse(
        session_id="test-session",
        success=True,
        session_token="test-token",
    )
    decoded = uplift_server.StartSessionResponse.FromString(
        response.SerializeToString()
    )
    assert decoded.session_token == "test-token"

    service = uplift_server.FlashVSR(
        default_height=64,
        default_width=64,
        default_scale=2,
        default_sparse_ratio=2.0,
        attention_mode="full",
        compile_network=False,
        use_cuda_graph=False,
        dtype=uplift_server.torch.bfloat16,
        device="cpu",
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    uplift_server._add_flash_vsr_servicer_to_server(service, server)
    server.stop(grace=None).wait()
