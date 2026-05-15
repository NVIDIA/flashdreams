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

"""Unit tests for checkpoint loading utilities."""

import os
import tempfile

import torch

from flashdreams.core.checkpoint.load import load_checkpoint

S3_PTH_PATH = "s3://flashdreams/assets/checkpoints/autoencoders/taew2_1.pth"


def test_load_checkpoint_from_s3() -> None:
    """Test loading .pth checkpoints from S3."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_dict = load_checkpoint(
            checkpoint_path=S3_PTH_PATH,
            local_cache_dir=tmp_dir,
            credential_path="credentials/s3_checkpoint.secret",
        )

        local_path = os.path.join(tmp_dir, S3_PTH_PATH.split("s3://")[-1])
        assert os.path.exists(local_path)
        assert os.path.getsize(local_path) > 0

        state_dict_from_local = torch.load(local_path)
        for k, v in state_dict.items():
            assert (v == state_dict_from_local[k]).all()
