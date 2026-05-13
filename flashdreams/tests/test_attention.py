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

"""
Unit test for attention implementations.

Multi-GPU test can be run with:
    PYTHONPATH=. torchrun --nproc_per_node=2 -m pytest tests/test_attention.py
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from flashdreams.core.attention import NativeAttention, RingAttention


@pytest.mark.ci_gpu
def test_attention():
    # Run multi gpu test with
    # PYTHONPATH=. torchrun --nproc_per_node=2 -m pytest ...
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    assert torch.cuda.is_available()
    torch.cuda.set_device(f"cuda:{rank}")
    torch.cuda.manual_seed(0)

    if world_size > 1:
        assert dist.is_nccl_available()
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        device_mesh = init_device_mesh(
            device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("cp",)
        )
        rank = torch.distributed.get_rank()
        cp_group = device_mesh.get_group(mesh_dim=0)

    batch = 2
    nheads = 4
    qkv_len = 64  # full sequence length
    dim = 128
    dtype = torch.bfloat16

    attn_op1 = NativeAttention(qkv_format="bshd", backend="cudnn")
    attn_op2 = RingAttention(qkv_format="bshd", backend="cudnn")
    if world_size > 1:
        attn_op1.set_context_parallel_group(cp_group)
        assert (
            attn_op1.is_context_parallel_enabled()
            and attn_op1.context_parallel_size() == world_size
        )
        attn_op2.set_context_parallel_group(cp_group)
        assert (
            attn_op2.is_context_parallel_enabled()
            and attn_op2.context_parallel_size() == world_size
        )

    q = torch.randn((batch, qkv_len, nheads, dim), dtype=dtype, device="cuda")
    k = torch.randn((batch, qkv_len, nheads, dim), dtype=dtype, device="cuda")
    v = torch.randn((batch, qkv_len, nheads, dim), dtype=dtype, device="cuda")
    if world_size > 1:
        q = (
            distribute_tensor(q, device_mesh, [Shard(1)], src_data_rank=None)
            .to_local()
            .contiguous()
            .clone()
        )
        k = (
            distribute_tensor(k, device_mesh, [Shard(1)], src_data_rank=None)
            .to_local()
            .contiguous()
            .clone()
        )
        v = (
            distribute_tensor(v, device_mesh, [Shard(1)], src_data_rank=None)
            .to_local()
            .contiguous()
            .clone()
        )
        assert q.shape == (batch, qkv_len // world_size, nheads, dim)
        assert k.shape == (batch, qkv_len // world_size, nheads, dim)
        assert v.shape == (batch, qkv_len // world_size, nheads, dim)

    o1 = attn_op1(q, k, v)
    o2 = attn_op2(q, k, v)
    torch.testing.assert_close(o1, o2, atol=1e-4, rtol=1e-4)

    if world_size > 1:
        dist.destroy_process_group()
