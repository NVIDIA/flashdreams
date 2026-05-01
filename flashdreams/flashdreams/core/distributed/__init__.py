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

"""Distributed-training initialization helpers."""

import ctypes
import math
import os
from datetime import timedelta

import pynvml
import torch
import torch.distributed as dist
from loguru import logger


class Device:
    """Lightweight wrapper around an NVML device handle for CPU-affinity queries."""

    _nvml_affinity_elements = math.ceil((os.cpu_count() or 1) / 64)

    def __init__(self, device_idx: int):
        """Bind to the NVML handle for the GPU at ``device_idx``."""
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self) -> str:
        """Return the marketing name reported by NVML for this device."""
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_cpu_affinity(self) -> list[int]:
        """Return the indices of CPUs ideally affined to this GPU per NVML."""
        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinity(
            self.handle, Device._nvml_affinity_elements
        ):
            # NVML returns a sequence of 64-bit affinity bitmasks, low word first.
            affinity_string = "{:064b}".format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in the 0th element of the list
        return [i for i, e in enumerate(affinity_list) if e != 0]


def init() -> int | None:
    """Initialize distributed training."""
    if dist.is_initialized():
        return torch.cuda.current_device()

    pynvml.nvmlInit()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    try:
        device = Device(local_rank)
        os.sched_setaffinity(0, device.get_cpu_affinity())
    except (pynvml.NVMLError, OSError) as e:
        logger.warning(f"Failed to set device affinity: {e}")

    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if dist.is_available():
        torch.cuda.set_device(local_rank)
        timeout_seconds = os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800)
        timeout_timedelta = timedelta(seconds=int(timeout_seconds))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timeout_timedelta,
            device_id=local_rank,
        )
        logger.critical(
            f"Initialized distributed training with local rank {local_rank} with timeout {timeout_seconds}",
        )

    # Bump cudaLimitMaxL2FetchGranularity (id=0x05) to 128 bytes for better bandwidth.
    _libcudart = ctypes.CDLL("libcudart.so")
    p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
    logger.info(f"Training with {dist.get_world_size()} GPUs.")
