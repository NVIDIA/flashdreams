# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for synchronized data-ward flow sampling."""

import pytest
import torch

from flashdreams.infra.diffusion.scheduler import (
    DataFlowEulerSchedulerConfig,
    sample_synchronized,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_joint_matches_scalar_reference(dtype):
    samples = (
        torch.linspace(-1, 1, 12).reshape(3, 4).to(dtype),
        torch.zeros(2, 3, dtype=dtype),
    )
    schedulers = tuple(
        DataFlowEulerSchedulerConfig(shift=shift).setup() for shift in (12, 3)
    )
    calls = []

    def predict(values, times):
        calls.append(times)
        return tuple(value.float() * 0.1 + 0.2 for value in values)

    result = sample_synchronized(samples, schedulers, predict)
    assert len(calls) == 29
    for initial, scheduler, actual in zip(samples, schedulers, result, strict=True):
        expected = initial
        for index, timestep in enumerate(scheduler.timesteps):
            flow = expected.float() * 0.1 + 0.2
            denoised = expected + (1 - timestep.to(dtype)) * flow
            ratio = scheduler.sigmas[index + 1] / scheduler.sigmas[index]
            expected = (ratio * expected.float() + (1 - ratio) * denoised.float()).to(
                dtype
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        scalar = scheduler.sample(
            initial, lambda value, time: value.float() * 0.1 + 0.2
        )
        torch.testing.assert_close(actual, scalar, rtol=0, atol=0)


def test_mismatched_schedules_fail_before_prediction():
    schedulers = tuple(
        DataFlowEulerSchedulerConfig(num_inference_steps=n).setup() for n in (3, 4)
    )
    with pytest.raises(ValueError, match="equal lengths"):
        sample_synchronized(
            (torch.zeros(1), torch.zeros(1)),
            schedulers,
            lambda *_: pytest.fail("must not predict"),
        )


def test_reject_invalid_prediction_shape():
    scheduler = DataFlowEulerSchedulerConfig(num_inference_steps=2).setup()
    with pytest.raises(ValueError, match="shapes"):
        sample_synchronized(
            (torch.zeros(2),), (scheduler,), lambda *_: (torch.zeros(3),)
        )


def test_grid_survives_dtype_conversion():
    scheduler = DataFlowEulerSchedulerConfig().setup()
    expected = scheduler.sigmas.clone()
    scheduler.to(torch.bfloat16)
    assert scheduler.timesteps.dtype == torch.float32
    torch.testing.assert_close(scheduler.sigmas, expected, rtol=0, atol=0)
