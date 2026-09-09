# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronized sampling of coupled diffusion streams."""

from collections.abc import Callable
from typing import Protocol

from torch import Tensor


class StepScheduler(Protocol):
    """Indexed deterministic schedule for one member of a joint prediction."""

    timesteps: Tensor
    """One-dimensional grid with one entry per model evaluation."""

    def step(self, sample: Tensor, flow: Tensor, index: int) -> Tensor:
        """Advance one sample using the prediction at ``index``."""
        ...


def sample_synchronized(
    initial_samples: tuple[Tensor, ...],
    schedulers: tuple[StepScheduler, ...],
    predict_flow: Callable[
        [tuple[Tensor, ...], tuple[Tensor, ...]], tuple[Tensor, ...]
    ],
) -> tuple[Tensor, ...]:
    """Advance coupled streams with exactly one joint prediction per step.

    Args:
        initial_samples: Generated samples, excluding immutable conditioning.
        schedulers: One schedule per sample, all with the same number of steps.
        predict_flow: Joint predictor returning one flow of each sample's shape.

    Returns:
        Final samples in their original order, shapes, devices, and dtypes.

    Raises:
        ValueError: Stream counts, schedule lengths, or tensor geometry disagree.
    """
    if not initial_samples or len(initial_samples) != len(schedulers):
        raise ValueError("Provide a nonempty sample tuple and one scheduler per sample")
    grids = tuple(scheduler.timesteps for scheduler in schedulers)
    if any(grid.ndim != 1 or grid.numel() == 0 for grid in grids):
        raise ValueError("Schedules must be nonempty one-dimensional timestep grids")
    if any(grid.numel() != grids[0].numel() for grid in grids):
        raise ValueError("Coupled schedules must have equal lengths")
    samples = initial_samples
    for index in range(grids[0].numel()):
        times = tuple(
            grid[index].to(sample.device)
            for grid, sample in zip(grids, samples, strict=True)
        )
        flows = predict_flow(samples, times)
        if not isinstance(flows, tuple) or len(flows) != len(samples):
            raise ValueError("Joint predictor must return one flow per sample")
        for sample, flow in zip(samples, flows, strict=True):
            if flow.shape != sample.shape or flow.device != sample.device:
                raise ValueError("Predicted flows must match sample shapes and devices")
            if not flow.is_floating_point():
                raise ValueError("Predicted flows must be floating-point tensors")
        advanced = tuple(
            scheduler.step(sample, flow, index)
            for scheduler, sample, flow in zip(schedulers, samples, flows, strict=True)
        )
        if any(
            new.shape != old.shape or new.device != old.device or new.dtype != old.dtype
            for new, old in zip(advanced, samples, strict=True)
        ):
            raise ValueError("Schedulers must preserve sample shape, device, and dtype")
        samples = advanced
    return samples
