# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes for metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th


class BaseMetric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self._values: List[float] = []

    @abstractmethod
    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor],
    ) -> float:
        pass

    def compute_batch_list(
        self,
        preds: List,
        targets: Optional[List] = None,
    ) -> List[Any]:
        if targets is None:
            targets = [None] * len(preds)
        if len(preds) != len(targets):
            raise ValueError(
                f"preds ({len(preds)}) and targets ({len(targets)}) length mismatch"
            )

        has_compute_batch = hasattr(self, "compute_batch")
        if has_compute_batch and len(preds) > 0:
            shapes = {getattr(p, "shape", None) for p in preds}
            shapes.discard(None)
            has_targets = any(t is not None for t in targets)
            uniform = len(shapes) == 1
            target_ok = not has_targets or all(
                t is not None and getattr(t, "shape", None) == next(iter(shapes))
                for t in targets
            )
            if uniform and target_ok:
                pred_batch = np.stack(preds, axis=0)
                if has_targets:
                    target_batch = np.stack(targets, axis=0)
                    return list(self.compute_batch(pred_batch, target_batch))
                return list(self.compute_batch(pred_batch))

        return [
            (self.compute(p, t) if t is not None else self.compute(p))
            for p, t in zip(preds, targets)
        ]

    def update(
        self, pred: Union[np.ndarray, th.Tensor], target: Union[np.ndarray, th.Tensor]
    ) -> float:
        value = self.compute(pred, target)
        self._values.append(value)
        return value

    def reset(self):
        self._values = []

    def get_mean(self) -> float:
        if not self._values:
            return 0.0
        return float(np.mean(self._values))

    def get_std(self) -> float:
        if not self._values:
            return 0.0
        return float(np.std(self._values))

    def get_summary(self) -> Dict[str, float]:
        return {
            f"{self.name}_mean": self.get_mean(),
            f"{self.name}_std": self.get_std(),
            f"{self.name}_min": float(np.min(self._values)) if self._values else 0.0,
            f"{self.name}_max": float(np.max(self._values)) if self._values else 0.0,
            f"{self.name}_count": len(self._values),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, device={self.device})"


class MetricRegistry:
    """Registry for managing multiple metrics with singleton caching."""

    _metrics: Dict[str, type] = {}
    _instances: Dict[tuple, BaseMetric] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(metric_class: type):
            cls._metrics[name.lower()] = metric_class
            return metric_class

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseMetric:
        name_lower = name.lower()
        if name_lower not in cls._metrics:
            available = list(cls._metrics.keys())
            raise ValueError(f"Unknown metric: {name}. Available metrics: {available}")

        cache_key = (name_lower, tuple(sorted(kwargs.items())))
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls._metrics[name_lower](**kwargs)
        return cls._instances[cache_key]

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._metrics.keys())

    @classmethod
    def create_multiple(cls, names: List[str], **kwargs) -> Dict[str, BaseMetric]:
        return {name: cls.get(name, **kwargs) for name in names}
