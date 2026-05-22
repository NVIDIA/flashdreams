# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video quality metrics: DOVER."""

from typing import List, Union

import numpy as np
import torch

from .base import BaseMetric, MetricRegistry

try:
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
    from dover.models import DOVER as DOVERModel

    DOVER_AVAILABLE = True
except ImportError:
    DOVERModel = None
    UnifiedFrameSampler = None
    spatial_temporal_view_decomposition = None
    DOVER_AVAILABLE = False


def _fuse_dover_results(results: list) -> dict:
    """Fuse aesthetic and technical DOVER scores into overall (from DOVER's evaluate_a_set_of_videos.py)."""
    t = (results[1] - 0.1107) / 0.07355
    a = (results[0] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": float(1 / (1 + np.exp(-a))),
        "technical": float(1 / (1 + np.exp(-t))),
        "overall": float(1 / (1 + np.exp(-x))),
    }


@MetricRegistry.register("dover")
class DOVER(BaseMetric):
    """DOVER video quality metric (aesthetic + technical + overall).

    Requires the `dover` package and a config/weights path.
    Overall score is in [0, 1] — higher is better.

    Usage::

        metric = DOVER(device="cuda", config_path="/path/to/dover.yml")
        score = metric.compute("/path/to/video.mp4")
    """

    def __init__(self, device: str = "cuda", config_path: str = None):
        super().__init__(name="DOVER", device=device)

        if not DOVER_AVAILABLE:
            raise ImportError(
                "DOVER is not installed. Install it with: pip install dover\n"
                "See https://github.com/VQAssessment/DOVER"
            )

        import yaml

        if config_path is None:
            raise ValueError(
                "DOVER requires --dover_config pointing to a dover.yml file. "
                "Download weights and config from https://github.com/VQAssessment/DOVER"
            )

        with open(config_path) as f:
            self.opt = yaml.safe_load(f)

        self.evaluator = DOVERModel(**self.opt["model"]["args"]).to(device)
        self.evaluator.load_state_dict(
            torch.load(self.opt["test_load_path"], map_location=device)
        )
        self.evaluator.eval()

        self.sample_types = self.opt["data"]["val-l1080p"]["args"]["sample_types"]
        self.samplers = {}
        for stype, sopt in self.sample_types.items():
            if "t_frag" not in sopt:
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

        self._aesthetic_scores: List[float] = []
        self._technical_scores: List[float] = []
        self._overall_scores: List[float] = []

    def compute_from_path(self, video_path: str) -> dict:
        data, _ = spatial_temporal_view_decomposition(
            video_path, self.sample_types, self.samplers, is_train=False
        )
        for k, v in data.items():
            data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(
                3, 0, 1, 2
            )

        video = {}
        for key in ["aesthetic", "technical"]:
            if key in data:
                v = data[key].to(self.device)
                c, t, h, w = v.shape
                v = v.unsqueeze(0)
                num_clips = self.sample_types[key]["num_clips"]
                video[key] = (
                    v.reshape(1, c, num_clips, t // num_clips, h, w)
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(num_clips, c, t // num_clips, h, w)
                )

        with torch.no_grad():
            results = self.evaluator(video, reduce_scores=False)
            results = [np.mean(r.cpu().numpy()) for r in results]

        return _fuse_dover_results(results)

    def compute(self, pred: Union[str, np.ndarray, torch.Tensor], target=None) -> float:
        if not isinstance(pred, str):
            raise NotImplementedError(
                "DOVER requires a video file path, not raw frames."
            )
        return self.compute_from_path(pred)["overall"]

    def update(self, pred: Union[str, np.ndarray, torch.Tensor], target=None) -> float:
        if not isinstance(pred, str):
            raise NotImplementedError(
                "DOVER requires a video file path, not raw frames."
            )
        scores = self.compute_from_path(pred)
        self._aesthetic_scores.append(scores["aesthetic"])
        self._technical_scores.append(scores["technical"])
        self._overall_scores.append(scores["overall"])
        self._values.append(scores["overall"])
        return scores["overall"]

    def get_detailed_scores(self) -> dict:
        return {
            "aesthetic": self._aesthetic_scores.copy(),
            "technical": self._technical_scores.copy(),
            "overall": self._overall_scores.copy(),
        }

    def get_summary(self) -> dict:
        summary = super().get_summary()
        if self._aesthetic_scores:
            summary["DOVER_aesthetic_mean"] = float(np.mean(self._aesthetic_scores))
            summary["DOVER_aesthetic_std"] = float(np.std(self._aesthetic_scores))
            summary["DOVER_technical_mean"] = float(np.mean(self._technical_scores))
            summary["DOVER_technical_std"] = float(np.std(self._technical_scores))
        return summary

    def reset(self):
        super().reset()
        self._aesthetic_scores = []
        self._technical_scores = []
        self._overall_scores = []
