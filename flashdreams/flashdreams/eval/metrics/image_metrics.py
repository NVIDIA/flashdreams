# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image quality metrics: PSNR, SSIM, LPIPS, NIQE, MUSIQ, CLIPIQA."""

from typing import List, Union

import numpy as np
import torch as th

from .base import BaseMetric, MetricRegistry

try:
    import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


# Default mini-batch size for pyiqa-backed metrics (LPIPS, NIQE, MUSIQ, CLIPIQA)
# when the caller does not pass an explicit ``--batch_size``.
#
# Why a small constant instead of "all frames at once":
#   MUSIQ in particular is multi-scale and patch-based — it does NOT resize
#   the input to a fixed size before its first conv, so peak activation
#   memory scales linearly with (frames_in_batch × H × W). On a 2× upsampled
#   1440×2560 clip with ~50 frames, a single ``model(t)`` call into
#   pyiqa.MUSIQ allocated ~36 GiB inside the initial ``conv_root`` and
#   OOM'd an 80 GiB H100 (the rest of VRAM was held by CLIPIQA's CLIP
#   weights from the previous metric in the same process).
#
# Picking 4 keeps MUSIQ comfortably under ~10 GiB on a 2× alpadreams clip
# while still amortising the per-call kernel-launch / D2H sync overhead
# across multiple frames; NIQE / CLIPIQA / LPIPS are far cheaper per
# frame and remain bottlenecked by GPU compute, not by this batch size.
# Override with ``--batch_size N`` if you have a smaller GPU or a much
# larger input resolution.
DEFAULT_PYIQA_BATCH_SIZE = 4


def _batch_to_tensor(
    frames: Union[np.ndarray, th.Tensor], device: str, to_chw: bool = False
) -> th.Tensor:
    """Convert batch of frames (T, H, W, C) or (T, C, H, W) to GPU tensor."""
    if isinstance(frames, th.Tensor):
        if str(frames.device) == device:
            if to_chw and frames.ndim == 4 and frames.shape[3] in [1, 3]:
                frames = frames.permute(0, 3, 1, 2)
            return frames
        return frames.to(device)

    if to_chw and frames.ndim == 4 and frames.shape[3] in [1, 3]:
        frames = np.transpose(frames, (0, 3, 1, 2))
    return th.from_numpy(frames.copy()).float().to(device)


def _psnr_torch(pred: th.Tensor, target: th.Tensor, data_range: float) -> float:
    """Pure-torch PSNR — no torchmetrics required."""
    mse = th.mean((pred.float() - target.float()) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * th.log10(th.tensor(data_range**2) / mse).item())


@MetricRegistry.register("psnr")
class PSNR(BaseMetric):
    """Peak Signal-to-Noise Ratio. Higher is better (dB).

    Uses torchmetrics when available; falls back to a pure-torch implementation.
    """

    def __init__(self, device: str = "cuda", data_range: float = 255.0):
        super().__init__(name="PSNR", device=device)
        self.data_range = data_range
        self._metric = None

    def _load_model(self):
        if self._metric is None:
            try:
                from torchmetrics.image import PeakSignalNoiseRatio

                self._metric = PeakSignalNoiseRatio(data_range=self.data_range).to(
                    self.device
                )
            except ImportError:
                self._metric = "fallback"

    def _compute_one(self, pred_th: th.Tensor, target_th: th.Tensor) -> float:
        if self._metric == "fallback":
            return _psnr_torch(pred_th, target_th, self.data_range)
        return float(self._metric(pred_th, target_th).item())

    def _prep(self, x: Union[np.ndarray, th.Tensor]) -> th.Tensor:
        t = (
            th.from_numpy(x).float().to(self.device)
            if isinstance(x, np.ndarray)
            else x.float().to(self.device)
        )
        if t.ndim == 3 and t.shape[2] in [1, 3]:
            t = t.permute(2, 0, 1)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return t

    def compute(
        self, pred: Union[np.ndarray, th.Tensor], target: Union[np.ndarray, th.Tensor]
    ) -> float:
        self._load_model()
        with th.no_grad():
            return self._compute_one(self._prep(pred), self._prep(target))

    def compute_batch(
        self, pred: np.ndarray, target: np.ndarray, batch_size: int = None
    ) -> List[float]:
        self._load_model()
        num_frames = pred.shape[0]
        if batch_size is None or batch_size >= num_frames:
            pred_th = _batch_to_tensor(pred, self.device, to_chw=True)
            target_th = _batch_to_tensor(target, self.device, to_chw=True)
            with th.no_grad():
                return [
                    self._compute_one(pred_th[i : i + 1], target_th[i : i + 1])
                    for i in range(num_frames)
                ]

        all_psnr = []
        for i in range(0, num_frames, batch_size):
            end = min(i + batch_size, num_frames)
            pred_th = _batch_to_tensor(pred[i:end], self.device, to_chw=True)
            target_th = _batch_to_tensor(target[i:end], self.device, to_chw=True)
            with th.no_grad():
                all_psnr.extend(
                    [
                        self._compute_one(pred_th[j : j + 1], target_th[j : j + 1])
                        for j in range(end - i)
                    ]
                )
        return all_psnr


@MetricRegistry.register("ssim")
class SSIM(BaseMetric):
    """Structural Similarity Index using fused-ssim. Higher is better (0–1)."""

    def __init__(self, device: str = "cuda", data_range: float = 255.0):
        super().__init__(name="SSIM", device=device)
        self.data_range = data_range
        if not FUSED_SSIM_AVAILABLE:
            raise ImportError(
                "fused-ssim is not available. Install it with: "
                "pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation"
            )

    def _to_01(self, t: th.Tensor) -> th.Tensor:
        if t.dtype == th.uint8 or t.max() > 1.0:
            return t / self.data_range
        return t

    def compute(
        self, pred: Union[np.ndarray, th.Tensor], target: Union[np.ndarray, th.Tensor]
    ) -> float:
        def prep(x):
            if isinstance(x, np.ndarray):
                if x.ndim == 3 and x.shape[2] in [1, 3]:
                    x = np.transpose(x, (2, 0, 1))
                return th.from_numpy(x).float().unsqueeze(0).to(self.device)
            x = x.float()
            if x.ndim == 3:
                x = x.unsqueeze(0)
            return x.to(self.device)

        pred_th = self._to_01(prep(pred))
        target_th = self._to_01(prep(target))
        with th.no_grad():
            return float(
                fused_ssim.fused_ssim(
                    pred_th, target_th, padding="same", train=False
                ).item()
            )

    def compute_batch(
        self, pred: np.ndarray, target: np.ndarray, batch_size: int = None
    ) -> List[float]:
        num_frames = pred.shape[0]
        if batch_size is None or batch_size >= num_frames:
            pred_th = self._to_01(_batch_to_tensor(pred, self.device, to_chw=True))
            target_th = self._to_01(_batch_to_tensor(target, self.device, to_chw=True))
            with th.no_grad():
                return [
                    float(
                        fused_ssim.fused_ssim(
                            pred_th[i : i + 1],
                            target_th[i : i + 1],
                            padding="same",
                            train=False,
                        ).item()
                    )
                    for i in range(num_frames)
                ]

        all_ssim = []
        for i in range(0, num_frames, batch_size):
            end = min(i + batch_size, num_frames)
            pred_th = self._to_01(
                _batch_to_tensor(pred[i:end], self.device, to_chw=True)
            )
            target_th = self._to_01(
                _batch_to_tensor(target[i:end], self.device, to_chw=True)
            )
            with th.no_grad():
                all_ssim.extend(
                    [
                        float(
                            fused_ssim.fused_ssim(
                                pred_th[j : j + 1],
                                target_th[j : j + 1],
                                padding="same",
                                train=False,
                            ).item()
                        )
                        for j in range(end - i)
                    ]
                )
        return all_ssim


@MetricRegistry.register("lpips")
class LPIPS(BaseMetric):
    """Learned Perceptual Image Patch Similarity. Lower is better (0 = identical).

    Default backbone VGG is more sensitive than AlexNet; use --lpips_net to override.
    """

    def __init__(self, device: str = "cuda", net: str = "vgg"):
        super().__init__(name="LPIPS", device=device)
        self.net = net
        self._model = None

    def _load_model(self):
        if self._model is None:
            import pyiqa

            metric_name = f"lpips-{self.net}" if self.net != "alex" else "lpips"
            self._model = pyiqa.create_metric(
                metric_name, device=th.device(self.device)
            )

    def _prep(self, x: Union[np.ndarray, th.Tensor]) -> th.Tensor:
        if isinstance(x, np.ndarray):
            t = th.from_numpy(x).float().to(self.device)
        else:
            t = x.float().to(self.device)
        if t.max() > 1.0:
            t = t / 255.0
        if t.ndim == 3:
            if t.shape[2] in [1, 3]:
                t = t.permute(2, 0, 1)
            t = t.unsqueeze(0)
        return t

    def compute(
        self, pred: Union[np.ndarray, th.Tensor], target: Union[np.ndarray, th.Tensor]
    ) -> float:
        self._load_model()
        with th.no_grad():
            return float(self._model(self._prep(pred), self._prep(target)).item())

    def compute_batch(
        self, pred: np.ndarray, target: np.ndarray, batch_size: int = None
    ) -> List[float]:
        self._load_model()
        num_frames = pred.shape[0]
        # Same memory-safety cap as the no-reference pyiqa metrics — see
        # ``DEFAULT_PYIQA_BATCH_SIZE`` for the full rationale. LPIPS' VGG /
        # AlexNet backbone keeps full-resolution feature maps for the
        # perceptual distance, so a 50-frame 1440×2560 batch would balloon
        # past 80 GiB on the same GPU that already lost MUSIQ to OOM.
        if batch_size is None:
            batch_size = min(DEFAULT_PYIQA_BATCH_SIZE, num_frames)

        lpips_values = []
        for i in range(0, num_frames, batch_size):
            end = min(i + batch_size, num_frames)
            pred_th = _batch_to_tensor(pred[i:end], self.device, to_chw=True)
            target_th = _batch_to_tensor(target[i:end], self.device, to_chw=True)
            if pred_th.max() > 1.0:
                pred_th = pred_th / 255.0
            if target_th.max() > 1.0:
                target_th = target_th / 255.0
            with th.no_grad():
                out = self._model(pred_th, target_th)
                if out.numel() == 1:
                    lpips_values.append(float(out.item()))
                else:
                    lpips_values.extend(
                        [float(x) for x in out.flatten().cpu().tolist()]
                    )
        return lpips_values


def _make_pyiqa_no_ref(metric_name: str, device: str) -> object:
    import pyiqa

    return pyiqa.create_metric(metric_name, device=th.device(device))


def _pyiqa_compute(model, pred: Union[np.ndarray, th.Tensor], device: str) -> float:
    if isinstance(pred, np.ndarray):
        t = th.from_numpy(pred).float().to(device)
    else:
        t = pred.float().to(device)
    if t.max() > 1.0:
        t = t / 255.0
    if t.ndim == 3:
        if t.shape[2] in [1, 3]:
            t = t.permute(2, 0, 1)
        t = t.unsqueeze(0)
    with th.no_grad():
        return float(model(t).item())


def _pyiqa_compute_batch(
    model, pred: np.ndarray, device: str, batch_size: int
) -> List[float]:
    num_frames = pred.shape[0]
    # See ``DEFAULT_PYIQA_BATCH_SIZE`` for why we cap by default — passing
    # an explicit ``--batch_size`` from the CLI continues to win, including
    # very large values for callers who know their inputs fit.
    if batch_size is None:
        batch_size = min(DEFAULT_PYIQA_BATCH_SIZE, num_frames)
    values = []
    for i in range(0, num_frames, batch_size):
        end = min(i + batch_size, num_frames)
        t = _batch_to_tensor(pred[i:end], device, to_chw=True)
        if t.max() > 1.0:
            t = t / 255.0
        with th.no_grad():
            out = model(t)
            if out.numel() == 1:
                values.append(float(out.item()))
            else:
                values.extend([float(x) for x in out.flatten().cpu().tolist()])
    return values


@MetricRegistry.register("niqe")
class NIQE(BaseMetric):
    """Natural Image Quality Evaluator. Lower is better (no reference)."""

    def __init__(self, device: str = "cuda"):
        super().__init__(name="NIQE", device=device)
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = _make_pyiqa_no_ref("niqe", self.device)

    def compute(self, pred: Union[np.ndarray, th.Tensor], target=None) -> float:
        self._load_model()
        return _pyiqa_compute(self._model, pred, self.device)

    def compute_batch(
        self, pred: np.ndarray, target=None, batch_size: int = None
    ) -> List[float]:
        self._load_model()
        return _pyiqa_compute_batch(self._model, pred, self.device, batch_size)


@MetricRegistry.register("musiq")
class MUSIQ(BaseMetric):
    """Multi-Scale Image Quality Transformer. Higher is better (0–100, no reference).

    Models: musiq (default), musiq-ava, musiq-paq2piq, musiq-spaq.
    """

    def __init__(self, device: str = "cuda", model: str = "musiq"):
        super().__init__(name="MUSIQ", device=device)
        self.model_name = model
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = _make_pyiqa_no_ref(self.model_name, self.device)

    def compute(self, pred: Union[np.ndarray, th.Tensor], target=None) -> float:
        self._load_model()
        return _pyiqa_compute(self._model, pred, self.device)

    def compute_batch(
        self, pred: np.ndarray, target=None, batch_size: int = None
    ) -> List[float]:
        self._load_model()
        return _pyiqa_compute_batch(self._model, pred, self.device, batch_size)


@MetricRegistry.register("clipiqa")
class CLIPIQA(BaseMetric):
    """CLIP-based Image Quality Assessment. Higher is better (0–1, no reference).

    Models: clipiqa (default, zero-shot), clipiqa+, clipiqa+_vitL14_512, clipiqa+_rn50_512.
    Zero-shot clipiqa is more OOD-robust to generated/diffusion imagery.
    """

    def __init__(self, device: str = "cuda", model: str = "clipiqa"):
        super().__init__(name="CLIPIQA", device=device)
        self.model_name = model
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = _make_pyiqa_no_ref(self.model_name, self.device)

    def compute(self, pred: Union[np.ndarray, th.Tensor], target=None) -> float:
        self._load_model()
        return _pyiqa_compute(self._model, pred, self.device)

    def compute_batch(
        self, pred: np.ndarray, target=None, batch_size: int = None
    ) -> List[float]:
        self._load_model()
        return _pyiqa_compute_batch(self._model, pred, self.device, batch_size)
