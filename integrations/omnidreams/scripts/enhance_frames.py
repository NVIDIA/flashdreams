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

"""Learned sharpening/restoration pass for omnidreams output frames.

Runs a Real-ESRGAN checkpoint as a restoration filter: upscale x4 with the
net, then area-downscale back to 1x on GPU. This recovers plausible texture
on soft regions instead of merely amplifying existing edges like an
unsharp mask.

The SRVGGNetCompact and RRDBNet definitions mirror the public
Real-ESRGAN / BasicSR layouts so upstream checkpoints load directly
(no basicsr dependency).

Usage:
    python enhance_frames.py --input in.mp4 --output out.mp4
    python enhance_frames.py --input in.mp4 --output out.mp4 --model x4plus
    python enhance_frames.py --input in.mp4 --output out.mp4 --mode unsharp
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"


def cache_dir() -> Path:
    cache_root = os.environ.get(
        "FLASHDREAMS_CACHE_DIR", str(Path.home() / ".cache" / "flashdreams")
    )
    return Path(cache_root) / "realesrgan"


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style SR network used by realesr-general-x4v3."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            nn.PReLU(num_parameters=num_feat),
        ]
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.PReLU(num_parameters=num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsampler(self.body(x))
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


class ResidualDenseBlock(nn.Module):
    """Residual dense block used by RRDBNet."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-residual dense block used by RRDBNet."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet used by RealESRGAN_x4plus and the anime_6B variant (x4)."""

    upscale = 4

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


MODELS: dict[str, tuple[str, dict]] = {
    "general-x4v3": (f"{RELEASES}/v0.2.5.0/realesr-general-x4v3.pth", {}),
    "animevideov3": (
        f"{RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
        {"num_conv": 16},
    ),
    "x4plus": (f"{RELEASES}/v0.1.0/RealESRGAN_x4plus.pth", {"rrdb": True}),
    "x4plus-anime-6b": (
        f"{RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        {"rrdb": True, "num_block": 6},
    ),
}


def load_model(name: str, device: torch.device, fp16: bool) -> nn.Module:
    url, cfg = MODELS[name]
    weights = cache_dir() / url.rsplit("/", 1)[1]
    if not weights.is_file():
        weights.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading weights from {url}")
        urllib.request.urlretrieve(url, weights)  # noqa: S310
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    state = ckpt.get("params", ckpt.get("params_ema", ckpt))
    cfg = dict(cfg)
    if cfg.pop("rrdb", False):
        model: nn.Module = RRDBNet(**cfg)
    else:
        model = SRVGGNetCompact(**cfg)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if fp16:
        model.half()
    return model.to(memory_format=torch.channels_last)


def unsharp(frame: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Baseline unsharp mask: addWeighted 1.8/-0.8 with Gaussian sigma."""
    blur = cv2.GaussianBlur(frame, (0, 0), sigma)
    return cv2.addWeighted(frame, 1.8, blur, -0.8, 0)


@torch.inference_mode()
def enhance_batch(
    model: nn.Module,
    frames_bgr: np.ndarray,
    device: torch.device,
    fp16: bool,
) -> np.ndarray:
    """Run a (N, H, W, 3) uint8 BGR batch through SR-then-downscale."""
    x = torch.from_numpy(frames_bgr).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    x = x.to(torch.float16 if fp16 else torch.float32).div_(255.0)
    y = model(x)
    y = F.interpolate(y, scale_factor=1.0 / model.upscale, mode="area")
    y = y.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
    return y.permute(0, 2, 3, 1).cpu().numpy()


def process_video(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open input video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"cannot open output video for writing: {args.output}")

    device = torch.device(args.device)
    model = None
    if args.mode == "learned":
        model = load_model(args.model, device, args.fp16)
        if args.compile:
            model = torch.compile(model, mode="reduce-overhead")
        warm = np.zeros((1, height, width, 3), dtype=np.uint8)
        enhance_batch(model, warm, device, args.fp16)
        if device.type == "cuda":
            torch.cuda.synchronize()

    n_frames = 0
    gpu_seconds = 0.0
    wall_start = time.perf_counter()
    batch: list[np.ndarray] = []

    def flush() -> None:
        nonlocal n_frames, gpu_seconds
        if not batch:
            return
        stack = np.stack(batch)
        if args.mode == "learned":
            assert model is not None
            t0 = time.perf_counter()
            out = enhance_batch(model, stack, device, args.fp16)
            if device.type == "cuda":
                torch.cuda.synchronize()
            gpu_seconds += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            out = np.stack([unsharp(f) for f in stack])
            gpu_seconds += time.perf_counter() - t0
        for frame in out:
            writer.write(frame)
        n_frames += len(batch)
        batch.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        batch.append(frame)
        if len(batch) >= args.batch:
            flush()
    flush()
    cap.release()
    writer.release()

    wall = time.perf_counter() - wall_start
    if n_frames == 0:
        raise SystemExit("no frames decoded from input")
    print(
        f"mode={args.mode} frames={n_frames} "
        f"compute_fps={n_frames / gpu_seconds:.1f} "
        f"end_to_end_fps={n_frames / wall:.1f} "
        f"({width}x{height})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input mp4")
    parser.add_argument("--output", required=True, help="output mp4")
    parser.add_argument(
        "--mode",
        choices=["learned", "unsharp"],
        default="learned",
        help="learned SR restoration (default) or unsharp-mask baseline",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="general-x4v3",
        help="Real-ESRGAN checkpoint for --mode learned",
    )
    parser.add_argument("--batch", type=int, default=4, help="frames per GPU batch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", dest="fp16", action="store_false")
    parser.add_argument("--compile", action="store_true", help="torch.compile model")
    return parser.parse_args()


if __name__ == "__main__":
    process_video(parse_args())
