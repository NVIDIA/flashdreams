# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity probe: this package against a cosmos-framework Transfer run.

    python -m cosmos3_nano_sim_transfer.probes.parity \\
        --release RELEASE_DIR --vae Wan2.2_VAE.pth --out OUT_DIR \\
        --ref SAMPLE_DIR [--ref SAMPLE_DIR ...] [--floor LABEL=VIDEO ...] \\
        [--control NAME=VIDEO] [--history NAME=W/S|full] [--taehv-super] \\
        [--gpu-lock FILE ...]

A reference is one sample folder written by cosmos-framework's inference CLI
for a Transfer request: ``sample_args.json`` (prompt, seed, frame count, fps,
control) and ``vision.mp4``, with the run's ``setup_args.json`` (its
``kv_cache_inference_size`` and ``attention_sink_size`` overrides give the
history; none means full history) in the folder or its parent. A tensor dump
of the same run, ``ref.pt`` in either folder, enables the bit-level checks. It
is a ``torch.save`` dict with:

- ``packs``: per network pass ``text_ids`` ``[N]`` (first pass),
  ``position_ids`` ``[3, L]`` (the first pass starts with the ``N`` text
  positions, every pass ends with its latent's vision positions) and
  ``frame_idx`` (the latent);
- ``seeds``: per cache write ``frame_idx`` (``2k`` the control of latent
  ``k``, ``2k + 1`` its clean RGB), ``position_frame_idx`` (``k``),
  ``frame_latent`` ``[1, C, 1, h, w]`` and, optionally, the history limits
  ``sink_tokens`` and ``max_tokens`` of that write (checked against the
  reference's window);
- ``frames``: per latent ``noise`` and ``denoised``, ``[1, C, 1, h, w]``;
- ``decodes``: per latent ``pixels_u8`` ``[1, 3, t, H, W]``.

Per reference the probe runs the package's own path, as a live loop would: its
prompt ids, its Wan 2.2 control encoder fed a latent at a time, the rollout
with the reference's history, and the decoder, timing each stage per latent
and recording GPU memory. With a dump it first checks prompt ids, text and
vision positions of every pass and the initial noise against it, and that the
dump's history limits follow the reference's window; runs every step
teacher-forced on the reference's exact history (its control latents, its own
clean latents replayed), where only one step's numerics separate each latent
from the reference (with a window, again with full history, to show what a
wrong selection does); decodes the dump's latents (decoder only); and reruns
the rollout freely on the dump's control latents. Every decoded video is written as cosmos-framework writes its own
(lossless H.264 in YUV 4:2:0) and the file is compared with ``vision.mp4``:
PSNR per frame in RGB (peak 255, capped at 99 dB), median per second. The
frames as decoded are compared too, without the 4:2:0 conversion that both
files share. ``--floor LABEL=VIDEO``
names a second cosmos-framework run of the same request (another GPU); its
distance from ``vision.mp4``, measured the same way, is the reference's own
spread, and each run is reported against it second by second. Bit-exact
equality of the pixels is not expected: attention kernels and accumulation
order differ, and the rollout amplifies such differences over a clip.

The network weights are read into host memory before any ``--gpu-lock`` file
is taken (``flock``, exclusive, in the order given), so a shared GPU is held
only for GPU work. Writes ``OUT_DIR/<name>/metrics.json`` after each stage and
the decoded videos beside it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..config import HistoryProfile, TransferModelConfig, steady_from
from ..impl.control import (
    ControlEncoder,
    latent_frame_spans,
    read_control_video,
    resize_control,
    to_model_range,
)
from ..impl.decode import LatentDecoder, to_uint8
from ..impl.prompt import prompt_token_ids
from ..impl.rollout import TransferRollout, text_positions

PSNR_CAP = 99.0
"""PSNR reported for identical frames."""

SUPPORTED_SAMPLE_ARGS: dict[str, Any] = {
    "autoregressive": True,
    "guidance": 1.0,
    "num_conditional_frames": 0,
    "num_first_chunk_conditional_frames": 0,
    "share_vision_temporal_positions": True,
    "transfer_resize_mode": "direct_resize",
}
"""Request settings this package implements; a reference with others is refused."""

MEMORY_EVERY = 50
"""Record GPU memory after every this many latents (and after the last)."""


# -- reference runs ---------------------------------------------------------------------------


def parse_history(text: str, num_latents: int) -> HistoryProfile:
    """``W/S`` or ``full`` (every latent in view, no sinks)."""
    if text == "full":
        return HistoryProfile(num_latents, 0)
    window, _, sinks = text.partition("/")
    return HistoryProfile(int(window), int(sinks or 0))


def history_from_setup(setup_args: dict[str, Any], num_latents: int) -> HistoryProfile:
    """History of a cosmos-framework run from its ``experiment_overrides``.

    Raises:
        ValueError: The run does not state its history window.
    """
    overrides = dict(
        item.split("=", 1) for item in setup_args.get("experiment_overrides", [])
    )
    window = overrides.get("model.config.kv_cache_inference_size")
    sinks = overrides.get("model.config.attention_sink_size", "0")
    if window is None:
        raise ValueError(
            "setup_args.json does not set model.config.kv_cache_inference_size; "
            "pass --history NAME=W/S or NAME=full"
        )
    if window in ("null", "None"):
        return HistoryProfile(num_latents, 0)
    return HistoryProfile(int(window), int(sinks))


@dataclass(frozen=True)
class ReferenceRun:
    """One cosmos-framework Transfer sample and what is needed to rerun it."""

    name: str
    folder: Path
    prompt: str
    seed: int
    num_frames: int
    fps: float
    hint: str
    control_path: Path
    video_path: Path
    history: HistoryProfile
    num_steps: int
    dump_path: Path | None
    emphasize_control_in_prompt: bool = False
    prompt_aspect_ratio: str | None = None

    @property
    def num_latents(self) -> int:
        return len(latent_frame_spans(self.num_frames))

    @property
    def full_history(self) -> bool:
        """Every earlier latent stays in view (no window)."""
        return (
            self.history.sink_frames == 0
            and self.history.window_frames >= self.num_latents
        )

    @property
    def label(self) -> str:
        """``<name>_full`` or ``<name>_w<W>s<S>``: names the output folder and ``--floor`` keys."""
        if self.full_history:
            return f"{self.name}_full"
        return f"{self.name}_w{self.history.window_frames}s{self.history.sink_frames}"

    @classmethod
    def load(
        cls,
        folder: str | Path,
        *,
        control: str | Path | None = None,
        history: str | None = None,
    ) -> ReferenceRun:
        """Read a sample folder (see the module docstring).

        Args:
            folder: The sample folder (holds ``sample_args.json``).
            control: Control video to encode; the request's own input by
                default. The folder's ``control_<hint>.mp4`` is a lossy
                re-encode of it, so a run on that copy measures another
                generation; pass it here only knowingly.
            history: ``W/S`` or ``full``, overriding ``setup_args.json``.

        Raises:
            FileNotFoundError: A required file is missing, or the request's
                control input is gone and ``control`` is not given.
            ValueError: The request uses settings this package does not
                implement, or its length is not whole latents (``1 + 4n``
                frames), which the probe does not compare.
        """
        folder = Path(folder)
        args = json.loads((folder / "sample_args.json").read_text())
        for key, expected in SUPPORTED_SAMPLE_ARGS.items():
            if key in args and args[key] != expected:
                raise ValueError(
                    f"{folder}: {key}={args[key]!r} is not supported (needs {expected!r})"
                )
        emphasis = args.get("emphasize_control_in_prompt", False)
        if not isinstance(emphasis, bool):
            raise ValueError(f"{folder}: emphasize_control_in_prompt must be boolean")
        aspect_ratio = args.get("aspect_ratio")
        if aspect_ratio is not None and not isinstance(aspect_ratio, str):
            raise ValueError(f"{folder}: aspect_ratio must be a string or null")
        hints = [h for h in ("seg", "edge", "depth", "blur") if args.get(h)]
        if len(hints) != 1:
            raise ValueError(f"{folder}: expected exactly one control, got {hints}")
        hint = hints[0]
        if (
            float(args[hint].get("weight", 1.0)) != 1.0
            or float(args.get("control_guidance", 1.0)) != 1.0
        ):
            raise ValueError(f"{folder}: control weight and guidance must be 1")
        num_frames = int(args["num_frames"])
        if num_frames < 1 or (num_frames - 1) % 4:
            raise ValueError(
                f"{folder}: {num_frames} frames is not 1 + 4n; the probe compares "
                "requests of whole latents only"
            )
        if control is None:
            control = Path(args[hint]["control_path"])
            if not control.is_file():
                raise FileNotFoundError(
                    f"{folder}: the request's control input {control} is gone; "
                    f"pass --control {args.get('name') or folder.name}=VIDEO "
                    f"(the folder's control_{hint}.mp4 is a lossy re-encode)"
                )
        num_latents = len(latent_frame_spans(num_frames))
        if history is not None:
            profile = parse_history(history, num_latents)
        else:
            setup = next(
                (
                    p
                    for p in (
                        folder / "setup_args.json",
                        folder.parent / "setup_args.json",
                    )
                    if p.is_file()
                ),
                None,
            )
            if setup is None:
                raise FileNotFoundError(
                    f"no setup_args.json in {folder} or its parent; pass --history"
                )
            profile = history_from_setup(json.loads(setup.read_text()), num_latents)
        dump = next(
            (p for p in (folder / "ref.pt", folder.parent / "ref.pt") if p.is_file()),
            None,
        )
        return cls(
            name=str(args.get("name") or folder.name),
            folder=folder,
            prompt=str(args["prompt"]),
            seed=int(args["seed"]),
            num_frames=num_frames,
            fps=float(args["fps"]),
            hint=hint,
            control_path=Path(control),
            video_path=folder / "vision.mp4",
            history=profile,
            num_steps=int(args.get("num_steps", 4)),
            dump_path=dump,
            emphasize_control_in_prompt=emphasis,
            prompt_aspect_ratio=aspect_ratio,
        )


# -- metrics ----------------------------------------------------------------------------------


def frame_psnr(a: Tensor, b: Tensor) -> float:
    """PSNR of two uint8 frames (any layout), peak 255, capped at :data:`PSNR_CAP`."""
    mse = float(((a.double() - b.double()) ** 2).mean())
    return PSNR_CAP if mse == 0 else min(PSNR_CAP, 10 * math.log10(255.0**2 / mse))


def latent_psnr(latent: Tensor, reference: Tensor) -> float:
    """PSNR of a latent against a reference latent, peak ``max |reference|``."""
    reference = reference.double()
    mse = float(((latent.double() - reference) ** 2).mean())
    peak = float(reference.abs().max())
    return PSNR_CAP if mse == 0 else min(PSNR_CAP, 10 * math.log10(peak**2 / mse))


def per_second(values: Sequence[float], fps: float) -> list[float]:
    """Median of per-frame values over each second (``round(fps)`` frames)."""
    step = max(1, round(fps))
    return [float(np.median(values[i : i + step])) for i in range(0, len(values), step)]


def video_psnr(frames: Tensor, reference: Tensor) -> list[float]:
    """Per-frame PSNR of ``[T, H, W, 3]`` uint8 videos over their common length."""
    n = min(len(frames), len(reference))
    return [frame_psnr(frames[i], reference[i]) for i in range(n)]


def against_floor(psnr: Sequence[float], floor: Sequence[float]) -> dict[str, Any]:
    """Per-second margin over the reference's own spread (positive: closer than it)."""
    delta = [a - b for a, b in zip(psnr, floor, strict=False)]
    return {
        "floor_per_second": list(floor),
        "delta_per_second": delta,
        "all_seconds_at_or_above_floor": all(d >= 0 for d in delta),
        "worst_delta": min(delta) if delta else None,
    }


def read_video(path: Path, frames: int | None = None) -> Tensor:
    """``[T, H, W, 3]`` uint8 frames of a video file."""
    from flashdreams.infra.runner_io import read_video_window_rgb

    return torch.from_numpy(read_video_window_rgb(path, frame_count=frames).copy())


def write_video(path: Path, frames: Tensor, fps: float) -> None:
    """Write ``[T, H, W, 3]`` uint8 frames as cosmos-framework writes its outputs.

    Lossless H.264 (``qp=0``) in ``yuv420p``: the only loss is the conversion to
    subsampled YUV, which the reference's ``vision.mp4`` carries as well.
    """
    import mediapy

    mediapy.write_video(str(path), frames.numpy(), fps=fps, qp=0)


def compare_as_file(
    frames: Tensor, reference: Tensor, fps: float, path: Path
) -> dict[str, list[float]]:
    """Per-second PSNR of ``frames`` against ``reference`` (a cosmos-framework file).

    The frames are written with :func:`write_video` and read back, so both sides
    went through the same YUV 4:2:0 conversion, as when two cosmos-framework
    outputs are compared; the comparison of the frames as decoded is kept too.
    """
    write_video(path, frames, fps)
    encoded = read_video(path)
    return {
        "psnr_per_second_vs_vision_mp4": per_second(
            video_psnr(encoded, reference), fps
        ),
        "psnr_per_second_frames_vs_vision_mp4": per_second(
            video_psnr(frames, reference), fps
        ),
    }


# -- GPU session ------------------------------------------------------------------------------


def _lock_exclusive(handle: Any) -> None:
    """Block until an exclusive ``flock`` on ``handle`` is held.

    Raises:
        RuntimeError: The platform has no ``flock`` (Windows).
    """
    try:
        import fcntl
    except ImportError:
        raise RuntimeError("--gpu-lock needs flock (Linux or macOS)") from None
    fcntl.flock(handle, fcntl.LOCK_EX)


@contextmanager
def gpu_locks(paths: Sequence[str]) -> Iterator[float]:
    """Hold an exclusive ``flock`` on each file, taken in order; yields the wait in seconds."""
    started = time.perf_counter()
    handles = []
    try:
        for path in paths:
            handle = open(path, "a")  # noqa: SIM115 - held until the context exits
            handles.append(handle)
            _lock_exclusive(handle)
        yield time.perf_counter() - started
    finally:
        for handle in reversed(handles):
            handle.close()


def _sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _dump_controls(dump: dict[str, Any]) -> list[Tensor]:
    """The dump's control latents, ``[C, 1, h, w]`` each, in latent order."""
    controls = {
        int(s["position_frame_idx"]): s["frame_latent"][0]
        for s in dump["seeds"]
        if int(s["frame_idx"]) % 2 == 0
    }
    return [controls[k] for k in range(len(controls))]


def _dump_clean(dump: dict[str, Any]) -> list[Tensor]:
    """The clean latents the reference committed to its cache, ``[C, 1, h, w]`` each."""
    clean = {
        int(s["position_frame_idx"]): s["frame_latent"][0]
        for s in dump["seeds"]
        if int(s["frame_idx"]) % 2 == 1
    }
    return [clean[k] for k in range(len(clean))]


def _reference_history_limits(
    dump: dict[str, Any], ref: ReferenceRun, tokens_per_latent: int
) -> dict[str, Any]:
    """The history limits the reference used per cache write, against its window.

    cosmos-framework pins ``2 * sinks`` entries and keeps ``2 * (window - sinks - 1)``
    recent entries for a control, one more for a clean latent (its own control).
    This checks that the reference ran with the window the probe read from it
    (and runs the package with), counted by that rule. It does not look at the
    package's cache: that the stream cache selects by the same rule is the CPU
    test ``test_stream_cache_selects_what_cosmos_framework_selects``, and the
    teacher-forced run checks it on the GPU through every eviction.
    """
    seen: dict[str, set[tuple[int, int | None]]] = {"control": set(), "clean": set()}
    for seed in dump["seeds"]:
        if "sink_tokens" not in seed:
            return {}
        kind = "control" if int(seed["frame_idx"]) % 2 == 0 else "clean"
        limit = seed.get("max_tokens")
        seen[kind].add(
            (int(seed["sink_tokens"] or 0), None if limit is None else int(limit))
        )
    if ref.full_history:
        expected = {"control": {(0, None)}, "clean": {(0, None)}}
    else:
        window, sinks = ref.history.window_frames, ref.history.sink_frames
        recent = 2 * (window - sinks - 1) * tokens_per_latent
        pinned = 2 * sinks * tokens_per_latent
        expected = {
            "control": {(pinned, recent)},
            "clean": {(pinned, recent + tokens_per_latent)},
        }
    return {
        "reference_history_limits": {k: sorted(v, key=str) for k, v in seen.items()},
        "reference_history_limits_follow_window": seen == expected,
    }


def _dump_pixels(dump: dict[str, Any]) -> Tensor:
    """The dump's decoded frames, ``[T, H, W, 3]`` uint8."""
    pixels = torch.cat([d["pixels_u8"][0] for d in dump["decodes"]], dim=1)
    return pixels.permute(1, 2, 3, 0).contiguous()


def _frames(pixels: list[Tensor]) -> Tensor:
    """Concatenate ``[t, 3, H, W]`` uint8 chunks to ``[T, H, W, 3]``."""
    return torch.cat(pixels).permute(0, 2, 3, 1).contiguous()


def _median_from(values: Sequence[float], start: int) -> float:
    return float(np.median(values[min(start, len(values) - 1) :]))


class Probe:
    """One loaded network, encoder and decoder, run against several references."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.release = Path(args.release)
        self.model = TransferModelConfig.from_export(self.release)
        self.device = torch.device("cuda")

    def load(self) -> dict[str, float]:
        """Read the weights into host memory (no GPU work yet)."""
        from cosmos_interactive.checkpoint_load import load_exported_checkpoint

        started = time.perf_counter()
        net = load_exported_checkpoint(self.release, device="cpu").eval()
        with torch.no_grad():  # memory-mapped shards: touch every weight now
            for param in net.parameters():
                param.data.float().sum()
        self.net = net
        if self.args.vae:
            Path(self.args.vae).read_bytes()  # into the page cache
        return {"load_to_host_s": time.perf_counter() - started}

    def to_gpu(self) -> dict[str, float]:
        """Move the network and build the encoder and decoders on the GPU."""
        started = time.perf_counter()
        self.net = self.net.to(self.device)
        self.encoder = ControlEncoder(checkpoint_path=self.args.vae, device=self.device)
        self.decoder = LatentDecoder(
            "wan2.2", checkpoint_path=self.args.vae, device=self.device
        )
        self.taehv = (
            LatentDecoder("taehv-super", device=self.device)
            if self.args.taehv_super
            else None
        )
        _sync()
        return {"to_gpu_s": time.perf_counter() - started}

    def decode(
        self, latents: Sequence[Tensor], decoder: LatentDecoder
    ) -> tuple[Tensor, list[float]]:
        """Decode latents in order with a fresh decoder cache; ``[T, H, W, 3]`` and ms per latent."""
        decoder.reset()
        pixels, times = [], []
        for latent in latents:
            started = _sync()
            pixels.append(to_uint8(decoder.decode(latent)).cpu())
            times.append(1000 * (_sync() - started))
        return _frames(pixels), times

    def run(self, ref: ReferenceRun, out: Path) -> dict[str, Any]:
        out.mkdir(parents=True, exist_ok=True)
        n = ref.num_latents
        if self.args.max_latents:
            n = min(n, self.args.max_latents)
        metrics: dict[str, Any] = {
            "reference": {
                "name": ref.name,
                "label": ref.label,
                "folder": str(ref.folder),
                "control": str(ref.control_path),
                "dump": str(ref.dump_path) if ref.dump_path else None,
                "frames": ref.num_frames,
                "fps": ref.fps,
                "seed": ref.seed,
                "hint": ref.hint,
                "emphasize_control_in_prompt": ref.emphasize_control_in_prompt,
                "prompt_aspect_ratio": ref.prompt_aspect_ratio,
            },
            "history": [ref.history.window_frames, ref.history.sink_frames],
            "latents": n,
            "psnr_method": "per frame, RGB, peak 255, capped at 99 dB; median per second",
        }

        def save() -> None:
            (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

        text_ids = prompt_token_ids(
            ref.prompt,
            num_frames=ref.num_frames,
            fps=ref.fps,
            tokenizer_path=self.release / "text_tokenizer",
            hint=ref.hint,
            emphasize_control_in_prompt=ref.emphasize_control_in_prompt,
            aspect_ratio=ref.prompt_aspect_ratio,
        )
        metrics["text_tokens"] = int(text_ids.numel())
        reference = read_video(ref.video_path)
        frames_out = 1 + 4 * (n - 1)

        floor_path = self.args.floor.get(ref.label)
        if floor_path:
            floor_video = read_video(Path(floor_path))
            floor = per_second(video_psnr(floor_video[:frames_out], reference), ref.fps)
            metrics["floor"] = {
                "video": floor_path,
                "psnr_per_second": floor,
            }
            del floor_video
        else:
            floor = None
        save()

        if ref.num_steps != len(self.model.t_list):
            raise ValueError(
                f"{ref.name} asks for {ref.num_steps} steps; the checkpoint's "
                f"schedule has {len(self.model.t_list)}"
            )
        rollout = TransferRollout(
            self.net, self.model, history=ref.history, fps=ref.fps
        )
        rollout.start(text_ids, seed=ref.seed)

        dump = (
            torch.load(ref.dump_path, map_location="cpu", weights_only=False)
            if ref.dump_path
            else None
        )
        if dump is not None:
            metrics["checks"] = self._checks(dump, text_ids, rollout, ref, n)
            save()
            metrics["teacher_forced"] = self._teacher_forced(
                rollout, text_ids, ref, dump, n
            )
            save()
            if not ref.full_history:
                # Calibration: the same check with every latent kept in view shows
                # what a wrong history selection does to it.
                full = TransferRollout(
                    self.net, self.model, history=HistoryProfile(n, 0), fps=ref.fps
                )
                metrics["teacher_forced_full_history"] = self._teacher_forced(
                    full, text_ids, ref, dump, n
                )
                del full
                torch.cuda.empty_cache()
                save()
            ref_latents = [f["denoised"][0].float() for f in dump["frames"][:n]]
            dump_pixels = _dump_pixels(dump)[:frames_out]
            pixels, _ = self.decode(ref_latents, self.decoder)
            dump_file = compare_as_file(
                dump_pixels, reference, ref.fps, out / "dump_decode.mp4"
            )
            metrics["decoder_only"] = {
                "psnr_vs_dump_decode_median": float(
                    np.median(video_psnr(pixels, dump_pixels))
                ),
                **compare_as_file(pixels, reference, ref.fps, out / "decoder_only.mp4"),
                # The dump's own decode written like ours: checks the writer
                # against cosmos-framework's (identical conversion gives 99 dB).
                "dump_decode_psnr_per_second_vs_vision_mp4": dump_file[
                    "psnr_per_second_vs_vision_mp4"
                ],
                "dump_decode_frames_psnr_per_second_vs_vision_mp4": dump_file[
                    "psnr_per_second_frames_vs_vision_mp4"
                ],
            }
            save()
            metrics["ref_controls"] = self._rollout_on(
                rollout,
                text_ids,
                ref,
                _dump_controls(dump)[:n],
                ref_latents,
                dump_pixels,
                reference,
                floor,
                out / "ref_controls.mp4",
            )
            save()

        own, controls, latents = self._own_path(
            rollout, text_ids, ref, n, reference, floor, out
        )
        metrics["own_path"] = own
        if dump is not None:
            ref_controls = _dump_controls(dump)[:n]
            psnrs = [
                latent_psnr(a, b) for a, b in zip(controls, ref_controls, strict=True)
            ]
            own["control_latent_psnr_vs_dump"] = {
                "median": float(np.median(psnrs)),
                "min": float(min(psnrs)),
                "max_abs_diff": float(
                    max(
                        (a.float() - b.float()).abs().max()
                        for a, b in zip(controls, ref_controls, strict=True)
                    )
                ),
            }
        save()
        if self.taehv is not None:
            wan, _ = self.decode(latents, self.decoder)
            tae, times = self.decode(latents, self.taehv)
            own["taehv_super"] = {
                "ms_per_latent_median_from_3": _median_from(times, 3),
                "psnr_vs_wan_decode_median": float(np.median(video_psnr(tae, wan))),
                **compare_as_file(tae, reference, ref.fps, out / "own_path_taehv.mp4"),
            }
            save()
        del rollout
        torch.cuda.empty_cache()
        return metrics

    def _checks(
        self,
        dump: dict[str, Any],
        text_ids: Tensor,
        rollout: TransferRollout,
        ref: ReferenceRun,
        n: int,
    ) -> dict[str, Any]:
        packs = dump["packs"]
        ref_ids = torch.as_tensor(packs[0]["text_ids"]).reshape(-1)
        num_text = int(ref_ids.numel())
        first = torch.as_tensor(packs[0]["position_ids"]).float()
        vision_diff = 0.0
        for pack in packs:
            k = int(pack["frame_idx"])
            ids = torch.as_tensor(pack["position_ids"]).float()[
                :, -rollout.tokens_per_latent :
            ]
            vision_diff = max(
                vision_diff, float((ids - rollout.positions(k)).abs().max())
            )
        noise = {
            k: bool(
                torch.equal(
                    rollout.initial_noise(k).cpu(), dump["frames"][k]["noise"][0]
                )
            )
            for k in sorted({0, 1, n - 1})
        }
        return {
            "prompt_ids_equal": bool(torch.equal(text_ids, ref_ids)),
            "prompt_tokens": [int(text_ids.numel()), num_text],
            "text_positions_equal": bool(
                torch.equal(first[:, :num_text], text_positions(num_text))
            ),
            "vision_positions_max_abs_diff": vision_diff,
            "vision_positions_packs": len(packs),
            "initial_noise_bit_equal": noise,
            **_reference_history_limits(dump, ref, rollout.tokens_per_latent),
        }

    def _teacher_forced(
        self,
        rollout: TransferRollout,
        text_ids: Tensor,
        ref: ReferenceRun,
        dump: dict[str, Any],
        n: int,
    ) -> dict[str, Any]:
        """Each step on the reference's exact history: its control latents, and its
        own clean latents replayed into the cache. Only one step's numerics
        separate each output from the reference, so a wrong history selection
        shows as a drop where it starts, not hidden in the rollout's drift."""
        controls = _dump_controls(dump)[:n]
        clean = _dump_clean(dump)
        rollout.start(text_ids, seed=ref.seed)
        psnrs = []
        for k in range(n):
            replay = clean[k] if k < n - 1 else None
            latent = rollout.step(controls[k], last=k == n - 1, replay=replay)
            psnrs.append(latent_psnr(latent.cpu(), dump["frames"][k]["denoised"][0]))
        history = rollout.history
        return {
            "history": [history.window_frames, history.sink_frames],
            "latent_psnr_vs_dump": psnrs,
            "min": float(min(psnrs)),
            "median": float(np.median(psnrs)),
        }

    def _rollout_on(
        self,
        rollout: TransferRollout,
        text_ids: Tensor,
        ref: ReferenceRun,
        controls: list[Tensor],
        ref_latents: list[Tensor],
        dump_pixels: Tensor,
        reference: Tensor,
        floor: list[float] | None,
        video: Path,
    ) -> dict[str, Any]:
        """Rollout on given control latents, then decode; latent and pixel PSNR."""
        rollout.start(text_ids, seed=ref.seed)
        n = len(controls)
        latents, times = [], []
        for k, control in enumerate(controls):
            started = _sync()
            latents.append(rollout.step(control, last=k == n - 1).cpu())
            times.append(1000 * (_sync() - started))
        pixels, _ = self.decode(latents, self.decoder)
        compared = compare_as_file(pixels, reference, ref.fps, video)
        psnr = compared["psnr_per_second_vs_vision_mp4"]
        result: dict[str, Any] = {
            "latent_psnr_vs_dump": [
                latent_psnr(a, b) for a, b in zip(latents, ref_latents, strict=True)
            ],
            **compared,
            "psnr_per_second_vs_dump_decode": per_second(
                video_psnr(pixels, dump_pixels), ref.fps
            ),
            "model_ms_per_latent": times,
            "model_ms_median_from_3": _median_from(times, 3),
            "finite": all(bool(torch.isfinite(x).all()) for x in latents),
        }
        if floor is not None:
            result["vs_floor"] = against_floor(psnr, floor)
        return result

    def _own_path(
        self,
        rollout: TransferRollout,
        text_ids: Tensor,
        ref: ReferenceRun,
        n: int,
        reference: Tensor,
        floor: list[float] | None,
        out: Path,
    ) -> tuple[dict[str, Any], list[Tensor], list[Tensor]]:
        """Encode, generate and decode a latent at a time, as a live loop does."""
        frames = read_control_video(ref.control_path, max_frames=ref.num_frames)
        spans = latent_frame_spans(int(frames.shape[0]))[:n]
        self.encoder.reset()
        self.decoder.reset()
        rollout.start(text_ids, seed=ref.seed)
        torch.cuda.reset_peak_memory_stats()
        stages: dict[str, list[float]] = {
            "encode": [],
            "model": [],
            "decode": [],
            "total": [],
        }
        memory: list[dict[str, Any]] = []
        allocated: list[float] = []
        controls, latents, pixels = [], [], []
        for k, span in enumerate(spans):
            t0 = _sync()
            control = self.encoder.encode(
                to_model_range(resize_control(frames[span], ref.hint))
            )
            t1 = _sync()
            latent = rollout.step(control, last=k == n - 1)
            t2 = _sync()
            pixels.append(to_uint8(self.decoder.decode(latent)).cpu())
            t3 = _sync()
            for stage, ms in zip(
                stages, (t1 - t0, t2 - t1, t3 - t2, t3 - t0), strict=True
            ):
                stages[stage].append(1000 * ms)
            controls.append(control.cpu())
            latents.append(latent.cpu())
            allocated.append(torch.cuda.memory_allocated() / 2**30)
            if (k > 0 and k % MEMORY_EVERY == 0) or k == n - 1:
                memory.append(
                    {
                        "after_latent": k,
                        "allocated_gib": allocated[-1],
                        "peak_since_previous_gib": torch.cuda.max_memory_allocated()
                        / 2**30,
                        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                        "cached_tokens": rollout.cached_tokens,
                    }
                )
                torch.cuda.reset_peak_memory_stats()
        compared = compare_as_file(
            _frames(pixels), reference, ref.fps, out / "own_path.mp4"
        )
        psnr = compared["psnr_per_second_vs_vision_mp4"]
        steady = steady_from(ref.history, n)
        result: dict[str, Any] = {
            **compared,
            "ms_per_latent": stages,
            "steady_from_latent": steady,
            "steady_median_ms": {s: _median_from(v, steady) for s, v in stages.items()},
            "memory": memory,
            "allocated_gib_per_latent": allocated,
            "cache_gib": sum(g._k.numel() + g._v.numel() for g in rollout.cache.gen)
            * rollout.cache.gen[0]._k.element_size()
            / 2**30,
            "finite": all(bool(torch.isfinite(x).all()) for x in latents),
        }
        if floor is not None:
            result["vs_floor"] = against_floor(psnr, floor)
        return result, controls, latents


def _named(values: Sequence[str], flag: str) -> dict[str, str]:
    pairs = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep:
            raise SystemExit(f"{flag} takes NAME=VALUE, got {value!r}")
        pairs[name] = path
    return pairs


def _summary(metrics: dict[str, Any]) -> str:
    lines = [
        f"== {metrics['reference']['name']} history {metrics['history']} latents {metrics['latents']}"
    ]
    if "checks" in metrics:
        lines.append(f"checks {json.dumps(metrics['checks'])}")
    for run in ("teacher_forced", "teacher_forced_full_history"):
        if run in metrics:
            psnr = metrics[run]["latent_psnr_vs_dump"]
            lines.append(
                f"{run} {metrics[run]['history']} {[round(x, 1) for x in psnr]}"
            )
    if "floor" in metrics:
        lines.append(
            f"floor            {[round(x, 1) for x in metrics['floor']['psnr_per_second']]}"
        )
    for run in ("ref_controls", "own_path"):
        if run in metrics:
            psnr = metrics[run]["psnr_per_second_vs_vision_mp4"]
            lines.append(f"{run:16s} {[round(x, 1) for x in psnr]}")
            if "vs_floor" in metrics[run]:
                lines.append(
                    f"  vs floor {[round(x, 1) for x in metrics[run]['vs_floor']['delta_per_second']]}"
                )
    if "own_path" in metrics:
        lines.append(
            f"steady ms {json.dumps({k: round(v, 1) for k, v in metrics['own_path']['steady_median_ms'].items()})}"
        )
        lines.append(f"memory {json.dumps(metrics['own_path']['memory'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(
        description="Parity probe: this package against a cosmos-framework Transfer run."
    )
    parser.add_argument(
        "--release",
        required=True,
        help="release snapshot (checkpoint.json, transformer/, text_tokenizer/)",
    )
    parser.add_argument(
        "--vae",
        default=None,
        help="Wan2.2_VAE.pth; the FlashDreams default when omitted",
    )
    parser.add_argument(
        "--ref",
        action="append",
        required=True,
        help="cosmos-framework sample folder (repeatable)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--floor",
        action="append",
        default=[],
        help="LABEL=VIDEO: another run of the reference labeled NAME_full or NAME_wWsS",
    )
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        help="NAME=VIDEO: control video to encode",
    )
    parser.add_argument(
        "--history", action="append", default=[], help="NAME=W/S or NAME=full"
    )
    parser.add_argument(
        "--max-latents",
        type=int,
        default=None,
        help="stop each run after this many latents",
    )
    parser.add_argument(
        "--taehv-super",
        action="store_true",
        help="also decode the own path with TAEHV-super",
    )
    parser.add_argument(
        "--gpu-lock",
        action="append",
        default=[],
        help="file to flock exclusively before GPU work",
    )
    args = parser.parse_args(argv)
    args.floor = _named(args.floor, "--floor")
    controls = _named(args.control, "--control")
    histories = _named(args.history, "--history")

    refs = []
    for folder in args.ref:
        name = (
            json.loads((Path(folder) / "sample_args.json").read_text()).get("name")
            or Path(folder).name
        )
        refs.append(
            ReferenceRun.load(
                folder, control=controls.get(name), history=histories.get(name)
            )
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probe = Probe(args)
    session: dict[str, Any] = dict(probe.load())
    print(f"weights in host memory after {session['load_to_host_s']:.0f} s", flush=True)
    results = []
    with gpu_locks(args.gpu_lock) as waited:
        session["gpu_lock_wait_s"] = waited
        print(f"GPU locks held after {waited:.0f} s", flush=True)
        try:
            session.update(probe.to_gpu())
            session["device"] = torch.cuda.get_device_name()
            (out / "session.json").write_text(json.dumps(session, indent=1))
            for ref in refs:
                metrics = probe.run(ref, out / ref.label)
                print(_summary(metrics), flush=True)
                results.append(metrics)
        finally:
            del probe
            torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    main()
