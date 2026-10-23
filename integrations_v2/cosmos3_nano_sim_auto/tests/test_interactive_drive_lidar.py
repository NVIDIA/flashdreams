# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live LiDAR path ownership, scan timing and rig calibration checks."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_auto.apps.interactive_drive.lidar import LiveLidarStream
from cosmos3_nano_sim_auto.impl.lidar import LidarAssets, LidarSample
from cosmos_interactive.lidar import LidarGeometry, LidarTokenizer

pytestmark = pytest.mark.ci_cpu


def _stream():
    tokenizer = LidarTokenizer(
        None, LidarGeometry(channels=3, height=2, width=2, patch=1)
    )
    sample = LidarSample(
        torch.empty(0, 3, 2, 2),
        torch.empty(0),
        torch.zeros(2, 2, 3),
        torch.eye(4),
    )
    rendered = []

    def render(index, first, last):
        rendered.append((index, first.copy(), last.copy()))
        sweep = torch.ones(3, 2, 2)
        sweep[0] = float(last[0, 3]) + 1
        return sweep

    stream = LiveLidarStream(
        LidarAssets(tokenizer, sample), sensor_to_rig=np.eye(4), render_sweep=render
    )
    setattr(stream, "encoder", lambda metric: metric)
    return stream, rendered


def _path(first, last):
    path = np.repeat(np.eye(4, dtype=np.float32)[None], last - first, axis=0)
    path[:, 0, 3] = np.arange(first, last)
    return path


def test_live_sweeps_use_driven_trailing_intervals_and_bound_pose_memory():
    stream, rendered = _stream()
    stream.supply(_path(0, 1))
    stream.tokens(0, 1)
    assert rendered[0][1][0, 3] == rendered[0][2][0, 3] == 0
    for start in range(1, 25, 12):
        stream.supply(_path(start, start + 12))
        first_sweep = (start - 1) // 3 + 1
        stream.tokens(first_sweep, first_sweep + 4)
        assert set(stream.last_metrics) == {
            "lidar_control_render_s",
            "lidar_codec_encode_s",
        }
        assert all(value > 0 for value in stream.last_metrics.values())
        assert len(stream._poses) == 1
    assert [(a[1][0, 3], a[2][0, 3]) for a in rendered[1:]] == [
        (frame - 3, frame) for frame in range(3, 25, 3)
    ]
    assert stream.sweep_times(100, 3, "cpu").tolist() == [10.0, 10.1, 10.2]


def test_live_sweeps_refuse_unsupplied_or_repeated_poses_and_invalid_rig():
    stream, _ = _stream()
    with pytest.raises(ValueError, match="Supply"):
        stream.tokens(0, 1)
    stream.supply(_path(0, 1))
    stream.tokens(0, 1)
    with pytest.raises(ValueError, match="consecutively"):
        stream.tokens(0, 1)
    with pytest.raises(ValueError, match="Supply"):
        stream.tokens(1, 2)
    invalid = _path(0, 1)
    invalid[0, 0, 0] = -1
    with pytest.raises(ValueError, match="rigid"):
        stream.supply(invalid)


def test_control_and_generated_presenters_hold_sweeps_independently():
    stream, _ = _stream()
    viewport = SimpleNamespace(
        render_sweep=lambda sweep, *args, **kwargs: sweep[0, 0, 0].expand(3, 2, 2)
    )
    kwargs = dict(fps=30, viewport=viewport)
    stream.supply(_path(0, 1))
    stream.tokens(0, 1)
    assert (stream.control_frames(pixel_start=0, pixel_count=1, **kwargs) == 1).all()
    generated = torch.full((1, 3, 1, 2, 2), 99.0)
    stream.frames(generated, first_sweep=0, pixel_start=0, pixel_count=1, **kwargs)
    stream.supply(_path(1, 13))
    stream.tokens(1, 5)
    controls = stream.control_frames(pixel_start=1, pixel_count=12, **kwargs)
    assert controls[:, 0, 0, 0].tolist() == [1, 1, 4, 4, 4, 7, 7, 7, 10, 10, 10, 13]
    assert (stream.previous_range == 99).all()


def test_lidar_ranks_prefer_cameraless_ranks_one_sweep_each():
    from cosmos3_nano_sim_auto.apps.interactive_drive.lidar import lidar_ranks

    def mesh(world):
        return SimpleNamespace(world_size=world)

    assert lidar_ranks(None, 7) == [0]
    assert lidar_ranks(mesh(16), 7) == [7, 8, 9, 10]
    assert lidar_ranks(mesh(24), 7) == [7, 8, 9, 10]
    assert lidar_ranks(mesh(8), 7) == [7, 6, 5, 4]
    assert lidar_ranks(mesh(4), 7) == [3, 2, 1, 0]


def _split_worker(rank, init_file, output):
    from datetime import timedelta

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=3,
        # Spawned workers import slowly from shared filesystems.
        timeout=timedelta(seconds=300),
    )
    try:
        stream, rendered = _stream()
        # Rank 0 presents but renders nothing; ranks 1 and 2 split the sweeps
        # and rank 1 alone encodes, as an idle-rank layout on a real mesh does.
        ctx = SimpleNamespace(rank=rank, world_size=3, device=torch.device("cpu"))
        stream = LiveLidarStream(
            stream.assets,
            sensor_to_rig=np.eye(4),
            render_sweep=stream._render_sweep,
            parallel=cast(Any, ctx),
            renderers=[1, 2],
        )
        encoded = []
        setattr(
            stream, "encoder", lambda metric: encoded.append(metric.shape[2]) or metric
        )
        results = []
        stream.supply(_path(0, 1))
        stream.prefetch(0, 1)
        results.append(stream.tokens(0, 1))
        stream.supply(_path(1, 13))
        stream.prefetch(1, 5)
        results.append(stream.tokens(1, 5))
        torch.save(
            (results, stream.last_controls, [r[0] for r in rendered], encoded),
            f"{output}{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_sweeps_split_across_ranks_and_tokens_match_one_rank(tmp_path):
    import torch.multiprocessing as mp

    reference, _ = _stream()
    reference.supply(_path(0, 1))
    expected = [reference.tokens(0, 1)]
    reference.supply(_path(1, 13))
    expected.append(reference.tokens(1, 5))
    mp.spawn(
        _split_worker,
        args=(str(tmp_path / "init"), str(tmp_path / "result")),
        nprocs=3,
        join=True,
    )
    renders = {0: [], 1: [0, 1, 3], 2: [2, 4]}
    encodes = {0: [], 1: [1, 4], 2: []}
    for rank in range(3):
        tokens, controls, rendered, encoded = torch.load(
            f"{tmp_path / 'result'}{rank}.pt", weights_only=True
        )
        assert rendered == renders[rank] and encoded == encodes[rank]
        for got, want in zip(tokens, expected, strict=True):
            assert torch.equal(got, want)
        assert torch.equal(controls, reference.last_controls)


def _mesh_worker(rank, world, init_file, output):
    """One drive chunk's collectives in the backend's order, on a 5-rank mesh."""
    from datetime import timedelta

    import torch.distributed as dist
    from cosmos3_nano_sim_auto.apps.interactive_drive import backend
    from cosmos3_nano_sim_auto.apps.interactive_drive.lidar import lidar_ranks
    from cosmos3_nano_sim_auto.impl.decode import decoded_views

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=300),
    )
    try:
        cameras, span = 2, 4
        ctx = SimpleNamespace(
            rank=rank, world_size=world, is_main=rank == 0, device=torch.device("cpu")
        )
        owned = list(range(rank, cameras, world))

        def picture(view, first, last):
            frames = torch.arange(first, last, dtype=torch.uint8)[:, None, None, None]
            return (frames + 10 * view).expand(-1, 2, 2, 4).contiguous()

        ludus = SimpleNamespace(
            cameras=[0] * cameras,
            render=lambda first, last, views: torch.cat(
                [picture(view, first, last) for view in views]
                or [torch.empty(0, 2, 2, 4, dtype=torch.uint8)]
            ),
        )
        drawn = backend._DrawnRig(ludus, parallel=cast(Any, ctx))
        worker = SimpleNamespace(
            request=lambda *args, **kwargs: None,
            take=lambda start, end: ([], span),
            last_drawn=torch.cat([picture(view, 100, 100 + span) for view in owned])
            if owned
            else torch.empty(0, 2, 2, 4, dtype=torch.uint8),
        )
        driven = backend._DrivenWorker(
            worker,
            poses=cast(Any, SimpleNamespace(supplied=10**6, poses=lambda a, b: None)),
            drawn=drawn,
            owned=owned,
        )
        latent = torch.arange(cameras * 6, dtype=torch.float32).reshape(cameras, 6)
        results = []
        for restart in range(2):
            # A restart builds a fresh stream on every rank, as _start_rollout does.
            stream, rendered = _stream()
            stream = LiveLidarStream(
                stream.assets,
                sensor_to_rig=np.eye(4),
                render_sweep=stream._render_sweep,
                parallel=cast(Any, ctx),
                renderers=lidar_ranks(cast(Any, ctx), cameras),
            )
            setattr(stream, "encoder", lambda metric: metric)
            decodes = []
            setattr(stream, "decoder", lambda z: decodes.append(z.shape[2]) or z * 3)
            for (first, last), sweeps in (((0, 1), (0, 1)), ((1, 13), (1, 5))):
                stream.supply(_path(first, last))
                stream.prefetch(*sweeps)
                drawn.render(first, last, owned)
                driven.take(0, 1)
                tokens = stream.tokens(*sweeps)
                # Backend order: LiDAR decode starts, cameras decode, rank 0 waits.
                generated = torch.arange(12.0 * (sweeps[1] - sweeps[0])).reshape(
                    1, 3, -1, 2, 2
                )
                pending = stream.start_decode(generated)
                views = decoded_views(
                    latent,
                    (lambda x: x * 2) if owned else None,
                    parallel=cast(Any, ctx),
                )
                assert (pending is not None) == (rank == 0)
                if pending is not None:
                    torch.testing.assert_close(pending(), generated * 3)
                assert decodes == ([1, 4][: len(decodes)] if rank == 2 else [])
                results.append(
                    (
                        restart,
                        tokens,
                        stream.last_controls,
                        dict(stream.last_metrics),
                        [r[0] for r in rendered],
                        views if rank == 0 else None,
                    )
                )
        queued = drawn.take(len(drawn._queued)) if rank == 0 else []
        torch.save(
            (results, [[np.asarray(v).tolist() for v in f] for f in queued]),
            f"{output}{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_five_rank_drive_chunk_gathers_and_lidar_split_with_restart(tmp_path):
    """LiDAR split/encode/broadcast, RGB draw and worker gathers, decode gather."""
    import torch.multiprocessing as mp

    reference, _ = _stream()
    reference.supply(_path(0, 1))
    expected = [reference.tokens(0, 1)]
    reference.supply(_path(1, 13))
    expected.append(reference.tokens(1, 5))
    mp.spawn(
        _mesh_worker,
        args=(5, str(tmp_path / "init"), str(tmp_path / "result")),
        nprocs=5,
        join=True,
    )
    loaded = [
        torch.load(f"{tmp_path / 'result'}{rank}.pt", weights_only=False)
        for rank in range(5)
    ]
    # Cameras 0 and 1 live on ranks 0 and 1; sweeps go to ranks 2, 3, 4, 1.
    renders = {0: [], 1: [4], 2: [0, 1], 3: [2], 4: [3]}
    owner = [row[3] for row in loaded[2][0]]
    for rank, (results, queued) in enumerate(loaded):
        for index, (restart, tokens, controls, metrics, rendered, views) in enumerate(
            results
        ):
            assert torch.equal(tokens, expected[index % 2])
            assert controls.shape == (1, 3, (1, 4)[index % 2], 2, 2)
            # Every rank, rank 0 included, logs the owner's render/encode times;
            # the owner (2) and rank 0 also its decode time.
            assert metrics.keys() >= {"lidar_control_render_s", "lidar_codec_encode_s"}
            if rank not in (1, 3, 4):
                # Carried as float32 in the token broadcast.
                assert metrics == pytest.approx(owner[index], rel=1e-6)
            # The opening sweep is the owner's alone; the next chunk is split.
            opening = [0] if rank == 2 else []
            assert rendered == (renders[rank] if index % 2 else opening)
            if rank == 0:
                torch.testing.assert_close(views, torch.arange(12.0).reshape(2, 6) * 2)
    # Rank 0 queued every camera of every drawn and worker frame, in rig order.
    queued = loaded[0][1]
    # Per run: opening draw 1 + worker 4, chunk draw 12 + worker 4.
    assert len(queued) == 2 * (1 + 4 + 12 + 4)
    assert [frame[1][0][0][0] - frame[0][0][0][0] for frame in queued] == [10] * len(
        queued
    )
