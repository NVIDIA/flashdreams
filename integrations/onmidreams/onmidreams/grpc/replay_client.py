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

"""Replay recorded gRPC sessions for performance testing and regression detection.

This script reads a binary log file containing recorded gRPC requests and responses,
replays them against a running server, and compares performance metrics.

Requests are sent sequentially (waiting for each response before sending the next)
because the order matters for correct simulation—concurrent requests would result
in incorrect state.

Usage:
    # Replay session for performance testing
    python replay_client.py session.binlog --server localhost:50051

    # Show recording info without replaying
    python replay_client.py session.binlog --info

    # Generate comparison video (origin vs replay)
    python replay_client.py session.binlog --server localhost:50051 --compare-video comparison.mp4
"""

from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import grpc
import imageio.v3 as iio
import numpy as np
from loguru import logger
from onmidreams.grpc.protos import video_model_pb2, video_model_pb2_grpc
from onmidreams.grpc.recording_io import count_log_entries, read_log_entries
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass
class ReplayStats:
    """Statistics for a replayed RPC call."""

    seq: int
    method: str
    origin_duration_ns: int
    replay_duration_ns: int
    success: bool
    error: str | None = None

    @property
    def ratio(self) -> float:
        """Ratio of replay time to origin time (>1 means slower)."""
        if self.origin_duration_ns == 0:
            return 0.0
        return self.replay_duration_ns / self.origin_duration_ns


@dataclass
class FramePair:
    """A pair of origin and replayed frames for comparison."""

    origin: np.ndarray  # [H, W, 3] uint8
    replay: np.ndarray  # [H, W, 3] uint8


@dataclass
class OutputPair:
    """A pair of origin and replayed camera outputs (raw protos)."""

    origin: list[video_model_pb2.CameraOutput]
    replay: list[video_model_pb2.CameraOutput]


def decode_image_proto(image: video_model_pb2.Image) -> np.ndarray:
    """Decode an Image proto to a numpy array.

    Args:
        image: Image proto with data and format.

    Returns:
        Numpy array [H, W, 3] uint8 RGB.
    """
    img = Image.open(io.BytesIO(image.data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def camera_outputs_to_mosaic_frames(
    camera_outputs: list[video_model_pb2.CameraOutput],
    which_stream: Literal["rgb", "hdmap"] = "rgb",
) -> list[np.ndarray]:
    """Build one mosaic image per frame by stacking all camera views.

    For each time step, takes the i-th frame from each camera and arranges them
    in a horizontal strip (same height, widths preserved after resizing to common height).

    Args:
        camera_outputs: Per-camera outputs from VideoChunkReturn.camera_outputs.
        which_stream: 'rgb' for rgb_frames, 'hdmap' for hdmap_condition_frames.

    Returns:
        List of mosaic images [H, W_total, 3] uint8, one per frame index.
    """
    if not camera_outputs:
        return []

    frame_list_attr = (
        "rgb_frames" if which_stream == "rgb" else "hdmap_condition_frames"
    )
    num_frames = min(len(getattr(co, frame_list_attr)) for co in camera_outputs)
    if num_frames == 0:
        return []

    mosaics: list[np.ndarray] = []
    for i in range(num_frames):
        frames_i = [
            decode_image_proto(getattr(co, frame_list_attr)[i]) for co in camera_outputs
        ]
        # Resize to common height (use first frame's height) for clean horizontal stack
        h_common = frames_i[0].shape[0]
        resized = []
        for f in frames_i:
            if f.shape[0] != h_common:
                scale = h_common / f.shape[0]
                w_new = max(1, int(round(f.shape[1] * scale)))
                img = Image.fromarray(f).resize(
                    (w_new, h_common), Image.Resampling.LANCZOS
                )
                f = np.array(img, dtype=np.uint8)
            resized.append(f)
        mosaic = np.hstack(resized)
        mosaics.append(mosaic)
    return mosaics


class ReplayClient:
    """Replays recorded gRPC sessions against a running server."""

    def __init__(
        self,
        server_address: str,
        collect_frames: bool = False,
        disable_font: bool = False,
        disable_origin: bool = False,
        mock_burden_ms: float = 0.0,
        request_hdmap: bool = False,
        request_bev: bool = False,
        text_prompt: str = "",
        defer_frame_processing: bool = False,
    ):
        """Initialize the replay client.

        Args:
            server_address: Server address in host:port format.
            collect_frames: If True, collect frame pairs for comparison video.
            disable_font: If True, disable font in the comparison video.
            disable_origin: If True, disable origin in the comparison video.
            mock_burden_ms: Mock burden time in milliseconds to simulate client work.
            request_bev: If True, request BEV rendering from the server.
            defer_frame_processing: If True, defer frame processing until after all
                replay is done. If False (default), process frames immediately during
                each replay call (original behavior).
        """
        self.server_address = server_address
        self.collect_frames = collect_frames
        self.disable_font = disable_font
        self.disable_origin = disable_origin
        self.mock_burden_ms = mock_burden_ms
        self.request_hdmap = request_hdmap
        self.request_bev = request_bev
        self.text_prompt = text_prompt
        self.defer_frame_processing = defer_frame_processing
        self.channel = grpc.insecure_channel(
            server_address,
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
            ],
        )
        self.stub = video_model_pb2_grpc.WorldModelServiceStub(self.channel)

        # Session ID remapping: origin_id -> live_id
        self.session_map: dict[str, str] = {}

        # Collected statistics
        self.stats: list[ReplayStats] = []

        # Collected frame pairs for comparison (if collect_frames=True)
        self.frame_pairs: list[FramePair] = []
        self.hdmap_frame_pairs: list[FramePair] = []

        # BEV frames (no origin to compare, just collect replay frames)
        self.bev_frames: list[np.ndarray] = []

        # Session-level flags (set by start_session replay)
        self.return_hdmap_frames_flag: bool = False
        self.return_bev_map_flag: bool = False

        # Collected raw output pairs (processed after replay completes)
        self.output_pairs: list[OutputPair] = []

    def wait_for_server_ready(self, timeout_s: float) -> None:
        """Block until the gRPC channel reaches READY state.

        Args:
            timeout_s: Maximum time to wait for connectivity.

        Raises:
            TimeoutError: If server is not reachable before timeout.
        """
        logger.info(
            f"Waiting for gRPC server to become ready: {self.server_address} (timeout={timeout_s:.1f}s)"
        )
        try:
            grpc.channel_ready_future(self.channel).result(timeout=timeout_s)
        except grpc.FutureTimeoutError as e:
            raise TimeoutError(
                f"Timed out waiting for server readiness at {self.server_address} after {timeout_s:.1f}s"
            ) from e
        logger.info("Server channel is READY")

    def replay(self, recording_path: Path) -> list[ReplayStats]:
        """Replay all entries from a recording file.

        Requests are sent sequentially, waiting for each response before
        sending the next request. This is required because the order matters
        for correct simulation—concurrent requests would result in incorrect state.

        Args:
            recording_path: Path to the recording file.

        Returns:
            List of ReplayStats for each replayed entry.
        """
        logger.info(f"Replaying recording: {recording_path}")
        logger.info(f"Server: {self.server_address}")

        for entry in read_log_entries(recording_path, raise_on_malformed=True):
            # Replay the entry (sequential - wait for response before next request)
            stat = self._replay_entry(entry)
            self.stats.append(stat)

            # Log progress
            status = "OK" if stat.success else f"FAIL: {stat.error}"
            logger.info(
                f"[{stat.seq}] {stat.method}: "
                f"origin={stat.origin_duration_ns / 1e6:.1f}ms, "
                f"replay={stat.replay_duration_ns / 1e6:.1f}ms, "
                f"ratio={stat.ratio:.2f}x [{status}]"
            )

        # Process collected outputs into frame pairs (only when deferred processing is enabled)
        if self.collect_frames and self.defer_frame_processing and self.output_pairs:
            self._process_output_pairs()

        return self.stats

    def _process_single_output_pair(
        self,
        outputs_orig: list[video_model_pb2.CameraOutput],
        outputs_replay: list[video_model_pb2.CameraOutput],
    ) -> None:
        """Process a single pair of camera outputs into frame pairs.

        Args:
            outputs_orig: Original camera outputs from recorded response.
            outputs_replay: Replay camera outputs from live server response.
        """
        # RGB comparison (mosaic of all cameras)
        origin_mosaics = camera_outputs_to_mosaic_frames(
            outputs_orig, which_stream="rgb"
        )
        replay_mosaics = camera_outputs_to_mosaic_frames(
            outputs_replay, which_stream="rgb"
        )
        num_frames = min(len(origin_mosaics), len(replay_mosaics))
        for i in range(num_frames):
            self.frame_pairs.append(
                FramePair(origin=origin_mosaics[i], replay=replay_mosaics[i])
            )

        # HDMap condition comparison (only when return_hdmap_frames was requested)
        if self.return_hdmap_frames_flag:
            orig_hdmap = camera_outputs_to_mosaic_frames(
                outputs_orig, which_stream="hdmap"
            )
            replay_hdmap = camera_outputs_to_mosaic_frames(
                outputs_replay, which_stream="hdmap"
            )
            num_hdmap = min(len(orig_hdmap), len(replay_hdmap))
            for i in range(num_hdmap):
                self.hdmap_frame_pairs.append(
                    FramePair(origin=orig_hdmap[i], replay=replay_hdmap[i])
                )

    def _process_output_pairs(self) -> None:
        """Process collected output pairs into frame pairs for comparison video.

        Converts raw camera outputs into mosaic frames for RGB and HDMap comparison.
        Called after all replay entries have been processed.
        """
        logger.info(
            f"Processing {len(self.output_pairs)} output pairs into frame pairs..."
        )

        for output_pair in self.output_pairs:
            self._process_single_output_pair(output_pair.origin, output_pair.replay)

        logger.info(
            f"Processed into {len(self.frame_pairs)} RGB frame pairs, {len(self.hdmap_frame_pairs)} HDMap pairs"
        )

    def _replay_entry(self, entry: video_model_pb2.LogEntry) -> ReplayStats:
        """Replay a single log entry.

        Args:
            entry: The LogEntry to replay.

        Returns:
            ReplayStats with timing and success information.
        """
        # Determine which call variant is set
        call_type = entry.WhichOneof("call")
        if call_type is None:
            raise ValueError("LogEntry has no call set")

        try:
            if call_type == "start_session":
                duration_ns = self._replay_start_session(entry.start_session)
            elif call_type == "render_video_chunk":
                duration_ns = self._replay_render_video_chunk(entry.render_video_chunk)
            elif call_type == "close_session":
                duration_ns = self._replay_close_session(entry.close_session)
            else:
                raise ValueError(f"Unknown call type: {call_type}")

            return ReplayStats(
                seq=entry.seq,
                method=call_type,
                origin_duration_ns=entry.duration_ns,
                replay_duration_ns=duration_ns,
                success=True,
            )
        except Exception as e:
            return ReplayStats(
                seq=entry.seq,
                method=call_type,
                origin_duration_ns=entry.duration_ns,
                replay_duration_ns=-1,  # Not available
                success=False,
                error=str(e),
            )

    def _replay_start_session(self, call: video_model_pb2.StartSessionEntry) -> int:
        """Replay a start_session call, storing the session ID mapping.

        Args:
            call: StartSessionEntry containing the request/response pair.
        """
        # Create a copy of the request to optionally modify BEV settings
        request = video_model_pb2.SessionRequest()
        request.CopyFrom(call.request)

        # Optionally enable BEV rendering
        if self.request_bev:
            request.debug_options.return_bev_map = True
            logger.info("BEV rendering requested (using server defaults)")
        if self.request_hdmap:
            request.debug_options.return_hdmap_frames = True
            logger.info("HDMap rendering requested (using server defaults)")
        if self.text_prompt != "":
            request.text_prompt.positive = self.text_prompt

        start_ns = time.time_ns()
        response = self.stub.start_session(request, wait_for_ready=True)
        duration_ns = time.time_ns() - start_ns

        # Store mapping from origin session ID to new session ID
        origin_id = call.response.session_id
        self.session_map[origin_id] = response.session_id
        logger.debug(
            f"Session mapped: {origin_id[:8]}... -> {response.session_id[:8]}..."
        )

        # Store whether return_hdmap_frames and return_bev_map are set
        self.return_hdmap_frames_flag = request.debug_options.return_hdmap_frames
        self.return_bev_map_flag = request.debug_options.return_bev_map

        return duration_ns

    def _replay_close_session(self, call: video_model_pb2.CloseSessionEntry) -> int:
        """Replay a close_session call.

        Args:
            call: CloseSessionEntry containing the request/response pair.
        """
        origin_id = call.request.session_id
        if origin_id not in self.session_map:
            raise ValueError(f"Unknown session ID: {origin_id} (no mapping found)")

        remapped_request = video_model_pb2.SessionCloseRequest()
        remapped_request.CopyFrom(call.request)
        remapped_request.session_id = self.session_map[origin_id]

        start_ns = time.time_ns()
        _ = self.stub.close_session(remapped_request, wait_for_ready=True)
        duration_ns = time.time_ns() - start_ns
        return duration_ns

    def _replay_render_video_chunk(
        self, call: video_model_pb2.RenderVideoChunkEntry
    ) -> int:
        """Replay a render_video_chunk call with remapped session ID.

        Args:
            call: RenderVideoChunkEntry containing the request/response pair.

        Raises:
            ValueError: If the session ID is not in the mapping.
        """
        origin_id = call.request.session_id.session_id

        if origin_id not in self.session_map:
            raise ValueError(f"Unknown session ID: {origin_id} (no mapping found)")

        # Create a copy of the request with the remapped session ID
        remapped_request = video_model_pb2.VideoChunkRequest()
        remapped_request.CopyFrom(call.request)
        remapped_request.session_id.session_id = self.session_map[origin_id]

        start_ns = time.time_ns()
        repl_response: video_model_pb2.VideoChunkReturn = self.stub.render_video_chunk(
            remapped_request,
            wait_for_ready=True,
        )
        if self.mock_burden_ms > 0.0:
            time.sleep(self.mock_burden_ms / 1000.0)  # sleep for mock burden time
        duration_ns = time.time_ns() - start_ns

        # Collect frame pairs for comparison video (mosaic = all cameras stacked per frame)
        if self.collect_frames:
            outputs_orig = list(call.response.camera_outputs)
            outputs_replay = list(repl_response.camera_outputs)

            if self.defer_frame_processing:
                # Defer processing: collect raw outputs (processed after all replay is done)
                self.output_pairs.append(
                    OutputPair(origin=outputs_orig, replay=outputs_replay)
                )
            else:
                # Immediate processing (original behavior)
                self._process_single_output_pair(outputs_orig, outputs_replay)

            # BEV map frames (no origin to compare, just collect from replay response)
            if self.return_bev_map_flag:
                bev_frames_replay = list(repl_response.bev_map_frames)
                for bev_img in bev_frames_replay:
                    self.bev_frames.append(decode_image_proto(bev_img))

        return duration_ns

    def print_summary(self) -> None:
        """Print a summary of replay performance."""
        print("\n" + "=" * 70)
        print("REPLAY SUMMARY")
        print("=" * 70)

        if not self.stats:
            print("No entries replayed.")
            return

        successful = [s for s in self.stats if s.success]
        failed = [s for s in self.stats if not s.success]

        print(f"Total entries:  {len(self.stats)}")
        print(f"Successful:     {len(successful)}")
        print(f"Failed:         {len(failed)}")

        if successful:
            total_origin_ns = sum(s.origin_duration_ns for s in successful)
            total_replay_ns = sum(s.replay_duration_ns for s in successful)

            print("\nTiming (successful entries):")
            print(f"  origin total:  {total_origin_ns / 1e9:.2f}s")
            print(f"  Replay total:    {total_replay_ns / 1e9:.2f}s")
            print(f"  Overall ratio:   {total_replay_ns / total_origin_ns:.2f}x")

            # Per-method breakdown
            methods = sorted(set(s.method for s in successful))
            for method in methods:
                method_stats = [s for s in successful if s.method == method]
                if method_stats:
                    ratios = [s.ratio for s in method_stats]
                    origin_total = sum(s.origin_duration_ns for s in method_stats)
                    replay_total = sum(s.replay_duration_ns for s in method_stats)

                    print(f"\n  {method}:")
                    print(f"    Count:       {len(method_stats)}")
                    print(f"    origin total:  {origin_total / 1e6:.1f}ms")
                    print(f"    Replay total: {replay_total / 1e6:.1f}ms")
                    print(f"    Avg ratio:   {sum(ratios) / len(ratios):.2f}x")
                    print(f"    Min ratio:   {min(ratios):.2f}x")
                    print(f"    Max ratio:   {max(ratios):.2f}x")

        if failed:
            print("\nFailed entries:")
            for s in failed:
                print(f"  [{s.seq}] {s.method}: {s.error}")

        print("=" * 70)

    def _write_frame_pairs_video(
        self,
        frame_pairs: list[FramePair],
        output_path: Path,
        fps: int,
        label: str = "comparison",
    ) -> None:
        """Write a comparison video (origin on top, replay on bottom) from frame pairs.

        Handles resolution mismatches between origin and replay frames by padding
        to the maximum dimensions encountered.
        """
        if not frame_pairs:
            logger.warning(f"No {label} frame pairs collected. Skipping {output_path}.")
            return

        logger.info(
            f"Writing {label} video with {len(frame_pairs)} frames to {output_path}"
        )

        first_pair = frame_pairs[0]
        h = max(first_pair.origin.shape[0], first_pair.replay.shape[0])
        w = max(first_pair.origin.shape[1], first_pair.replay.shape[1])

        def _pad_to_same_size(img: np.ndarray, width: int, height: int) -> np.ndarray:
            if height != img.shape[0] or width != img.shape[1]:
                return np.array(
                    ImageOps.pad(Image.fromarray(img), (width, height), color=(0, 0, 0))
                )
            return img

        frames = []
        for i, pair in enumerate(frame_pairs):
            origin_res = "x".join(map(str, pair.origin.shape[:2]))
            replay_res = "x".join(map(str, pair.replay.shape[:2]))

            origin = _pad_to_same_size(pair.origin, width=w, height=h)
            replay = _pad_to_same_size(pair.replay, width=w, height=h)

            # Stack vertically: origin on top, replay on bottom
            if not self.disable_origin:
                combined = np.vstack([origin, replay])
            else:
                combined = replay

            if not self.disable_font:
                img = Image.fromarray(combined)
                # Add labels using PIL
                draw = ImageDraw.Draw(img)
                font_size = 48 if w > 1024 else 32
                font = ImageFont.load_default(size=font_size)

                # Draw text with shadow for visibility
                labels = []
                if not self.disable_origin:
                    labels.append((f"Origin {origin_res}", font_size))
                    labels.append((f"Replay {replay_res}", h + font_size))
                else:
                    labels.append((f"Replay {replay_res}", font_size))
                for text_label, y_offset in labels:
                    draw.text(
                        (w / 2 + 3, y_offset + 3),
                        text_label,
                        font=font,
                        fill=(0, 0, 0),
                        anchor="mm",
                    )
                    draw.text(
                        (w / 2 + 0, y_offset + 0),
                        text_label,
                        font=font,
                        fill=(255, 255, 255),
                        anchor="mm",
                    )

                frame = np.array(img)
            else:
                frame = combined

            frames.append(frame)
            if (i + 1) % 100 == 0:
                logger.info(f"  Prepared {i + 1}/{len(frame_pairs)} frames")

        logger.info("Encoding video with H.264...")
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
        logger.info(f"Comparison video saved: {output_path}")

    def write_comparison_video(self, output_path: Path, fps: int = 16) -> None:
        """Write RGB comparison video (original on top, replay on bottom)."""
        self._write_frame_pairs_video(
            self.frame_pairs, output_path, fps, label="RGB comparison"
        )

    def write_hdmap_comparison_video(self, output_path: Path, fps: int = 16) -> None:
        """Write HDMap condition comparison video (original on top, replay on bottom)."""
        self._write_frame_pairs_video(
            self.hdmap_frame_pairs, output_path, fps, label="HDMap comparison"
        )

    def write_bev_video(self, output_path: Path, fps: int = 16) -> None:
        """Write BEV map frames to video (no comparison, just the generated frames)."""
        if not self.bev_frames:
            logger.warning(f"No BEV frames collected. Skipping {output_path}.")
            return

        logger.info(
            f"Writing BEV video with {len(self.bev_frames)} frames to {output_path}"
        )

        logger.info("Encoding video with H.264...")
        iio.imwrite(
            output_path,
            self.bev_frames,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
        logger.info(f"BEV video saved: {output_path}")


def show_recording_info(recording_path: Path) -> None:
    """Display information about a recording file without replaying it.

    Args:
        recording_path: Path to the recording file.
    """
    print(f"\nRecording: {recording_path}")
    print(f"File size: {recording_path.stat().st_size / 1024:.1f} KB")

    entry_count = count_log_entries(recording_path)
    print(f"Total entries: {entry_count}")

    # Read and summarize entries
    start_sessions = 0
    video_chunks = 0
    total_duration_ns = 0
    session_ids: set[str] = set()

    for entry in read_log_entries(recording_path):
        total_duration_ns += entry.duration_ns
        call_type = entry.WhichOneof("call")

        if call_type == "start_session":
            start_sessions += 1
            session_ids.add(entry.start_session.response.session_id)
        elif call_type == "render_video_chunk":
            video_chunks += 1

    print("\nBreakdown:")
    print(f"  start_session calls:       {start_sessions}")
    print(f"  render_video_chunk calls:  {video_chunks}")
    print(f"  Unique sessions:           {len(session_ids)}")
    print(f"  Total recorded duration:   {total_duration_ns / 1e9:.2f}s")


def main() -> None:
    """Main entry point for the replay client."""
    parser = argparse.ArgumentParser(
        description="Replay recorded gRPC sessions for performance testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay session for performance testing
  python replay_client.py session.binlog --server localhost:50051

  # Show recording info without replaying
  python replay_client.py session.binlog --info

  # Generate comparison video (origin vs replay)
  python replay_client.py session.binlog --server localhost:50051 --compare-video comparison.mp4

  # Also generate HDMap render comparison
  python replay_client.py session.binlog --server localhost:50051 --compare-video out.mp4 --compare-video-hdmap out_hdmap.mp4

  # Request BEV rendering and save to video
  python replay_client.py session.binlog --server localhost:50051 --compare-video-bev bev.mp4
""",
    )
    parser.add_argument(
        "recording",
        type=Path,
        help="Path to the recording file (.binlog)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="Server address in host:port format (default: localhost:50051)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show recording info without replaying",
    )
    parser.add_argument(
        "--compare-video",
        type=Path,
        metavar="PATH",
        help="Generate RGB comparison video (origin on top, replay on bottom)",
    )
    parser.add_argument(
        "--compare-video-hdmap",
        type=Path,
        metavar="PATH",
        help="Generate HDMap condition comparison video (origin vs replay)",
    )
    parser.add_argument(
        "--compare-video-bev",
        type=Path,
        metavar="PATH",
        help="Generate BEV (Bird's Eye View) map video from server-rendered frames",
    )
    parser.add_argument(
        "--request-hdmap",
        action="store_true",
        help="Request HDMap rendering from the server (enables return_hdmap_frames in debug_options)",
    )
    parser.add_argument(
        "--request-bev",
        action="store_true",
        help="Request BEV rendering from the server (enables return_bev_map in debug_options)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for comparison video(s) (default: 30)",
    )
    parser.add_argument(
        "--disable-font",
        action="store_true",
        help="Disable font in the comparison video. Default: False.",
    )
    parser.add_argument(
        "--disable-origin",
        action="store_true",
        help="Disable origin in the comparison video. Default: False.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="Max seconds to wait for server channel readiness before replay (default: 600)",
    )
    parser.add_argument(
        "--mock-burden-ms",
        type=float,
        default=0.0,
        metavar="MILLISECONDS",
        help="Mock burden time in milliseconds. Default: 0.0",
    )
    parser.add_argument(
        "--defer-frame-processing",
        action="store_true",
        help="Defer frame processing until after all replay is done. Default: False.",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="",
        help="Text prompt for generation. Default: empty string.",
    )
    args = parser.parse_args()

    if not args.recording.exists():
        logger.error(f"Recording file not found: {args.recording}")
        return

    if args.info:
        show_recording_info(args.recording)
        return

    collect_frames = (
        args.compare_video is not None
        or args.compare_video_hdmap is not None
        or args.compare_video_bev is not None
    )

    # Auto-enable request_bev if compare-video-bev is requested
    request_bev = args.request_bev or args.compare_video_bev is not None
    request_hdmap = args.request_hdmap or args.compare_video_hdmap is not None

    client = ReplayClient(
        args.server,
        collect_frames=collect_frames,
        disable_font=args.disable_font,
        disable_origin=args.disable_origin,
        mock_burden_ms=args.mock_burden_ms,
        request_bev=request_bev,
        request_hdmap=request_hdmap,
        text_prompt=args.text_prompt,
        defer_frame_processing=args.defer_frame_processing,
    )
    client.wait_for_server_ready(timeout_s=args.wait_timeout)
    client.replay(args.recording)
    client.print_summary()

    if args.compare_video:
        client.write_comparison_video(args.compare_video, fps=args.fps)
    if args.compare_video_hdmap:
        client.write_hdmap_comparison_video(args.compare_video_hdmap, fps=args.fps)
    if args.compare_video_bev:
        client.write_bev_video(args.compare_video_bev, fps=args.fps)


if __name__ == "__main__":
    main()
