#!/usr/bin/env python3
"""Parse interactive-drive logs and extract FPS metrics."""
import re
import sys
from collections import defaultdict
from pathlib import Path

def analyze_log(log_path):
    if not Path(log_path).exists():
        print(f"ERROR: Log file not found: {log_path}")
        return

    chunk_timings = []
    model_times = []

    with open(log_path) as f:
        for line in f:
            if "[world-model] next_chunk" in line:
                match = re.search(r"total_ms=(\d+\.?\d*)", line)
                if match:
                    total_ms = float(match.group(1))
                    chunk_timings.append(total_ms)

                match = re.search(r"model_ms=(\d+\.?\d*)", line)
                if match:
                    model_ms = float(match.group(1))
                    model_times.append(model_ms)

    if not chunk_timings:
        print("No chunk timings found in log")
        return

    # Calculate FPS (frames per 1000ms / total_ms * num_frames_per_block)
    fps_per_chunk = [1000.0 / (t / 8) for t in chunk_timings]  # 8 frames per block
    avg_fps = sum(fps_per_chunk) / len(fps_per_chunk)
    avg_chunk_ms = sum(chunk_timings) / len(chunk_timings)
    avg_model_ms = sum(model_times) / len(model_times) if model_times else 0

    print("\n" + "="*60)
    print("INTERACTIVE-DRIVE PERFORMANCE METRICS")
    print("="*60)
    print(f"Total chunks analyzed: {len(chunk_timings)}")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average chunk time: {avg_chunk_ms:.1f}ms")
    print(f"Average model time: {avg_model_ms:.1f}ms")
    print(f"Min FPS: {min(fps_per_chunk):.1f}")
    print(f"Max FPS: {max(fps_per_chunk):.1f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    log_path = r"C:\tmp\idrive_perf.log"
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    analyze_log(log_path)
