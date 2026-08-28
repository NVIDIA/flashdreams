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

"""Inspect persisted LingBot-VA Robotwin action artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

ROBOTWIN_ACTION_CHANNEL_NAMES = (
    "left_delta_x_m",
    "left_delta_y_m",
    "left_delta_z_m",
    "left_relative_qx",
    "left_relative_qy",
    "left_relative_qz",
    "left_relative_qw",
    "left_gripper",
    "right_delta_x_m",
    "right_delta_y_m",
    "right_delta_z_m",
    "right_relative_qx",
    "right_relative_qy",
    "right_relative_qz",
    "right_relative_qw",
    "right_gripper",
)
"""Column names for the 16 action channels selected by the Robotwin config."""

_ARTIFACT_TYPE = "flashdreams.runtime_v2.tensor_artifacts"
_MANIFEST_FILENAME = "tensor_artifacts.json"


def load_action_artifact(source: str | Path) -> FloatArray:
    """Load and validate one LingBot-VA action array.

    A directory must contain a committed FlashDreams tensor-artifact manifest.
    A direct ``.npy`` path is also accepted for older validation outputs created
    before the manifest sink was available.

    Args:
        source: Artifact directory or direct ``actions.npy`` path.

    Returns:
        A finite, non-empty float32 array with shape ``[step, 16]``.

    Raises:
        FileNotFoundError: A required manifest or array is missing.
        ValueError: The manifest or action tensor violates the expected schema.
    """
    source_path = Path(source)
    expected_shape: tuple[int, ...] | None = None
    expected_dtype: str | None = None
    if source_path.is_dir():
        array_path, expected_shape, expected_dtype = _artifact_path(source_path)
    else:
        array_path = source_path

    loaded = np.load(array_path, allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        raise ValueError(f"Action artifact is not a NumPy array: {array_path}")
    if expected_shape is not None and tuple(loaded.shape) != expected_shape:
        raise ValueError(
            f"Action array shape {tuple(loaded.shape)} does not match manifest "
            f"shape {expected_shape}."
        )
    if expected_dtype is not None and str(loaded.dtype) != expected_dtype:
        raise ValueError(
            f"Action array dtype {loaded.dtype} does not match manifest dtype "
            f"{expected_dtype}."
        )
    if loaded.dtype != np.float32:
        raise ValueError(f"Actions must have dtype float32; received {loaded.dtype}.")
    if loaded.ndim != 2 or loaded.shape[0] == 0 or loaded.shape[1] != 16:
        raise ValueError(
            "Actions must have non-empty shape [step, 16]; received "
            f"{tuple(loaded.shape)}."
        )
    if not np.isfinite(loaded).all():
        raise ValueError("Actions contain non-finite values.")
    return cast(FloatArray, loaded)


def write_action_csv(actions: FloatArray, output_path: str | Path) -> Path:
    """Write named Robotwin action channels as a CSV table.

    Args:
        actions: Validated action values with shape ``[step, 16]``.
        output_path: Destination CSV path.

    Returns:
        The resolved output path.
    """
    _validate_actions(actions)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("step", *ROBOTWIN_ACTION_CHANNEL_NAMES))
        for step, row in enumerate(actions):
            writer.writerow((step, *(float(value) for value in row)))
    return destination.resolve()


def write_action_plot(actions: FloatArray, output_path: str | Path) -> Path:
    """Plot the two-arm Robotwin trajectory channels for human inspection.

    The plot is diagnostic only: it does not apply a robot's initial pose,
    execute the actions, or establish task success or physical safety.

    Args:
        actions: Validated action values with shape ``[step, 16]``.
        output_path: Destination image path.

    Returns:
        The resolved output path.

    Raises:
        RuntimeError: Matplotlib is not installed.
    """
    _validate_actions(actions)
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Action plotting requires the 'visualization' package extra."
        ) from error

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(actions.shape[0])
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    panels = (
        (axes[0, 0], range(0, 3), "Left end-effector translation delta"),
        (axes[0, 1], range(3, 7), "Left relative quaternion"),
        (axes[1, 0], range(8, 11), "Right end-effector translation delta"),
        (axes[1, 1], range(11, 15), "Right relative quaternion"),
    )
    for axis, channel_ids, title in panels:
        for channel_id in channel_ids:
            axis.plot(
                steps,
                actions[:, channel_id],
                label=ROBOTWIN_ACTION_CHANNEL_NAMES[channel_id],
            )
        axis.set_title(title)
        axis.set_ylabel("predicted value")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")

    axes[0, 0].plot(
        steps,
        actions[:, 7],
        linestyle="--",
        label=ROBOTWIN_ACTION_CHANNEL_NAMES[7],
    )
    axes[1, 0].plot(
        steps,
        actions[:, 15],
        linestyle="--",
        label=ROBOTWIN_ACTION_CHANNEL_NAMES[15],
    )
    axes[0, 0].legend(fontsize="small")
    axes[1, 0].legend(fontsize="small")
    axes[1, 0].set_xlabel("action step")
    axes[1, 1].set_xlabel("action step")
    figure.suptitle("LingBot-VA Robotwin predicted actions")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination.resolve()


def _artifact_path(
    output_dir: Path,
) -> tuple[Path, tuple[int, ...], str]:
    """Resolve ``actions.npy`` from a complete tensor-artifact manifest."""
    manifest_path = output_dir / _MANIFEST_FILENAME
    with manifest_path.open(encoding="utf-8") as manifest_file:
        payload = json.load(manifest_file)
    if not isinstance(payload, Mapping):
        raise ValueError("Tensor artifact manifest must be a JSON object.")
    manifest = cast(Mapping[str, Any], payload)
    if manifest.get("artifact_type") != _ARTIFACT_TYPE:
        raise ValueError("Directory is not a FlashDreams tensor-artifact output.")
    if manifest.get("complete") is not True:
        raise ValueError("Tensor artifact output is incomplete and cannot be consumed.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Tensor artifact manifest has no artifact list.")
    action_entries = [
        cast(Mapping[str, Any], entry)
        for entry in artifacts
        if isinstance(entry, Mapping) and entry.get("name") == "actions"
    ]
    if len(action_entries) != 1:
        raise ValueError("Manifest must declare exactly one 'actions' artifact.")
    entry = action_entries[0]
    if entry.get("emitted") is not True:
        raise ValueError("Manifest declares 'actions' but no array was emitted.")
    if entry.get("dimension_names") != ["step", "channel"]:
        raise ValueError("Actions must use dimensions ['step', 'channel'].")
    relative_path = entry.get("path")
    shape = entry.get("shape")
    dtype = entry.get("dtype")
    if not isinstance(relative_path, str) or Path(relative_path).name != relative_path:
        raise ValueError("Actions manifest path must be a local filename.")
    if not (
        isinstance(shape, list)
        and len(shape) == 2
        and all(isinstance(value, int) for value in shape)
    ):
        raise ValueError("Actions manifest shape must contain two integer dimensions.")
    if not isinstance(dtype, str):
        raise ValueError("Actions manifest dtype is missing.")
    return output_dir / relative_path, tuple(shape), dtype


def _validate_actions(actions: FloatArray) -> None:
    """Validate programmatically supplied action data before export."""
    if actions.dtype != np.float32:
        raise ValueError(f"Actions must have dtype float32; received {actions.dtype}.")
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 16:
        raise ValueError(
            "Actions must have non-empty shape [step, 16]; received "
            f"{tuple(actions.shape)}."
        )
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain non-finite values.")


def _parse_args(commandline_args: Sequence[str] | None) -> argparse.Namespace:
    """Parse action inspector arguments."""
    parser = argparse.ArgumentParser(
        description="Plot or export LingBot-VA Robotwin action artifacts.",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Artifact output directory or direct actions.npy path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Plot destination (default: <source>/actions.png).",
    )
    parser.add_argument("--csv-output", type=Path, help="Optional CSV destination.")
    return parser.parse_args(commandline_args)


def main(commandline_args: Sequence[str] | None = None) -> None:
    """Run the LingBot-VA action artifact inspector."""
    args = _parse_args(commandline_args)
    source = cast(Path, args.source)
    default_parent = source if source.is_dir() else source.parent
    output = cast(Path | None, args.output) or default_parent / "actions.png"
    actions = load_action_artifact(source)
    plot_path = write_action_plot(actions, output)
    print(f"Wrote action plot: {plot_path}")
    csv_output = cast(Path | None, args.csv_output)
    if csv_output is not None:
        csv_path = write_action_csv(actions, csv_output)
        print(f"Wrote action CSV: {csv_path}")


if __name__ == "__main__":
    main()
