# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output sink persisting named tensor artifacts as NumPy arrays."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from torch import Tensor

from flashdreams.api_v2.output_sink import OutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.tensor_artifact import TensorArtifactSchema


class TensorArtifactOutputSink(OutputSink):
    """Collect declared tensor outputs and atomically write one ``.npy`` per name."""

    def __init__(self, output_dir: str | Path) -> None:
        """
        Args:
            output_dir: Directory receiving ``<artifact-name>.npy`` files.
        """
        self._output_dir = Path(output_dir)
        self._schemas: dict[str, TensorArtifactSchema] | None = None
        self._chunks: dict[str, list[Tensor]] = {}

    def open(self, session_desc: SessionDesc) -> None:
        """Prepare to collect artifacts declared by ``session_desc``."""
        self._schemas = {
            schema.name: schema for schema in session_desc.tensor_artifact_schemas
        }
        self._chunks = {name: [] for name in self._schemas}

    def write(self, result: StepResult) -> None:
        """Collect the tensor artifacts carried by one model result.

        Raises:
            RuntimeError: Called before :meth:`open`.
            ValueError: A result is undeclared, duplicated, or changes schema.
        """
        if self._schemas is None:
            raise RuntimeError(
                "TensorArtifactOutputSink.open() must run before write()."
            )
        seen: set[str] = set()
        for artifact in result.tensor_artifacts:
            name = artifact.schema.name
            if name in seen:
                raise ValueError(
                    f"Step {result.step_index} emitted tensor artifact {name!r} twice."
                )
            seen.add(name)
            expected = self._schemas.get(name)
            if expected is None:
                raise ValueError(
                    f"Tensor artifact {name!r} was not declared by the session."
                )
            if artifact.schema != expected:
                raise ValueError(
                    f"Tensor artifact {name!r} does not match its session schema."
                )
            self._chunks[name].append(artifact.tensor.detach().to("cpu").contiguous())

    def close(self) -> None:
        """Write collected tensors and release buffered state.

        Can be called before :meth:`open` or more than once.
        """
        schemas = self._schemas
        if schemas is None:
            return
        chunks = self._chunks
        self._schemas = None
        self._chunks = {}

        outputs: dict[str, Tensor] = {}
        for name, artifact_chunks in chunks.items():
            if not artifact_chunks:
                continue
            schema = schemas[name]
            if schema.concatenate_axis is None:
                if len(artifact_chunks) != 1:
                    raise ValueError(
                        f"Tensor artifact {name!r} permits one output but received "
                        f"{len(artifact_chunks)}."
                    )
                outputs[name] = artifact_chunks[0]
            else:
                outputs[name] = torch.cat(artifact_chunks, dim=schema.concatenate_axis)
        if not outputs:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path]] = []
        try:
            for name, tensor in outputs.items():
                destination = self._output_dir / f"{name}.npy"
                with tempfile.NamedTemporaryFile(
                    mode="w+b",
                    prefix=f".{name}.",
                    suffix=".tmp",
                    dir=self._output_dir,
                    delete=False,
                ) as temporary:
                    _write_numpy(temporary, tensor)
                    temporary_path = Path(temporary.name)
                staged.append((temporary_path, destination))
            for temporary_path, destination in staged:
                os.replace(temporary_path, destination)
        finally:
            for temporary_path, _ in staged:
                temporary_path.unlink(missing_ok=True)


def _write_numpy(file: BinaryIO, tensor: Tensor) -> None:
    """Write ``tensor`` to an open temporary file and flush it to disk."""
    np.save(file, tensor.numpy(), allow_pickle=False)
    file.flush()
    os.fsync(file.fileno())


__all__ = ["TensorArtifactOutputSink"]
