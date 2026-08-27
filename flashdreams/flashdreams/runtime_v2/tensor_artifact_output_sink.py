# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-output sink persisting named tensor artifacts as NumPy arrays."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import IO

import numpy as np
import torch
from torch import Tensor

from flashdreams.runtime_v2.model_output_sink import ModelOutputSink
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.tensor_artifact import TensorArtifactSchema

_MANIFEST_FILENAME = "tensor_artifacts.json"
"""Commit marker and machine-readable schema for persisted tensor artifacts."""

_ARTIFACT_TYPE = "flashdreams.runtime_v2.tensor_artifacts"
_SCHEMA_VERSION = 1


class TensorArtifactOutputSink(ModelOutputSink):
    """Transactionally persist the latest generation as NumPy arrays."""

    def __init__(self, output_dir: str | Path) -> None:
        """
        Args:
            output_dir: Directory receiving ``<artifact-name>.npy`` files and
                their ``tensor_artifacts.json`` manifest.
        """
        self._output_dir = Path(output_dir)
        self._schemas: dict[str, TensorArtifactSchema] | None = None
        self._generation: int | None = None
        self._chunks: dict[str, list[Tensor]] = {}

    def open(self, session_desc: SessionDesc) -> None:
        """Prepare to collect artifacts declared by ``session_desc``.

        The output directory and an incomplete commit manifest are created
        here so invalid or unwritable destinations fail before generation.

        Raises:
            ValueError: The session declares no tensor artifacts.
        """
        schemas = {
            schema.name: schema for schema in session_desc.tensor_artifact_schemas
        }
        if not schemas:
            raise ValueError(
                "TensorArtifactOutputSink requires a session that declares at least "
                "one tensor artifact."
            )

        self._output_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest_atomically(
            self._output_dir,
            _manifest_payload(schemas, {}, generation=None, complete=False),
        )
        self._schemas = schemas
        self._generation = None
        self._chunks = {name: [] for name in schemas}

    def write(self, generation: int, results: Sequence[StepResult]) -> None:
        """Collect one complete model-step result batch.

        A newer generation discards all buffered older-generation chunks. A
        late batch from an older generation is ignored.

        Raises:
            RuntimeError: Called before :meth:`open`.
            ValueError: An output is undeclared, duplicated across the result
                batch, changes schema, cannot be represented by NumPy, or is
                incompatible with an earlier chunk.
        """
        schemas = self._schemas
        if schemas is None:
            raise RuntimeError(
                "TensorArtifactOutputSink.open() must run before write()."
            )
        if self._generation is not None and generation < self._generation:
            return
        if self._generation is None or generation > self._generation:
            self._generation = generation
            self._chunks = {name: [] for name in schemas}

        seen: set[str] = set()
        staged: list[tuple[str, Tensor]] = []
        for result in results:
            for artifact in result.tensor_artifacts:
                name = artifact.schema.name
                if name in seen:
                    raise ValueError(
                        f"Model step {result.step_index} emitted tensor artifact "
                        f"{name!r} more than once across its output channels."
                    )
                seen.add(name)
                expected = schemas.get(name)
                if expected is None:
                    raise ValueError(
                        f"Tensor artifact {name!r} was not declared by the session."
                    )
                if artifact.schema != expected:
                    raise ValueError(
                        f"Tensor artifact {name!r} does not match its session schema."
                    )
                tensor = _numpy_compatible_tensor(name, artifact.tensor)
                self._validate_next_chunk(expected, tensor)
                staged.append((name, tensor))

        for name, tensor in staged:
            self._chunks[name].append(tensor)

    def _validate_next_chunk(
        self, schema: TensorArtifactSchema, tensor: Tensor
    ) -> None:
        """Validate ``tensor`` against chunks already buffered for ``schema``."""
        chunks = self._chunks[schema.name]
        if not chunks:
            return
        if schema.concatenate_axis is None:
            raise ValueError(
                f"Tensor artifact {schema.name!r} permits at most one output "
                "per generation."
            )
        first = chunks[0]
        if tensor.dtype != first.dtype:
            raise ValueError(
                f"Tensor artifact {schema.name!r} changed dtype from "
                f"{first.dtype} to {tensor.dtype}."
            )
        axis = schema.concatenate_axis % tensor.ndim
        mismatches = [
            dimension
            for dimension, (expected, received) in enumerate(
                zip(first.shape, tensor.shape, strict=True)
            )
            if dimension != axis and expected != received
        ]
        if mismatches:
            raise ValueError(
                f"Tensor artifact {schema.name!r} changed non-concatenated "
                f"dimensions from {tuple(first.shape)} to {tuple(tensor.shape)}."
            )

    def close(self, *, commit: bool = True) -> None:
        """Persist buffered tensors from a successful run and release state.

        Can be called before :meth:`open` or more than once. When ``commit`` is
        false, buffered tensors are discarded and the incomplete manifest
        written by :meth:`open` remains the directory's commit marker.
        """
        schemas = self._schemas
        if schemas is None:
            return
        chunks = self._chunks
        generation = self._generation
        self._schemas = None
        self._generation = None
        self._chunks = {}

        if not commit:
            return

        outputs: dict[str, Tensor] = {}
        for name, artifact_chunks in chunks.items():
            if not artifact_chunks:
                continue
            schema = schemas[name]
            outputs[name] = (
                artifact_chunks[0]
                if schema.concatenate_axis is None
                else torch.cat(artifact_chunks, dim=schema.concatenate_axis)
            )
        staged: list[tuple[Path, Path]] = []
        manifest_temporary_path: Path | None = None
        backups: dict[Path, Path | None] = {}
        try:
            for name, tensor in outputs.items():
                destination = self._output_dir / f"{name}.npy"
                temporary_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w+b",
                        prefix=f".{name}.",
                        suffix=".tmp",
                        dir=self._output_dir,
                        delete=False,
                    ) as temporary:
                        temporary_path = Path(temporary.name)
                        _write_numpy(temporary.file, tensor)
                except BaseException:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
                    raise
                assert temporary_path is not None
                staged.append((temporary_path, destination))

            manifest_temporary_path = _stage_manifest(
                self._output_dir,
                _manifest_payload(
                    schemas, outputs, generation=generation, complete=True
                ),
            )
            for name in schemas:
                destination = self._output_dir / f"{name}.npy"
                backups[destination] = _stage_backup(destination)
            try:
                for temporary_path, destination in staged:
                    os.replace(temporary_path, destination)
                for name in schemas.keys() - outputs.keys():
                    (self._output_dir / f"{name}.npy").unlink(missing_ok=True)
                os.replace(
                    manifest_temporary_path,
                    self._output_dir / _MANIFEST_FILENAME,
                )
            except BaseException as error:
                try:
                    _restore_artifacts(backups)
                except BaseException as rollback_error:
                    raise error from rollback_error
                raise
        finally:
            for temporary_path, _ in staged:
                temporary_path.unlink(missing_ok=True)
            if manifest_temporary_path is not None:
                manifest_temporary_path.unlink(missing_ok=True)
            for backup in backups.values():
                if backup is not None:
                    backup.unlink(missing_ok=True)


def _numpy_compatible_tensor(name: str, tensor: Tensor) -> Tensor:
    """Detach ``tensor`` onto CPU and prove NumPy can represent its dtype."""
    try:
        result = tensor.detach().to("cpu").contiguous()
        result.numpy()
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            f"Tensor artifact {name!r} with dtype {tensor.dtype} cannot be "
            "stored as a NumPy array."
        ) from error
    return result


def _manifest_payload(
    schemas: dict[str, TensorArtifactSchema],
    outputs: dict[str, Tensor],
    *,
    generation: int | None,
    complete: bool,
) -> dict[str, object]:
    """Describe declared and emitted artifacts for a run."""
    artifacts: list[dict[str, object]] = []
    for name, schema in schemas.items():
        tensor = outputs.get(name)
        artifacts.append(
            {
                "name": name,
                "path": f"{name}.npy" if tensor is not None else None,
                "emitted": tensor is not None,
                "dimension_names": list(schema.dimension_names),
                "concatenate_axis": schema.concatenate_axis,
                "dtype": str(tensor.numpy().dtype) if tensor is not None else None,
                "shape": (
                    [int(dimension) for dimension in tensor.shape]
                    if tensor is not None
                    else None
                ),
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "complete": complete,
        "generation": generation,
        "artifacts": artifacts,
    }


def _write_manifest_atomically(output_dir: Path, payload: dict[str, object]) -> None:
    """Publish ``payload`` through an atomic replace."""
    temporary_path = _stage_manifest(output_dir, payload)
    try:
        os.replace(temporary_path, output_dir / _MANIFEST_FILENAME)
    finally:
        temporary_path.unlink(missing_ok=True)


def _stage_manifest(output_dir: Path, payload: dict[str, object]) -> Path:
    """Write and flush a manifest temporary file in ``output_dir``."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".tensor-artifacts.",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            _write_json(temporary.file, payload)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    assert temporary_path is not None
    return temporary_path


def _stage_backup(destination: Path) -> Path | None:
    """Copy an existing artifact to a flushed temporary backup."""
    if not destination.exists():
        return None
    temporary_path: Path | None = None
    try:
        with destination.open("rb") as source:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{destination.name}.backup.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                shutil.copyfileobj(source, temporary.file)
                temporary.file.flush()
                os.fsync(temporary.file.fileno())
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    assert temporary_path is not None
    return temporary_path


def _restore_artifacts(backups: dict[Path, Path | None]) -> None:
    """Restore artifact destinations to their state before publication."""
    for destination, backup in backups.items():
        if backup is None:
            destination.unlink(missing_ok=True)
        else:
            os.replace(backup, destination)


def _write_numpy(file: IO[bytes], tensor: Tensor) -> None:
    """Write ``tensor`` to an open temporary file and flush it to disk."""
    np.save(file, tensor.numpy(), allow_pickle=False)
    file.flush()
    os.fsync(file.fileno())


def _write_json(file: IO[bytes], payload: dict[str, object]) -> None:
    """Write deterministic JSON to an open file and flush it to disk."""
    file.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    file.flush()
    os.fsync(file.fileno())


__all__ = ["TensorArtifactOutputSink"]
