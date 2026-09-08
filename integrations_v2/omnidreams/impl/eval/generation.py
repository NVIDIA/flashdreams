# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashDreams generation orchestration for staged evaluation cases."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from omnidreams.impl.eval.manifest import StagedCase


@dataclass(frozen=True)
class GenerationResult:
    uuid: str
    output_dir: Path
    generated_video_path: Path
    log_path: Path
    command: tuple[str, ...]


def generate_cases(
    cases: Sequence[StagedCase],
    *,
    run_root: Path,
    recipe: str,
    total_blocks: int,
    flashdreams_run: str = "flashdreams-run",
    force: bool = False,
    dry_run: bool = False,
    stream_logs: bool = False,
) -> list[GenerationResult]:
    """Run Interactive Drive for each staged case."""

    results: list[GenerationResult] = []
    for case in cases:
        result = generation_result_for_case(
            case,
            run_root=run_root,
            recipe=recipe,
            total_blocks=total_blocks,
            flashdreams_run=flashdreams_run,
        )
        results.append(result)
        if result.generated_video_path.exists() and not force:
            metadata_path = result.generated_video_path.parent / "generation.json"
            if not metadata_path.exists():
                _write_generation_metadata(result)
            continue
        if dry_run:
            continue
        result.output_dir.mkdir(parents=True, exist_ok=True)
        _run_generation_command(result, stream_logs=stream_logs)
        _write_generation_metadata(result)
    return results


def generation_result_for_case(
    case: StagedCase,
    *,
    run_root: Path,
    recipe: str,
    total_blocks: int,
    flashdreams_run: str,
) -> GenerationResult:
    output_dir = run_root / "generated" / case.case.uuid / "runner"
    generated_video_path = run_root / "generated" / case.case.uuid / "generated.mp4"
    log_path = run_root / "generated" / case.case.uuid / "flashdreams-run.log"
    stats_path = output_dir / f"stats_{recipe}.json"
    command = (
        flashdreams_run,
        recipe,
        "--mode",
        "mp4",
        "--output-path",
        str(generated_video_path),
        "--stats-path",
        str(stats_path),
        "--backpressure-mode",
        "block",
        "--presentation-mode",
        "on_demand",
        "--",
        "--prompt",
        case.prompt_text,
        "--camera",
        case.case.camera,
        "--total-blocks",
        str(total_blocks),
        "--no-ui",
    )
    return GenerationResult(
        uuid=case.case.uuid,
        output_dir=output_dir,
        generated_video_path=generated_video_path,
        log_path=log_path,
        command=command,
    )


def _run_generation_command(result: GenerationResult, *, stream_logs: bool) -> None:
    if stream_logs:
        completed = subprocess.run(list(result.command), check=False)
        if completed.returncode:
            raise RuntimeError(
                f"flashdreams-run failed for {result.uuid} with exit code "
                f"{completed.returncode}; logs were streamed to the terminal"
            )
        return

    result.log_path.parent.mkdir(parents=True, exist_ok=True)
    with result.log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(result.command)}\n\n")
        log.flush()
        completed = subprocess.run(
            list(result.command),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        _print_log_tail(result.log_path)
        raise RuntimeError(
            f"flashdreams-run failed for {result.uuid} with exit code "
            f"{completed.returncode}; see {result.log_path}"
        )


def _print_log_tail(path: Path, *, line_count: int = 80) -> None:
    """Print the tail of a subprocess log before surfacing a failure."""

    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    print(f"=== tail {line_count} lines of {path} ===")
    for line in lines[-line_count:]:
        print(line)
    print(f"=== end tail of {path} ===")


def _write_generation_metadata(result: GenerationResult) -> None:
    metadata_path = result.generated_video_path.parent / "generation.json"
    metadata_path.write_text(
        json.dumps(
            {
                "uuid": result.uuid,
                "output_dir": str(result.output_dir),
                "generated_video_path": str(result.generated_video_path),
                "log_path": str(result.log_path),
                "command": list(result.command),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def remove_generated_case(run_root: Path, uuid: str) -> None:
    """Delete generated artifacts for one UUID."""

    path = run_root / "generated" / uuid
    if path.exists():
        shutil.rmtree(path)
