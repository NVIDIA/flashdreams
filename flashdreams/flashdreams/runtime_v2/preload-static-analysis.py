# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report preparation calls found statically in application source modules."""

from __future__ import annotations

import ast
import importlib.util
import sys
import tokenize
import warnings
from collections.abc import Iterable
from pathlib import Path

PREPARATION_METHODS = frozenset(
    {
        "subprocess.Popen",
        "huggingface_hub.hf_hub_download",
        "huggingface_hub.snapshot_download",
        "urllib.request.urlopen",
        "urllib.request.urlretrieve",
    }
)


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _name(node.value)
        return None if owner is None else f"{owner}.{node.attr}"
    return None


def _aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                bound = imported.asname or imported.name.partition(".")[0]
                aliases[bound] = imported.name if imported.asname else bound
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for imported in node.names:
                if imported.name != "*":
                    aliases[imported.asname or imported.name] = (
                        f"{node.module}.{imported.name}"
                    )
    return aliases


def _resolve(name: str, aliases: dict[str, str]) -> str:
    first, separator, remainder = name.partition(".")
    resolved = aliases.get(first, first)
    return f"{resolved}.{remainder}" if separator else resolved


def _source_path(module_name: str) -> Path | None:
    module = sys.modules.get(module_name)
    source = getattr(module, "__file__", None)
    if source is None:
        return None
    path = Path(source)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            return None
    return path if path.suffix == ".py" and path.is_file() else None


def find_preparation_calls(
    module_names: Iterable[str],
) -> tuple[tuple[Path, int, str], ...]:
    """Return preparation calls found in the supplied loaded application modules."""
    findings: list[tuple[Path, int, str]] = []
    for module_name in sorted(set(module_names)):
        path = _source_path(module_name)
        if path is None:
            continue
        with tokenize.open(path) as source:
            tree = ast.parse(source.read(), filename=str(path))
        aliases = _aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = _name(node.func)
            if called is None:
                continue
            method = _resolve(called, aliases)
            if method in PREPARATION_METHODS:
                findings.append((path, node.lineno, method))
    return tuple(findings)


def _is_external(path: Path) -> bool:
    resolved = path.resolve()
    if any(
        part.casefold() in {"site-packages", "dist-packages"} for part in resolved.parts
    ):
        return True
    try:
        resolved.relative_to(Path(sys.base_prefix).resolve())
    except ValueError:
        return False
    return True


def _location(path: Path, line: int, *, external: bool) -> str:
    if not external:
        try:
            path = path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            pass
    return f"{path}:{line}"


def _module_sources(module_names: Iterable[str]) -> dict[str, Path]:
    return {
        module_name: path
        for module_name in sorted(set(module_names))
        if (path := _source_path(module_name)) is not None
    }


def _internal_calls(
    sources: dict[str, Path],
) -> tuple[tuple[Path, int, str], ...]:
    calls: list[tuple[Path, int, str]] = []
    for path in sources.values():
        if _is_external(path):
            continue
        with tokenize.open(path) as source:
            tree = ast.parse(source.read(), filename=str(path))
        aliases = _aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = _name(node.func)
            if called is None:
                continue
            resolved = _resolve(called, aliases)
            if resolved != called:
                calls.append((path, node.lineno, resolved))
    return tuple(calls)


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_part, right_part in zip(left.split("."), right.split(".")):
        if left_part != right_part:
            break
        length += 1
    return length


def _last_internal_call(
    external_path: Path,
    sources: dict[str, Path],
    internal_calls: tuple[tuple[Path, int, str], ...],
) -> tuple[Path, int, str] | None:
    resolved_external_path = external_path.resolve()
    external_modules = tuple(
        module_name
        for module_name, path in sources.items()
        if path.resolve() == resolved_external_path
    )
    if not external_modules:
        return None
    ranked = [
        (
            max(
                _common_prefix_length(target, external_module)
                for external_module in external_modules
            ),
            path,
            line,
            target,
        )
        for path, line, target in internal_calls
    ]
    matching = [item for item in ranked if item[0] > 0]
    if not matching:
        return None
    _, path, line, target = min(
        matching,
        key=lambda item: (-item[0], str(item[1]), item[2], item[3]),
    )
    return path, line, target


def _report_section(
    title: str,
    findings: tuple[tuple[Path, int, str], ...],
    *,
    external: bool,
) -> list[str]:
    lines = [f"## {title}", ""]
    if not findings:
        return [*lines, "No findings."]
    lines.extend(("| Method | Location |", "| --- | --- |"))
    lines.extend(
        f"| `{method}` | `{_location(path, line, external=external)}` |"
        for path, line, method in findings
    )
    return lines


def _external_report(
    findings: tuple[tuple[Path, int, str], ...],
    module_names: Iterable[str],
) -> list[str]:
    lines = ["## External", ""]
    if not findings:
        return [*lines, "No findings."]
    sources = _module_sources(module_names)
    internal_calls = _internal_calls(sources)
    for path, line, method in findings:
        lines.extend(
            (
                f"### `{method}`",
                "",
                f"External call: `{_location(path, line, external=True)}`",
                "",
                "Static call stack (best-effort package trace):",
            )
        )
        internal_call = _last_internal_call(path, sources, internal_calls)
        if internal_call is None:
            lines.append("- Last internal call: _not statically resolvable_.")
        else:
            internal_path, internal_line, target = internal_call
            lines.append(
                f"- Last internal call into `{target.partition('.')[0]}`: "
                f"`{_location(internal_path, internal_line, external=False)}` "
                f"calls `{target}`."
            )
        lines.extend(
            (
                "- External preparation call: "
                f"`{_location(path, line, external=True)}` calls `{method}`.",
                "",
            )
        )
    return lines


def _print_report(
    findings: tuple[tuple[Path, int, str], ...],
    module_names: Iterable[str],
) -> None:
    internal = tuple(item for item in findings if not _is_external(item[0]))
    external = tuple(item for item in findings if _is_external(item[0]))
    lines = [
        "# Preload static analysis report",
        "",
        "| Scope | Findings |",
        "| --- | ---: |",
        f"| Internal | {len(internal)} |",
        f"| External | {len(external)} |",
        "",
        *_report_section("Internal", internal, external=False),
        "",
        *_external_report(external, module_names),
    ]
    print("\n".join(lines), flush=True)


def warn_about_preparation_calls(module_names: Iterable[str]) -> None:
    """Warn for preparation calls that may run outside application initialization."""
    module_names = tuple(module_names)
    findings = find_preparation_calls(module_names)
    _print_report(findings, module_names)
    for path, line, method in findings:
        warnings.warn_explicit(
            f"{method} may perform preparation outside IApplication.init",
            RuntimeWarning,
            str(path),
            line,
        )
