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

"""Host-native offline bundle builder for FlashDreams v2 applications."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    entry_points,
    packages_distributions,
)
from pathlib import Path
from textwrap import dedent
from urllib.parse import unquote, urlparse

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications_v2"
"""Installed-entry-point group containing FlashDreams v2 applications."""

_COMMAND_SEPARATOR = ":::"
"""Separator between this tool's options and a complete runtime command."""

_FAILURE_GUIDANCE = "Packager failed. This is either a packager bug, missing environment variable, or packaged application crash."
"""Guidance appended to application-preload failures."""

_SAFE_SLUG = re.compile(r"^[A-Za-z0-9_.-]+$")
_INSTALLER_OUTPUT = "INSTALLER_OUTPUT.txt"
_PREPARATION_ISSUES = "PREPARATION_ISSUES.txt"
_RUNTIME_CACHE_ROOT_ENV = "FLASHDREAMS_RUNTIME_CACHE_DIR"
_RUNTIME_CACHE_SEED_MARKER = ".flashdreams-cache-seed"
_SUPPORTED_MODES = ("native-window", "webrtc")

_CACHE_PATHS = {
    "FLASHDREAMS_CACHE_DIR": "flashdreams",
    "FLASHDREAMS_UV_CACHE_DIR": "uv",
    "FLASHDREAMS_HF_CACHE_DIR": "huggingface",
    "FLASHDREAMS_TRITON_CACHE_DIR": "triton",
    "HF_HOME": "huggingface",
    "HF_HUB_CACHE": "huggingface/hub",
    "HF_ASSETS_CACHE": "huggingface/assets",
    "HF_XET_CACHE": "huggingface/xet",
    "TRANSFORMERS_CACHE": "huggingface/transformers",
    "TORCH_HOME": "torch",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda",
    "NUMBA_CACHE_DIR": "numba",
    "XDG_CACHE_HOME": "xdg",
    "UV_CACHE_DIR": "uv",
    "LUDUS_CACHE_DIR": "ludus",
    "LUDUS_PHYSX_CACHE": "ludus/physx",
    "PYINSTALLER_CONFIG_DIR": "pyinstaller",
}
"""Cache variables and their locations below the portable cache root."""


class PackageError(RuntimeError):
    """User-facing bundle construction failure."""


def _parser() -> argparse.ArgumentParser:
    """Build the tool's command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Preload a FlashDreams v2 application into private caches, then "
            "freeze it as a host-native offline PyInstaller bundle."
        ),
        epilog=(
            "Usage: %(prog)s [OPTIONS] ::: flashdreams-run-v2 SLUG "
            "--mode native-window [RUNTIME_ARGS] -- [APPLICATION_ARGS]."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Bundle directory (default: artifacts/<slug>-bundle).",
    )
    parser.add_argument(
        "--preload-timeout",
        type=float,
        help="Stop init and preload after this many seconds.",
    )
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Skip the one-block runtime check; initialization still runs.",
    )
    parser.add_argument(
        "--hidden-import",
        action="append",
        default=[],
        help="Additional PyInstaller hidden import (repeatable).",
    )
    parser.add_argument(
        "--collect-executable",
        action="append",
        default=[],
        help="Executable from PATH to include in the runtime (repeatable).",
    )
    return parser


def _parse_arguments(arguments: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split tool options from the complete application command."""
    parser = _parser()
    if _COMMAND_SEPARATOR in arguments:
        separator = arguments.index(_COMMAND_SEPARATOR)
        return parser.parse_args(arguments[:separator]), arguments[separator + 1 :]
    parser.parse_args(arguments)
    parser.error("a flashdreams-run-v2 command must follow :::")


def _v2_invocation(command: list[str]) -> tuple[str, list[str]]:
    """Return the slug and configuration from a v2 command line."""
    if not command:
        raise PackageError("A flashdreams-run-v2 command must follow :::.")
    executable = Path(command[0]).name.lower().removesuffix(".exe")
    if executable != "flashdreams-run-v2":
        raise PackageError("The command after ::: must start with flashdreams-run-v2.")
    if len(command) == 1:
        raise PackageError("The flashdreams-run-v2 command is missing its slug.")
    return command[1], command[2:]


def _runtime_arguments(config: list[str]) -> list[str]:
    """Return arguments owned by the runtime rather than the application."""
    return config[: config.index("--")] if "--" in config else config


def _application_arguments(config: list[str]) -> list[str]:
    """Return arguments owned by the configured application."""
    if "--" not in config:
        return []
    return config[config.index("--") + 1 :]


def _option_values(arguments: list[str], option: str) -> list[str | None]:
    """Return every value supplied for one long runtime option."""
    values: list[str | None] = []
    prefix = f"{option}="
    for index, argument in enumerate(arguments):
        if argument == option:
            values.append(arguments[index + 1] if index + 1 < len(arguments) else None)
        elif argument.startswith(prefix):
            values.append(argument[len(prefix) :])
    return values


def _validate_config(config: list[str]) -> None:
    """Reject runtime configurations that cannot produce an interactive bundle."""
    runtime_arguments = _runtime_arguments(config)
    modes = _option_values(runtime_arguments, "--mode")
    if not modes or modes[-1] not in _SUPPORTED_MODES:
        supported = " or ".join(_SUPPORTED_MODES)
        raise PackageError(f"Packaging requires --mode {supported}.")
    if _option_values(runtime_arguments, "--preload-application"):
        raise PackageError(
            "Do not pass --preload-application; the packager controls preparation."
        )


def _cache_environment(
    cache_root: Path,
    *,
    preparation_issues_path: Path | None = None,
) -> dict[str, str]:
    """Return an environment with every supported cache below ``cache_root``."""
    env = os.environ.copy()
    env["HF_HUB_DISABLE_SYMLINKS"] = "1"
    for name, relative_path in _CACHE_PATHS.items():
        cache_path = cache_root / relative_path
        cache_path.mkdir(parents=True, exist_ok=True)
        env[name] = str(cache_path)
    if preparation_issues_path is not None:
        env["FLASHDREAMS_PREPARATION_ISSUES_PATH"] = str(preparation_issues_path)
    return env


def _application_module(slug: str) -> str:
    """Return the module registered for ``slug`` or its import fallback."""
    for entry_point in entry_points(group=_APPLICATION_ENTRY_POINT_GROUP):
        if entry_point.name == slug:
            return entry_point.value.partition(":")[0]
    return slug.replace("-", "_")


def _metadata_distributions(module: str) -> tuple[str, ...]:
    """Return installed distributions whose metadata the registry needs."""
    names = {"flashdreams"}
    names.update(packages_distributions().get(module.partition(".")[0], ()))
    return tuple(sorted(names))


def _path_from_file_url(url: str) -> Path:
    """Return a local path from an editable distribution's file URL.

    Args:
        url: Absolute ``file:`` URL from ``direct_url.json``.

    Returns:
        Decoded local filesystem path.

    Raises:
        PackageError: ``url`` does not identify a local file.
    """
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise PackageError(f"Editable dependency has a non-file URL: {url!r}.")
    path = unquote(parsed.path)
    if parsed.netloc and parsed.netloc != "localhost":
        path = f"//{parsed.netloc}{path}"
    elif os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    return Path(path)


def _local_dependency_modules(distributions: tuple[str, ...]) -> tuple[str, ...]:
    """Return top-level modules from transitive editable dependencies."""
    dependency_roots: dict[str, Path] = {}
    pending = list(distributions)
    visited: set[str] = set()
    while pending:
        distribution_name = canonicalize_name(pending.pop())
        if distribution_name in visited:
            continue
        visited.add(distribution_name)
        try:
            requirements = distribution(distribution_name).requires or ()
        except PackageNotFoundError:
            continue
        for requirement_text in requirements:
            requirement = Requirement(requirement_text)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            try:
                dependency = distribution(requirement.name)
            except PackageNotFoundError:
                continue
            direct_url = dependency.read_text("direct_url.json")
            if direct_url is None:
                continue
            editable = json.loads(direct_url)
            if not editable.get("dir_info", {}).get("editable"):
                continue
            dependency_name = canonicalize_name(requirement.name)
            dependency_roots[dependency_name] = _path_from_file_url(editable["url"])
            pending.append(dependency_name)

    modules = []
    for module, owners in packages_distributions().items():
        roots = {
            dependency_roots[owner]
            for owner in map(canonicalize_name, owners)
            if owner in dependency_roots
        }
        if not roots:
            continue
        spec = importlib.util.find_spec(module)
        if spec is None:
            continue
        locations = spec.submodule_search_locations
        source_path = (
            Path(next(iter(locations))) if locations else Path(spec.origin or "")
        )
        if any(source_path.resolve().is_relative_to(root) for root in roots):
            modules.append(module)
    return tuple(sorted(modules))


def _installed_extra_modules(distributions: tuple[str, ...]) -> tuple[str, ...]:
    """Return top-level modules from installed native optional dependencies."""
    dependency_names: set[str] = set()
    dependency_modules: set[str] = set()
    for distribution_name in distributions:
        try:
            requirements = distribution(distribution_name).requires or ()
        except PackageNotFoundError:
            continue
        for requirement_text in requirements:
            requirement = Requirement(requirement_text)
            if requirement.marker is None or "extra" not in str(requirement.marker):
                continue
            try:
                dependency = distribution(requirement.name)
            except PackageNotFoundError:
                continue
            native_files = [
                Path(file)
                for file in dependency.files or ()
                if Path(file).suffix.lower() in {".dll", ".dylib", ".pyd", ".so"}
            ]
            if not native_files:
                continue
            dependency_names.add(canonicalize_name(requirement.name))
            for file in native_files:
                module = file.parts[0]
                if (
                    module.isidentifier()
                    and importlib.util.find_spec(module) is not None
                ):
                    dependency_modules.add(module)

    dependency_modules.update(
        module
        for module, owners in packages_distributions().items()
        if dependency_names.intersection(map(canonicalize_name, owners))
    )
    return tuple(sorted(dependency_modules))


def _extension_submodules(module: str) -> tuple[str, ...]:
    """Return compiled Python extension modules owned below ``module``."""
    extensions: set[str] = set()
    for distribution_name in packages_distributions().get(module, ()):
        for file in distribution(distribution_name).files or ():
            path = Path(file)
            if path.suffix.lower() not in {".pyd", ".so"}:
                continue
            parts = path.parts
            if not parts or parts[0] != module:
                continue
            extensions.add(".".join((*parts[:-1], path.name.split(".", 1)[0])))
    return tuple(sorted(extensions))


def _nvrtc_builtins() -> tuple[tuple[Path, Path], ...]:
    """Return NVRTC builtins binaries and their installed parent directories."""
    binaries: dict[Path, Path] = {}
    for distribution_name in ("nvidia-cuda-nvrtc", "torch"):
        try:
            installed = distribution(distribution_name)
        except PackageNotFoundError:
            continue
        for file in installed.files or ():
            relative_path = Path(file)
            if "nvrtc-builtins" not in relative_path.name.lower():
                continue
            source = Path(str(installed.locate_file(file))).resolve()
            if source.is_file():
                binaries[source] = relative_path.parent
    return tuple(sorted(binaries.items()))


def _module_search_path(module: str) -> Path:
    """Return the import root for a package, including editable installs."""
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise PackageError(f"Cannot locate package {module!r}.")
    if spec.submodule_search_locations:
        path = Path(next(iter(spec.submodule_search_locations))).resolve()
        for _ in module.split("."):
            path = path.parent
        return path
    if not spec.origin:
        raise PackageError(f"Cannot locate package {module!r}.")
    path = Path(spec.origin).resolve().parent
    for _ in module.split(".")[1:]:
        path = path.parent
    return path


def _launcher_source(
    slug: str, config: list[str], cache_seed_id: str | None = None
) -> str:
    """Return a launcher that seeds and uses a writable runtime cache."""
    embedded_runtime_arguments = _runtime_arguments(config)
    embedded_application_arguments = _application_arguments(config)
    return dedent(
        f"""\
        # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
        # SPDX-License-Identifier: Apache-2.0
        import importlib.machinery
        import importlib.util
        import os
        import shutil
        import sys
        from pathlib import Path

        bundle_root = (
            Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent
        )
        seed_cache_root = bundle_root / "cache"
        seed_cache_root.mkdir(parents=True, exist_ok=True)
        cache_root = seed_cache_root
        cache_seed_id = {cache_seed_id!r}
        if getattr(sys, "frozen", False):
            configured_cache_root = os.environ.get({_RUNTIME_CACHE_ROOT_ENV!r})
            if configured_cache_root:
                cache_root = Path(configured_cache_root).expanduser()
                if not cache_root.is_absolute():
                    raise RuntimeError(
                        f"{_RUNTIME_CACHE_ROOT_ENV} must be an absolute path."
                    )
            elif os.name == "nt":
                local_app_data = os.environ.get("LOCALAPPDATA")
                if not local_app_data:
                    raise RuntimeError(
                        "LOCALAPPDATA is required unless "
                        f"{_RUNTIME_CACHE_ROOT_ENV} is set."
                    )
                cache_root = (
                    Path(local_app_data)
                    / "FlashDreams"
                    / {slug!r}
                    / "cache"
                )
            else:
                cache_root = (
                    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                    / "flashdreams"
                    / {slug!r}
                )

            cache_root = cache_root.resolve()
            seed_cache_root = seed_cache_root.resolve()
            if cache_root != seed_cache_root:
                if cache_root.is_relative_to(
                    seed_cache_root
                ) or seed_cache_root.is_relative_to(cache_root):
                    raise RuntimeError(
                        "The runtime cache cannot contain, or be contained by, "
                        "the bundled cache."
                    )
                marker = cache_root / {_RUNTIME_CACHE_SEED_MARKER!r}
                if cache_seed_id is not None and (
                    not marker.is_file()
                    or marker.read_text(encoding="utf-8") != cache_seed_id
                ):
                    shutil.copytree(seed_cache_root, cache_root, dirs_exist_ok=True)
                    marker.write_text(cache_seed_id, encoding="utf-8")

        cache_paths = {_CACHE_PATHS!r}
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        for name, relative_path in cache_paths.items():
            cache_path = cache_root / relative_path
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(cache_path)

        if getattr(sys, "frozen", False):
            runtime_root = Path(getattr(sys, "_MEIPASS", bundle_root))
            os.environ["PATH"] = (
                str(runtime_root) + os.pathsep + os.environ.get("PATH", "")
            )
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["DIFFUSERS_OFFLINE"] = "1"
            os.environ["LOCAL_FILES_ONLY"] = "1"
            os.environ["TRITON_BACKENDS_IN_TREE"] = "1"

            # Validation builds JIT extensions against this exact environment. Reuse
            # those binaries because rebuilding would require a target-side compiler.
            from torch.utils import cpp_extension

            original_cpp_extension_load = cpp_extension.load

            def load_cpp_extension(name, *args, **kwargs):
                extension_dir = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / name
                if kwargs.get("is_python_module", True):
                    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                        binary = extension_dir / f"{{name}}{{suffix}}"
                        if not binary.is_file():
                            continue
                        spec = importlib.util.spec_from_file_location(name, binary)
                        if spec is None or spec.loader is None:
                            break
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        sys.modules[name] = module
                        return module
                return original_cpp_extension_load(name, *args, **kwargs)

            cpp_extension.load = load_cpp_extension

            try:
                from ludus_renderer import _physx_native
            except ModuleNotFoundError:
                pass
            else:
                def reuse_preloaded_physx(cache_root, _physx_root):
                    module_dir = (
                        Path(cache_root)
                        / "module"
                        / (sys.implementation.cache_tag or "python")
                    )
                    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                        binary = module_dir / f"ludus_physx_native{{suffix}}"
                        if binary.is_file():
                            return binary
                    raise RuntimeError(
                        f"Preloaded Ludus PhysX module is missing from {{module_dir}}."
                    )

                _physx_native._configure_and_build = reuse_preloaded_physx

        from flashdreams.runtime_v2.cli import entrypoint, split_arguments

        runtime_arguments, application_arguments = split_arguments(sys.argv[1:])
        application_arguments = {embedded_application_arguments!r} + application_arguments
        entrypoint([
            {slug!r},
            *{embedded_runtime_arguments!r},
            *runtime_arguments,
            *(["--", *application_arguments] if application_arguments else []),
        ])
        """
    )


def _preload_source(
    slug: str, config: list[str], *, skip_runtime_validation: bool = False
) -> str:
    """Return a process entrypoint that preloads and validates an application."""
    preload_arguments = [slug, "--preload-application"]
    if skip_runtime_validation:
        preload_arguments.append("--skip-preload-validation")
    runtime_arguments = iter(_runtime_arguments(config))
    for argument in runtime_arguments:
        if argument == "--mode":
            next(runtime_arguments)
        elif not argument.startswith("--mode="):
            preload_arguments.append(argument)
    if "--" in config:
        preload_arguments.extend(("--", *_application_arguments(config)))
    return dedent(
        f"""\
        # SPDX-License-Identifier: Apache-2.0
        from flashdreams.runtime_v2.cli import entrypoint
        from triton.runtime.driver import driver as triton_driver

        entrypoint({preload_arguments!r})
        triton_driver.active.get_current_target()
        """
    )


def _bundle_readme_source(slug: str) -> str:
    """Return instructions for the generated bundle root."""
    executable = _executable_name(slug)
    launch = f".\\{executable}" if sys.platform == "win32" else f"./{executable}"
    platform_name = "Windows" if sys.platform == "win32" else "Linux"
    return dedent(
        f"""\
        # {slug} offline bundle

        ## Launch

        This bundle was built for {platform_name}. Run it from this directory:

        ```text
        {launch}
        ```

        The application and both runtime/application argument sets are embedded in
        the executable. Extra command-line arguments are appended to that configuration.

        The packaged `cache/` is copied once to a writable per-user cache. Set
        `{_RUNTIME_CACHE_ROOT_ENV}` to an absolute writable path to override it.
        The packaged cache remains the offline seed.

        ## Expected processes during runtime

        | Process | Expected count | Purpose |
        | --- | ---: | --- |
        | `{executable}` | 1 | Hosts the model loop and selected client-window UI. |

        `python`, `cmake`, and `ninja` processes are not expected during normal runtime;
        Python and validated native extensions are contained in this bundle.

        ## Bundle root contents

        | Path | Description |
        | --- | --- |
        | `{executable}` | Compiled launcher for this operating system. |
        | `data/` | Python runtime, application code, native libraries, and GPU assets. |
        | `cache/` | Model weights, scenes, compiled kernels, and native extensions. |
        | `{_INSTALLER_OUTPUT}` | Application preload/validation and PyInstaller output. |
        | `{_PREPARATION_ISSUES}` | Optional preload warnings with application call stacks. |
        | `README.md` | This launch and bundle-layout guide. |

        Keep the complete directory together. The executable is not standalone from
        `data/` and `cache/`, and moving only the executable will not work.
        """
    )


def _pyinstaller_command(
    *,
    launcher: Path,
    slug: str,
    application_module: str,
    build_root: Path,
    hidden_imports: tuple[str, ...] = (),
    executables: tuple[str, ...] = (),
) -> list[str]:
    """Prepare and return the PyInstaller command for an application package."""
    module_root = application_module.partition(".")[0]
    slangpy_shaders = _module_search_path("slangpy") / "slangpy" / "shaders"
    metadata_distributions = _metadata_distributions(application_module)
    local_modules = {module_root}
    local_modules.update(_local_dependency_modules(metadata_distributions))
    native_modules = set(
        _installed_extra_modules(tuple(packages_distributions().get(module_root, ())))
    )
    source_modules = {*local_modules, "flashdreams"}
    local_modules.discard("flashdreams")

    hooks = build_root / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    for module in source_modules:
        (hooks / f"hook-{module}.py").write_text(
            "module_collection_mode = 'pyz+py'\n", encoding="utf-8"
        )

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        slug,
        "--contents-directory",
        "data",
        "--distpath",
        str(build_root / "dist"),
        "--workpath",
        str(build_root / "work"),
        "--specpath",
        str(build_root / "spec"),
        "--additional-hooks-dir",
        str(hooks),
        "--paths",
        str(_module_search_path("flashdreams.runtime_v2")),
        "--collect-data",
        "flashdreams",
        "--collect-all",
        "triton",
        "--collect-all",
        "torch",
        "--collect-all",
        "slangpy",
        "--add-data",
        f"{slangpy_shaders}{os.pathsep}shaders",
        "--collect-submodules",
        "transformers",
        "--hidden-import",
        application_module,
    ]
    for module in sorted(local_modules):
        command.extend(("--paths", str(_module_search_path(module))))
        command.extend(("--collect-all", module))
    for module in sorted(native_modules):
        command.extend(("--collect-data", module))
        command.extend(("--collect-binaries", module))
        for extension in _extension_submodules(module):
            command.extend(("--hidden-import", extension))
    for distribution_name in metadata_distributions:
        command.extend(("--copy-metadata", distribution_name))
    for hidden_import in hidden_imports:
        command.extend(("--hidden-import", hidden_import))
    for binary, installed_parent in _nvrtc_builtins():
        for destination in sorted({Path("."), installed_parent}):
            command.extend(("--add-binary", f"{binary}{os.pathsep}{destination}"))
    for executable_name in executables:
        executable = shutil.which(executable_name)
        if executable is None:
            raise PackageError(f"Cannot locate executable {executable_name!r} on PATH.")
        command.extend(("--add-binary", f"{executable}{os.pathsep}."))
    command.append(str(launcher))
    return command


def _validate_slug(slug: str) -> None:
    """Reject slugs that cannot safely become directory or executable names."""
    if not _SAFE_SLUG.fullmatch(slug) or slug in {".", ".."}:
        raise PackageError(
            "The slug may contain only letters, numbers, dots, underscores, "
            "and hyphens."
        )


def _executable_name(slug: str) -> str:
    """Return the platform-native executable name for ``slug``."""
    return f"{slug}.exe" if sys.platform == "win32" else slug


def package(
    slug: str,
    config: list[str],
    output: Path,
    preload_timeout: float | None = None,
    hidden_imports: tuple[str, ...] = (),
    executables: tuple[str, ...] = (),
    skip_runtime_validation: bool = False,
) -> Path:
    """Preload and freeze one configured v2 application into ``output``.

    Args:
        slug: Registered v2 application slug.
        config: Arguments after the slug in ``flashdreams-run-v2``.
        output: Final bundle directory, which must not already exist.
        preload_timeout: Optional limit for application preload and validation.
        hidden_imports: Additional modules that PyInstaller cannot discover.
        executables: Executables from PATH required by the frozen runtime.
        skip_runtime_validation: Skip the preload's one-block runtime check.

    Returns:
        Absolute path to the completed bundle.

    Raises:
        PackageError: The request, preload run, or frozen bundle is invalid.
    """
    _validate_slug(slug)
    _validate_config(config)
    if preload_timeout is not None and (
        not math.isfinite(preload_timeout) or preload_timeout <= 0
    ):
        raise PackageError("--preload-timeout must be finite and greater than zero.")
    if sys.platform != "win32" and not sys.platform.startswith("linux"):
        raise PackageError("Only Windows and Linux bundles are supported.")
    if importlib.util.find_spec("PyInstaller") is None:
        raise PackageError(
            "PyInstaller is unavailable in this environment. Install pyinstaller>=6.22.2."
        )

    destination = output.expanduser().resolve()
    if destination.exists():
        raise PackageError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".fd-{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    launcher = staging / "launcher.py"
    preload = staging / "preload.py"
    build_root = staging / ".build"
    cache_root = staging / "cache"
    installer_output = staging / _INSTALLER_OUTPUT
    preparation_issues = staging / _PREPARATION_ISSUES
    succeeded = False

    try:
        cache_root.mkdir()
        preload.write_text(
            _preload_source(
                slug,
                config,
                skip_runtime_validation=skip_runtime_validation,
            ),
            encoding="utf-8",
        )
        print(f"Preloading {slug!r} into {cache_root}", flush=True)
        try:
            with installer_output.open("ab") as output_stream:
                output_stream.write(b"=== Application preload and validation ===\n")
                output_stream.flush()
                preload_result = subprocess.run(
                    [sys.executable, str(preload)],
                    env=_cache_environment(
                        cache_root,
                        preparation_issues_path=preparation_issues,
                    ),
                    check=False,
                    timeout=preload_timeout,
                    stdout=output_stream,
                    stderr=subprocess.STDOUT,
                )
                output_stream.write(
                    f"\nExit code: {preload_result.returncode}\n\n".encode()
                )
        except subprocess.TimeoutExpired as error:
            with installer_output.open("ab") as output_stream:
                output_stream.write(b"\nTimed out.\n\n")
            raise PackageError(
                f"Preload timed out for {slug!r}.\n{_FAILURE_GUIDANCE}"
            ) from error
        if preload_result.returncode:
            raise PackageError(
                f"Preload failed for {slug!r} with exit code "
                f"{preload_result.returncode}.\n{_FAILURE_GUIDANCE}"
            )

        application_module = _application_module(slug)
        launcher.write_text(
            _launcher_source(slug, config, uuid.uuid4().hex), encoding="utf-8"
        )
        executable_name = _executable_name(slug)
        print(f"Building {executable_name}", flush=True)
        with installer_output.open("ab") as output_stream:
            output_stream.write(b"=== PyInstaller build ===\n")
            output_stream.flush()
            pyinstaller_result = subprocess.run(
                _pyinstaller_command(
                    launcher=launcher,
                    slug=slug,
                    application_module=application_module,
                    build_root=build_root,
                    hidden_imports=hidden_imports,
                    executables=executables,
                ),
                cwd=staging,
                env=_cache_environment(cache_root),
                check=False,
                stdout=output_stream,
                stderr=subprocess.STDOUT,
            )
            output_stream.write(
                f"\nExit code: {pyinstaller_result.returncode}\n\n".encode()
            )
        if pyinstaller_result.returncode:
            raise PackageError(
                f"PyInstaller failed with exit code {pyinstaller_result.returncode}."
            )

        built_bundle = build_root / "dist" / slug
        executable = built_bundle / executable_name
        if not executable.is_file() or not (built_bundle / "data").is_dir():
            raise PackageError(
                "PyInstaller completed without the expected executable/data layout."
            )

        for item in built_bundle.iterdir():
            shutil.move(str(item), staging / item.name)
        (staging / "README.md").write_text(
            _bundle_readme_source(slug), encoding="utf-8"
        )

        shutil.rmtree(build_root)
        launcher.unlink()
        preload.unlink()
        staging.replace(destination)
        succeeded = True
        return destination
    finally:
        if not succeeded:
            shutil.rmtree(staging, ignore_errors=True)


def main(arguments: list[str] | None = None) -> int:
    """Run the command-line bundle builder."""
    parsed, command = _parse_arguments(
        list(sys.argv[1:] if arguments is None else arguments)
    )
    try:
        slug, config = _v2_invocation(command)
        output = parsed.output or Path("artifacts") / f"{slug}-bundle"
        completed = package(
            slug,
            config,
            output,
            preload_timeout=parsed.preload_timeout,
            hidden_imports=tuple(parsed.hidden_import),
            executables=tuple(parsed.collect_executable),
            skip_runtime_validation=parsed.skip_runtime_validation,
        )
    except PackageError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Bundle ready: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
