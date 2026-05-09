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

"""Config primitives: ``InstantiateConfig`` (``_target`` + ``setup``) and ``derive_config`` patching."""

import copy
from dataclasses import dataclass
from typing import Any, TypeVar


class PrintableConfig:
    """Config base class providing a multi-line ``__str__`` for human-readable dumps."""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


@dataclass
class InstantiateConfig(PrintableConfig):
    """Config carrying a ``_target`` class plus its kwargs, instantiable via ``setup``."""

    _target: type

    def setup(self, **kwargs: Any) -> Any:
        """Instantiate the configured object."""
        return self._target(self, **kwargs)


T = TypeVar("T")


def derive_config(base_config: T, **changes: Any) -> T:
    """Deep-copy a base config and apply nested keyword overrides.

    Nested ``dict`` values walk into both dataclass attributes and nested
    dicts; leaf values overwrite directly. Raises ``KeyError`` on unknown
    paths.

    Examples:

        new_config = derive_config(
            base_config,
            tokenizer=WanVAEInterfaceConfig(checkpoint_path=...),
            dit=dict(len_t=3, checkpoint_path=...),
        )
    """

    def _is_patchable_object(x: Any) -> bool:
        # Object is patchable if it has attribute storage (__dict__).
        return hasattr(x, "__dict__")

    def _get_field(target: Any, key: str, path: str) -> Any:
        if isinstance(target, dict):
            if key not in target:
                raise KeyError(f"Unknown key at {path}: {key}")
            return target[key]
        if hasattr(target, key):
            return getattr(target, key)
        raise KeyError(f"Unknown field at {path}: {type(target).__name__}.{key}")

    def _set_field(target: Any, key: str, value: Any, path: str) -> None:
        if isinstance(target, dict):
            if key not in target:
                raise KeyError(f"Unknown key at {path}: {key}")
            target[key] = value
            return
        if hasattr(target, key):
            setattr(target, key, value)
            return
        raise KeyError(f"Unknown field at {path}: {type(target).__name__}.{key}")

    def _recursive_patch(
        target: Any, patch: dict[str, Any], path: str = "root"
    ) -> None:
        # Apply patch recursively to dict/object target.
        for key, value in patch.items():
            current = _get_field(target, key, path)
            current_path = f"{path}.{key}"
            if isinstance(value, dict):
                # Nested patch: current can be dict or object.
                if isinstance(current, dict) or _is_patchable_object(current):
                    _recursive_patch(current, value, current_path)
                else:
                    raise TypeError(
                        f"Cannot apply nested dict patch to non-nested field at {current_path} "
                        f"(type={type(current).__name__})"
                    )
            else:
                # Leaf patch: direct assignment.
                _set_field(target, key, value, path)

    # Deep-copy base config so original config is never mutated.
    cfg = copy.deepcopy(base_config)
    _recursive_patch(cfg, changes, path=type(cfg).__name__)
    return cfg
