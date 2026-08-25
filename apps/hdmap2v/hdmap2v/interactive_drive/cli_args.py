# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

_EXPLICIT_ARG_DESTS_ATTR = "_explicit_arg_dests"


def explicit_arg_dests(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> frozenset[str]:
    """Return parser destination names whose option strings appear in ``argv``."""
    option_dests = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    explicit = set()
    for token in argv:
        option = token.split("=", 1)[0]
        dest = option_dests.get(option)
        if dest is not None:
            explicit.add(dest)
    return frozenset(explicit)


def arg_was_explicit(args: argparse.Namespace, dest: str) -> bool:
    """Return whether a parsed namespace field came from an explicit CLI flag."""
    return dest in getattr(args, _EXPLICIT_ARG_DESTS_ATTR, frozenset())


class ExplicitArgTrackingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that records which optional arguments users supplied."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        raw_args = sys.argv[1:] if args is None else list(args)
        parsed = super().parse_args(raw_args, namespace)
        setattr(parsed, _EXPLICIT_ARG_DESTS_ATTR, explicit_arg_dests(self, raw_args))
        return parsed
