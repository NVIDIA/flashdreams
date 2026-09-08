# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit command-line option tracking."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

_EXPLICIT_ARG_DESTS_ATTR = "_explicit_arg_dests"


def explicit_arg_dests(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> frozenset[str]:
    """Return parser destinations whose option strings appear in ``argv``."""
    option_dests = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    return frozenset(
        destination
        for token in argv
        if (destination := option_dests.get(token.split("=", 1)[0])) is not None
    )


def arg_was_explicit(args: argparse.Namespace, destination: str) -> bool:
    """Return whether a namespace field came from an explicit CLI option."""
    return destination in getattr(args, _EXPLICIT_ARG_DESTS_ATTR, frozenset())


class ExplicitArgTrackingArgumentParser(argparse.ArgumentParser):
    """Record the optional arguments supplied by the user."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        """Parse arguments and attach their explicitly supplied destinations."""
        raw_args = sys.argv[1:] if args is None else list(args)
        parsed = super().parse_args(raw_args, namespace)
        assert parsed is not None
        setattr(parsed, _EXPLICIT_ARG_DESTS_ATTR, explicit_arg_dests(self, raw_args))
        return parsed
