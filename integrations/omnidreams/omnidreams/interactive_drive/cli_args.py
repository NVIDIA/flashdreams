# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

_EXPLICIT_ARG_DESTS_ATTR = "_explicit_arg_dests"


def _resolve_option_action(
    parser: argparse.ArgumentParser, token: str
) -> argparse.Action | None:
    """Resolve a raw argv token to the argparse action that consumes it."""
    option = token.split("=", 1)[0]
    action = parser._option_string_actions.get(option)
    if action is not None:
        return action

    if not getattr(parser, "allow_abbrev", True):
        return None
    if len(option) < 2 or option[0] not in parser.prefix_chars:
        return None

    matches = []
    if option[1] in parser.prefix_chars:
        for option_string, candidate in parser._option_string_actions.items():
            if option_string.startswith(option):
                matches.append(candidate)
    else:
        short_option = option[:2]
        for option_string, candidate in parser._option_string_actions.items():
            if option_string == short_option or option_string.startswith(option):
                matches.append(candidate)

    if len(matches) == 1:
        return matches[0]
    return None


def explicit_arg_dests(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> frozenset[str]:
    """Return parser destination names whose options appear in ``argv``."""
    explicit = set()
    for token in argv:
        action = _resolve_option_action(parser, token)
        if action is not None:
            explicit.add(action.dest)
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
