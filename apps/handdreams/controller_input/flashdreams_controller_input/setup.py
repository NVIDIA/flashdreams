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

"""Public setup commands provided by installed FlashDreams integrations."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import entry_points


def main(argv: list[str] | None = None) -> None:
    """Dispatch optional setup without loading inference or native dependencies."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="flashdreams-controller-input", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Set up an optional integration")
    setup.add_argument("integration", help="Integration to set up, e.g. isaac-teleop")
    # Leave integration-specific flags, including --help, for its own parser.
    args = parser.parse_args(arguments[:2])
    matches = list(entry_points(group="flashdreams.setup", name=args.integration))
    if not matches:
        parser.error(
            f"Setup provider {args.integration!r} is not installed. For isaac-teleop, "
            "install the flashdreams-controller-input package from the HandDreams "
            "release, or use its source launch.py setup-isaac-teleop command."
        )
    if len(matches) != 1:
        parser.error(
            f"Multiple setup providers are installed for {args.integration!r}."
        )
    matches[0].load()(arguments[2:])


if __name__ == "__main__":
    main()
