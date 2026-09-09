"""Runtime debug/telemetry gate for the streaming performance study.

Set ``FLASHDREAMS_DEBUG=1`` to enable the per-chunk METRIC telemetry and the
client-side latent-download / decode-timing instrumentation. Off by default so
production streaming paths pay no measurement cost. Reviewers: anything guarded
by ``FD_DEBUG`` is instrumentation, not part of the core decode-skip change.
"""

import os

FD_DEBUG = os.environ.get("FLASHDREAMS_DEBUG", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
    "off",
)
