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

"""Reliability-shrunk gain prediction: gain*(t) = alpha*(t) x rho(t).

Combines a gate JSON (per-timestep unbiased alpha*, the systematic share of
the drift error) with TBIN lines from a ``TBIN_EVAL`` trainer run (rho(t) =
per-timestep val R^2 of the trained corrector) into the predicted deploy
schedule, and ranks the standard dial arms by RMS distance to it
(gain-prediction analysis, analysis arm 2026-07-24).

Usage::

    python gain_predict.py <gate.json> "t=1000 R2=+0.36; t=803 R2=+0.39"
"""

from __future__ import annotations

import json
import re
import sys


def main() -> None:
    gate = {
        float(t): v["alpha_star_unbiased"]
        for t, v in json.load(open(sys.argv[1]))["per_timestep"].items()
    }
    rho = {
        float(m[0]): float(m[1])
        for m in re.findall(r"t=(\d+)\s+R2=([+-][\d.]+)", sys.argv[2])
    }
    ts = sorted(gate, reverse=True)
    pred = {t: gate[t] * rho[t] for t in ts}
    print("predicted gain*(t):", {int(t): round(g, 3) for t, g in pred.items()})

    arms = {
        "corr (flat 1.0)": {t: 1.0 for t in ts},
        "corr050 (flat 0.5)": {t: 0.5 for t in ts},
        "corrgate (a*x1)": dict(gate),
        "corrgate050 (a*x0.5)": {t: gate[t] * 0.5 for t in ts},
        "corrgate025 (a*x0.25)": {t: gate[t] * 0.25 for t in ts},
    }

    def rms(s):
        return (sum((s[t] - pred[t]) ** 2 for t in ts) / len(ts)) ** 0.5

    for name, s in sorted(arms.items(), key=lambda kv: rms(kv[1])):
        print(f"  {name:24s} rms-dist {rms(s):.3f}")


if __name__ == "__main__":
    main()
