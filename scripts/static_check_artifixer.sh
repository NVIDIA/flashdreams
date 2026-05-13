#!/bin/bash
# Fast static check for the artifixer plugin. Runs on any host with system
# python3 (no torch / flashdreams / uv required). Catches syntax errors,
# malformed pyproject.toml, and dangling imports against the flashdreams
# source tree. Designed as a sub-second iteration loop for Phase 1 scaffold
# changes before we go through the full slurm-based smoke test.
#
# Usage:
#   bash scripts/static_check_artifixer.sh

set -euo pipefail

REPO_DIR="$(git -C "${PWD}" rev-parse --show-toplevel)"
cd "${REPO_DIR}"

echo "[static-check] py_compile artifixer sources"
for f in $(find integrations/artifixer -name '*.py' -not -path '*/__pycache__/*'); do
    python3 -m py_compile "$f"
done

echo "[static-check] pyproject.toml entry-point matches a config.py attribute"
python3 - <<'PY'
import re

with open("integrations/artifixer/pyproject.toml") as f:
    pyproject = f.read()
m = re.search(
    r'\[project\.entry-points\."flashdreams\.runner_configs"\]\s*\n((?:"[^"]+"\s*=\s*"[^"]+"\s*\n)+)',
    pyproject,
)
assert m, "missing [project.entry-points.\"flashdreams.runner_configs\"] block"
entries = dict(re.findall(r'"([^"]+)"\s*=\s*"([^"]+)"', m.group(1)))

with open("integrations/artifixer/artifixer/config.py") as f:
    config_py = f.read()

for slug, target in entries.items():
    module, attr = target.split(":", 1)
    assert module == "artifixer.config", f"unexpected module in {slug!r}: {module}"
    assert re.search(rf"^{re.escape(attr)}\s*=", config_py, flags=re.MULTILINE), (
        f"entry-point {slug!r} -> {attr} not defined in artifixer/config.py"
    )
print(f"  ok: {len(entries)} entry-points all resolve to config.py attributes")
PY

echo "[static-check] flashdreams imports resolve to classes in the source tree"
python3 - <<'PY'
import os
import re
from pathlib import Path

import_re = re.compile(
    r"from\s+(flashdreams\.[\w.]+)\s+import\s+(?:\(\s*([\s\S]*?)\s*\)|([^\n]+))",
)
flashdreams_files = list(Path("flashdreams").rglob("*.py"))
for source_path in sorted(Path("integrations/artifixer/artifixer").rglob("*.py")):
    src = source_path.read_text()
    for module, paren_blob, inline_blob in import_re.findall(src):
        names_blob = paren_blob or inline_blob
        for name in re.split(r"[,\s]+", names_blob):
            name = name.strip()
            if not name:
                continue
            pat = re.compile(rf"^class {re.escape(name)}\b|^{re.escape(name)}\s*=", re.MULTILINE)
            hit = any(pat.search(p.read_text(errors="ignore")) for p in flashdreams_files)
            assert hit, (
                f"{source_path}: from {module} import {name} — no class/def in flashdreams/"
            )
        print(f"  ok: from {module} import ({names_blob.strip()})")
PY

echo "[static-check] all good"
