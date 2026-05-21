#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert NV-style Vulkan GLSL mesh shaders to VK_EXT_mesh_shader.

The .glsl files in this directory are authored against ``GL_NV_mesh_shader``.
This script rewrites them in place to target the modern
``GL_EXT_mesh_shader`` extension so they compile and run on Vulkan 1.3
drivers that do not expose the NV variant.

Run once after touching the originals; commit the EXT-style output.

Translations:
  - ``#extension GL_NV_mesh_shader``         -> ``GL_EXT_mesh_shader``
  - ``#extension GL_NV_fragment_shader_barycentric`` -> ``GL_EXT_fragment_shader_barycentric``
  - ``taskNV {out,in} NAME { ... };``        -> ``struct NAME { ... }; taskPayloadSharedEXT NAME payload;``
  - References to payload fields (in code)   -> ``payload.FIELD``
  - ``gl_TaskCountNV = X;``                  -> deferred ``EmitMeshTasksEXT(X, 1, 1); return;``
  - ``gl_PrimitiveCountNV = X;``             -> ``SetMeshOutputsEXT(MAX_V, X);`` (first assign) and
                                                ``_prim_count = X; SetMeshOutputsEXT(MAX_V, _prim_count);`` (subsequent)
  - ``gl_MeshVerticesNV``                    -> ``gl_MeshVerticesEXT``
  - ``gl_MeshPrimitivesNV``                  -> ``gl_MeshPrimitivesEXT``
  - ``gl_PrimitiveIndicesNV[3*i+{0,1,2}]``   -> ``gl_PrimitiveTriangleIndicesEXT[i] = uvec3(...)``
  - ``perprimitiveNV``                       -> ``perprimitiveEXT``
  - ``gl_BaryCoordNV``                       -> ``gl_BaryCoordEXT``
  - ``#version 450``                         -> ``#version 460``

Usage:
    python3 ludus_renderer/shaders/nv_to_ext.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TASK_FILES = ["ts_polyline.task.glsl", "ts_polygon.task.glsl", "ts_obstacle.task.glsl"]
MESH_FILES = ["ts_polyline.mesh.glsl", "ts_polygon.mesh.glsl", "ts_obstacle.mesh.glsl"]
FRAG_FILES = ["ts_polyline.frag.glsl", "ts_polygon.frag.glsl", "ts_obstacle.frag.glsl"]

# GLSL type and qualifier keywords that, when followed by a name, mean the
# name is being declared (not used). Used to skip prefixing in declarations.
_TYPE_KEYWORDS = (
    r"u?int|float|double|bool|"
    r"[uib]?vec[234]|mat[234](?:x[234])?|"
    r"sampler\w*|image\w*|atomic_uint|void|"
    r"in|out|inout|const|highp|mediump|lowp|uniform|buffer|shared|"
    r"coherent|volatile|restrict|readonly|writeonly|patch|sample|"
    r"centroid|smooth|flat|noperspective|invariant|precise|layout|"
    r"taskPayloadSharedEXT|perprimitiveEXT|perprimitiveNV"
)


# ---------------------------------------------------------------------------
# Brace-depth aware iteration: yield (start, end) of every TOP-LEVEL struct
# definition body (excluding the surrounding `struct NAME { ... };`).
# ---------------------------------------------------------------------------

def find_struct_bodies(code: str) -> list[tuple[int, int]]:
    """Return character ranges (start_brace_inclusive, end_brace_exclusive)
    that lie inside any ``struct NAME { ... }`` block, including nested."""
    ranges: list[tuple[int, int]] = []
    stack: list[int] = []  # opening brace positions of struct blocks
    inside_struct = 0       # nesting depth of struct{}-tracked braces
    i = 0
    n = len(code)
    while i < n:
        # Skip strings / line comments / block comments to be safe.
        if code.startswith("//", i):
            j = code.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        ch = code[i]
        if ch == "{":
            # Look back: did we just see "struct NAME"?
            head = code[max(0, i - 80):i]
            if re.search(r"\bstruct\s+\w+\s*$", head):
                stack.append(i)
                inside_struct += 1
            else:
                if inside_struct > 0:
                    inside_struct += 1
            i += 1
            continue
        if ch == "}":
            if inside_struct > 0:
                inside_struct -= 1
                if inside_struct == 0 and stack:
                    start = stack.pop()
                    ranges.append((start, i + 1))
            i += 1
            continue
        i += 1
    return ranges


# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------

def translate_common(code: str) -> str:
    code = code.replace("#extension GL_NV_mesh_shader : require",
                        "#extension GL_EXT_mesh_shader : require")
    code = code.replace("#extension GL_NV_fragment_shader_barycentric : require",
                        "#extension GL_EXT_fragment_shader_barycentric : require")
    code = code.replace("perprimitiveNV", "perprimitiveEXT")
    code = code.replace("gl_BaryCoordNV", "gl_BaryCoordEXT")
    code = re.sub(r"#version\s+450(?:\s+core)?", "#version 460", code)
    return code


def convert_task_payload(code: str) -> tuple[str, list[str]]:
    """Convert NV ``taskNV out/in NAME { ... };`` blocks to EXT.

    Returns the rewritten code and the list of field names that appear in
    the payload (so we can later prefix bare references with ``payload.``).
    """
    fields: list[str] = []

    def replacer(m: re.Match) -> str:
        typename = m.group(2)
        body = m.group(3)
        for _, fname in re.findall(r"^\s+(\w+)\s+(\w+)\s*;", body, re.MULTILINE):
            if fname not in fields:
                fields.append(fname)
        # Note: the trailing `;` after the original `}` is consumed by the
        # outer regex pattern via `\s*;?`, so we add a single one below.
        return f"struct {typename} {{{body}}};\ntaskPayloadSharedEXT {typename} payload;"

    code = re.sub(
        r"taskNV\s+(out|in)\s+(\w+)\s*\{([^}]*)\}\s*;?",
        replacer, code, flags=re.DOTALL,
    )
    return code, fields


def find_top_level_bodies(code: str) -> list[tuple[int, int, str]]:
    """Return (body_start, body_end, params_string) ranges of every
    top-level ``{ ... }`` block plus the preceding ``(...)`` parameter list
    (empty string if none). Used to scope local-variable shadow detection
    so that function parameters also shadow payload fields.
    """
    bodies: list[tuple[int, int, str]] = []
    depth = 0
    open_stack: list[int] = []
    i = 0
    n = len(code)
    while i < n:
        if code.startswith("//", i):
            j = code.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        ch = code[i]
        if ch == "{":
            open_stack.append(i)
            depth += 1
        elif ch == "}":
            if open_stack and depth == 1:
                start = open_stack.pop()
                # Look back for matching ( ... ) just before {
                head = code[max(0, start - 256):start]
                pm = re.search(r"\(([^()]*)\)\s*$", head)
                params = pm.group(1) if pm else ""
                bodies.append((start, i + 1, params))
            elif open_stack:
                open_stack.pop()
            depth -= 1
        i += 1
    return bodies


def strip_comments(code: str) -> str:
    """Return a copy of `code` with line / block comments blanked out
    (preserving lengths and newlines) so regexes do not match inside
    comments but offsets still line up with the original string.
    """
    out = list(code)
    n = len(code)
    i = 0
    while i < n:
        if code.startswith("//", i):
            j = code.find("\n", i)
            end = n if j < 0 else j
            for k in range(i, end):
                if out[k] != "\n":
                    out[k] = " "
            i = end
            continue
        if code.startswith("/*", i):
            j = code.find("*/", i + 2)
            end = n if j < 0 else j + 2
            for k in range(i, end):
                if out[k] != "\n":
                    out[k] = " "
            i = end
            continue
        i += 1
    return "".join(out)


def prefix_payload_fields(code: str, fields: list[str]) -> str:
    """Prefix bare uses of payload field names with ``payload.``.

    Skips:
      - Anything inside any ``struct { ... }`` body.
      - Names immediately preceded by a GLSL type / qualifier keyword
        (declarations, function parameters, local variables).
      - References inside top-level function bodies that locally redeclare
        the field name (e.g. ``bool is_bev = ...``) -- that local variable
        shadows the payload field for the remainder of the function.
    """
    if not fields:
        return code

    # Run regex matching on a comment-stripped view so we don't rewrite
    # field names that appear inside `// ...` or `/* ... */`. Splicing is
    # done against the original `code` to preserve comments verbatim.
    scrubbed = strip_comments(code)

    struct_ranges = find_struct_bodies(scrubbed)
    fn_bodies = find_top_level_bodies(scrubbed)

    shadow_decl_re = {
        f: re.compile(
            r"\b(?:" + _TYPE_KEYWORDS + r")\s+" + re.escape(f) + r"\s*[=;,)\]]"
        )
        for f in fields
    }
    param_decl_re = {
        f: re.compile(r"\b(?:" + _TYPE_KEYWORDS + r")\s+" + re.escape(f) + r"\b")
        for f in fields
    }

    shadow_ranges: list[tuple[int, int, str]] = []
    for body_start, body_end, params in fn_bodies:
        body_code = scrubbed[body_start:body_end]
        for f in fields:
            if shadow_decl_re[f].search(body_code) or param_decl_re[f].search(params):
                shadow_ranges.append((body_start, body_end, f))

    def in_struct(pos: int) -> bool:
        for s, e in struct_ranges:
            if s <= pos < e:
                return True
        return False

    def is_shadowed(pos: int, fname: str) -> bool:
        for s, e, f in shadow_ranges:
            if f == fname and s <= pos < e:
                return True
        return False

    type_re = re.compile(r"\b(?:" + _TYPE_KEYWORDS + r")\s*$")
    field_re = re.compile(
        r"(?<!\.)(?<!\w)(" + "|".join(re.escape(f) for f in fields) + r")(?!\w)"
    )

    matches = list(field_re.finditer(scrubbed))
    for m in reversed(matches):
        if in_struct(m.start()):
            continue
        before = scrubbed[max(0, m.start() - 64):m.start()]
        if type_re.search(before):
            continue
        if is_shadowed(m.start(), m.group(1)):
            continue
        code = code[:m.start()] + "payload." + m.group(1) + code[m.end():]
    return code


def translate_task(code: str) -> str:
    """Convert a task shader's NV semantics to EXT."""
    code, fields = convert_task_payload(code)

    code = re.sub(r"gl_TaskCountNV\s*=\s*([^;]+);", r"_task_count = \1;", code)

    if "_task_count" in code:
        # Declare local at the top of main()
        m = re.search(r"void\s+main\s*\(\s*\)\s*\{", code)
        if m:
            insert = m.end()
            code = code[:insert] + "\n    uint _task_count = 0u;" + code[insert:]
        # Emit on every exit path
        code = code.replace("return;", "EmitMeshTasksEXT(_task_count, 1, 1); return;")
        # And on fall-through at the end of main (last '}' in the file)
        last = code.rfind("}")
        code = code[:last] + "    EmitMeshTasksEXT(_task_count, 1, 1);\n" + code[last:]

    code = prefix_payload_fields(code, fields)
    return code


def translate_mesh(code: str) -> str:
    """Convert a mesh shader's NV builtins to EXT."""
    code, fields = convert_task_payload(code)

    max_v = "128"
    mv = re.search(r"max_vertices\s*=\s*(\d+)", code)
    if mv:
        max_v = mv.group(1)

    # Every `gl_PrimitiveCountNV = X;` becomes a `_prim_count = X;` followed by
    # `SetMeshOutputsEXT(MAX_V, _prim_count);`. The single declaration of
    # `_prim_count` lives at the top of main(), inserted below.
    code = re.sub(
        r"gl_PrimitiveCountNV\s*=\s*([^;]+);",
        lambda m: f"_prim_count = {m.group(1).strip()}; SetMeshOutputsEXT({max_v}u, _prim_count);",
        code,
    )
    code = code.replace("gl_PrimitiveCountNV", "_prim_count")
    if "_prim_count" in code:
        main_m = re.search(r"void\s+main\s*\(\s*\)\s*\{", code)
        if main_m:
            code = code[:main_m.end()] + "\n    uint _prim_count = 0u;" + code[main_m.end():]

    code = code.replace("gl_MeshVerticesNV",   "gl_MeshVerticesEXT")
    code = code.replace("gl_MeshPrimitivesNV", "gl_MeshPrimitivesEXT")

    # Triple-write patterns. NV indexes a flat uint array (3 entries per
    # triangle); EXT indexes a uvec3 array (1 entry per triangle).
    triple_patterns = [
        # [3*i+0], [3*i+1], [3*i+2]
        (r"gl_PrimitiveIndicesNV\[\s*3\s*\*\s*(\w+)\s*\+\s*0u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*3\s*\*\s*\1\s*\+\s*1u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*3\s*\*\s*\1\s*\+\s*2u?\s*\]\s*=\s*([^;]+);",
         r"gl_PrimitiveTriangleIndicesEXT[\1] = uvec3(\2, \3, \4);"),
        # [i*3+0], [i*3+1], [i*3+2]
        (r"gl_PrimitiveIndicesNV\[\s*(\w+)\s*\*\s*3u?\s*\+\s*0u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\1\s*\*\s*3u?\s*\+\s*1u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\1\s*\*\s*3u?\s*\+\s*2u?\s*\]\s*=\s*([^;]+);",
         r"gl_PrimitiveTriangleIndicesEXT[\1] = uvec3(\2, \3, \4);"),
        # [i*3], [i*3+1], [i*3+2] -- first without explicit +0u
        (r"gl_PrimitiveIndicesNV\[\s*(\w+)\s*\*\s*3u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\1\s*\*\s*3u?\s*\+\s*1u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\1\s*\*\s*3u?\s*\+\s*2u?\s*\]\s*=\s*([^;]+);",
         r"gl_PrimitiveTriangleIndicesEXT[\1] = uvec3(\2, \3, \4);"),
        # [(base+i)*3+0/1/2]
        (r"gl_PrimitiveIndicesNV\[\s*\(\s*(\w+)\s*\+\s*(\w+)\s*\)\s*\*\s*3u?\s*\+\s*0u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\(\s*\1\s*\+\s*\2\s*\)\s*\*\s*3u?\s*\+\s*1u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\(\s*\1\s*\+\s*\2\s*\)\s*\*\s*3u?\s*\+\s*2u?\s*\]\s*=\s*([^;]+);",
         r"gl_PrimitiveTriangleIndicesEXT[\1 + \2] = uvec3(\3, \4, \5);"),
        # [A*3 + B*3 + 0/1/2]
        (r"gl_PrimitiveIndicesNV\[\s*(\w+)\s*\*\s*3u?\s*\+\s*(\w+)\s*\*\s*3u?\s*\+\s*0u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\1\s*\*\s*3u?\s*\+\s*\2\s*\*\s*3u?\s*\+\s*1u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\1\s*\*\s*3u?\s*\+\s*\2\s*\*\s*3u?\s*\+\s*2u?\s*\]\s*=\s*([^;]+);",
         r"gl_PrimitiveTriangleIndicesEXT[\1 + \2] = uvec3(\3, \4, \5);"),
        # [BASE + 0u], [BASE + 1u], [BASE + 2u] -- single base var with explicit +0u
        (r"gl_PrimitiveIndicesNV\[\s*(\w+)\s*\+\s*0u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\1\s*\+\s*1u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\1\s*\+\s*2u?\s*\]\s*=\s*([^;]+);",
         r"gl_PrimitiveTriangleIndicesEXT[(\1) / 3u] = uvec3(\2, \3, \4);"),
        # [BASE], [BASE + 1u], [BASE + 2u] -- single base var no explicit +0u
        (r"gl_PrimitiveIndicesNV\[\s*(\w+)\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\1\s*\+\s*1u?\s*\]\s*=\s*([^;]+);\s*"
         r"gl_PrimitiveIndicesNV\[\s*\1\s*\+\s*2u?\s*\]\s*=\s*([^;]+);",
         r"gl_PrimitiveTriangleIndicesEXT[(\1) / 3u] = uvec3(\2, \3, \4);"),
    ]
    for pat, repl in triple_patterns:
        code = re.sub(pat, repl, code)

    if "gl_PrimitiveIndicesNV" in code:
        code = code.replace("gl_PrimitiveIndicesNV",
                            "gl_PrimitiveTriangleIndicesEXT /* TODO: needs uvec3 rewrite */")

    code = prefix_payload_fields(code, fields)
    return code


def translate_fragment(code: str) -> str:
    """Fragment shaders only need the common translations.  ``perprimitiveEXT``
    requires ``GL_EXT_mesh_shader`` to be enabled in the fragment stage too."""
    if "perprimitiveEXT" in code and "GL_EXT_mesh_shader" not in code:
        if "#extension GL_EXT_fragment_shader_barycentric : require" in code:
            code = code.replace(
                "#extension GL_EXT_fragment_shader_barycentric : require",
                "#extension GL_EXT_fragment_shader_barycentric : require\n"
                "#extension GL_EXT_mesh_shader : require",
                1,
            )
        else:
            code = re.sub(r"(#version\s+\d+)",
                          r"\1\n#extension GL_EXT_mesh_shader : require", code, count=1)
    return code


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def convert_file(path: str) -> str:
    with open(path) as f:
        src = f.read()
    src = translate_common(src)
    name = os.path.basename(path)
    if name in TASK_FILES:
        src = translate_task(src)
    elif name in MESH_FILES:
        src = translate_mesh(src)
    elif name in FRAG_FILES:
        src = translate_fragment(src)
    return src


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=SCRIPT_DIR,
                        help="Output directory for converted .glsl files (default: in-place).")
    args = parser.parse_args()

    files = TASK_FILES + MESH_FILES + FRAG_FILES
    issues = 0
    os.makedirs(args.out_dir, exist_ok=True)
    for fn in files:
        src_path = os.path.join(SCRIPT_DIR, fn)
        if not os.path.exists(src_path):
            print(f"  missing: {fn}", file=sys.stderr)
            issues += 1
            continue
        out = convert_file(src_path)
        for bad in ("GL_NV_mesh_shader", "taskNV", "gl_TaskCountNV",
                    "gl_PrimitiveCountNV", "gl_MeshVerticesNV", "gl_MeshPrimitivesNV",
                    "perprimitiveNV", "gl_BaryCoordNV"):
            if bad in out:
                print(f"  {fn}: residual {bad!r}", file=sys.stderr)
                issues += 1
        out_path = os.path.join(args.out_dir, fn)
        with open(out_path, "w") as f:
            f.write(out)
        print(f"  wrote {fn} ({len(out)} bytes)")

    if issues:
        print(f"FAILED: {issues} issues")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
