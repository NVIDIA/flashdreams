---
name: python-docstring-style
description: Write Python docstrings and inline comments matching the flashdreams house style — SPDX header, one-line module docstring, Google-style function docstrings (Args/Returns/Raises), PEP 257 attribute docstrings on dataclass/class fields *and on module-level constants*, double-backticks for code references, imperative first sentences, and signpost-style inline block comments (kept, not stripped, on a tightening pass). Use when authoring or editing any .py file under flashdreams/, when adding a new module/class/function/field/constant, when polishing comments, or when the user asks about docstring or comment style.
---

# Python docstring style (flashdreams)

House style distilled from `flashdreams/`. Match it when adding or editing Python code so agent output is indistinguishable from human-written code.

## File header

Every `.py` file starts with the SPDX + Apache-2.0 block, then a blank line, then the module docstring, then a blank line, then imports.

```python
# SPDX-FileCopyrightText: Copyright (c) <YEAR> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Block KV cache for causal attention with a fixed-size local window."""
```

**`<YEAR>` is the *current* calendar year, not a hardcoded literal.** Before stamping the header into a brand-new file, look up today's date (the agent host's clock, the system context, or `date +%Y` in a shell) and substitute it. The conversation's training-cutoff year is *not* the source of truth — emit e.g. `Copyright (c) 2027 …` if the file is being created in 2027.

When **editing an existing file**, leave the existing year alone — the SPDX year reflects when the file was first authored, not when it was last touched. Only update the year if the file genuinely had no header before, or you're explicitly asked to refresh copyright years across the tree.

## Module docstring

**One line**, noun phrase (not imperative), describes what the module provides — not how.

- `"""Tensor and object splitting/gathering primitives for context parallelism."""`
- `"""Reusable CUDA-graph capture wrapper for stateful inference callables."""`
- `"""Multi-view, HDMap-conditioned Cosmos DiT for streaming alpadreams."""`

If one line genuinely doesn't fit, use a one-line summary + blank line + wrapped prose, but prefer tightening the summary.

## Inventories in headers — short, concrete, kept current

A small inventory in a module / class header (the kind of "what's in here" line you'd want when you `cd` into the file for the first time) is *helpful*, not harmful, as long as it stays accurate. Agents are good at keeping these in sync, so the old "rarely get updated, drop them" rule is too strict for an agent-edited codebase.

Keep an inventory when it's:

- **Short and concrete** — one parenthetical or one comma-separated tail, not a paragraph.
- **About a stable, named surface** — sub-modules, public helpers in this file, supported model versions, supported backends. Things that the next reader gains real orientation from.
- **Easy to verify against the file** — if a reader can scan the file and check the line, it'll get fixed when something is added or removed.

Examples that should stay:

- `"""Camera-pose math: SE(3) helpers, relative poses, and Plücker rays."""`
- `"""Unified Wan inference pipeline (Wan 2.1 / Wan 2.2, T2V and I2V)."""`
- `"""Text encoders (Cosmos Qwen, UMT5)."""` — when this file genuinely defines exactly those two and adding a third is the kind of edit that touches this docstring anyway.

Still avoid:

- **Inventories of every kwarg / every option / every internal class.** Those belong in `Args:` / attribute docstrings / autodoc — the `Literal[...]` in the signature *is* the source of truth, so don't echo it in prose.
- **Loose, sprawling inventories** that try to enumerate everything the module / class touches: `"""Core building blocks shared by all recipes (attention, checkpointing, distributed, I/O, configs, …)."""` — at this point the inventory is no longer load-bearing, just decoration.

Class / function summary anti-examples:

- Bad: `"""Native attention (math / efficient / cudnn / flash) with bhsd|bshd layout."""` — duplicates the `Literal[...]`.
- Good: `"""Native attention module with configurable QKV layout and SDPA backend."""`
- Bad: `"""Configure attention format (bshd|bhsd) and backend (math, efficient, cudnn, flash)."""`
- Good: `"""Configure attention format and backend."""`

When in doubt: a one-line inventory that names *files / public symbols / model versions* is fine; an inline inventory of *every parameter value* belongs in the signature.

## Function / method docstring

Google style. Imperative first sentence. Blank line before the sections. `Args:`, `Returns:`, `Raises:` — include only what applies. Use double backticks for code references (``` ``x`` ```, ``` ``None`` ```, ``` ``x.shape[seq_dim] // cp_size`` ```).

```python
def split_inputs_cp(
    x: Tensor, seq_dim: int, cp_group: ProcessGroup | None = None
) -> Tensor:
    """Slice a tensor along ``seq_dim`` to this rank's CP shard.

    Args:
        x: Input tensor.
        seq_dim: Dimension to split along (negative indexing supported).
        cp_group: CP process group; ``None`` returns ``x`` unchanged.

    Returns:
        Contiguous slice of length ``x.shape[seq_dim] // cp_size``.

    Raises:
        AssertionError: ``seq_dim`` is not divisible by the CP size.
    """
```

Rules:

- First sentence imperative (`"Slice …"`, `"Capture …"`, `"Return …"`), one line, ends with a period.
- Don't repeat the type annotation in `Args:` — only the semantic meaning and any non-obvious constraints.
- Describe `None` / default behavior inline: `"cp_group: CP process group; ``None`` returns ``x`` unchanged."`.
- When a parameter is typed `T | None` for *signature reasons* (e.g. matching a base class) but is **required at runtime**, explain *why* the type is optional. A bare `"required (raises if None)"` is not enough — the next reader will ask "then why is it Optional?". Example:

  ```text
  cache: Per-rollout encoder cache. Typed ``Optional`` only to match
      the :class:`Encoder` base signature (some encoders are stateless);
      this encoder advances per-AR-step state in ``cache`` and asserts
      when it is ``None``.
  ```

- `Returns:` describes shape / semantics, not the type (type is in the signature).
- Omit `Raises:` for purely internal asserts that callers shouldn't reason about; include it when a caller might want to catch / pattern-match.
- Skip docstrings entirely for trivial private helpers (`_prefix`) whose body is self-explanatory. When in doubt, write one.

## Class docstring

Short one-liner on the `"""` line when the class is self-contained:

```python
@dataclass(kw_only=True)
class CosmosTransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for the Cosmos transformer."""
```

Multi-paragraph when the class has non-obvious structure, phases, or usage order. Use **custom section labels** (not Google's set) when they genuinely help — `Phases:`, `Per-step usage:`, `Note:`, `Typical usage example:`:

```python
class BlockKVCache:
    """
    KV cache for causal attention with a fixed-size local window and CUDA-graph support.

    Keys and values can have arbitrary shape ``[..., total_size, ...]``; the sequence
    (rolling) dimension is given by ``seq_dim`` (dimension index, can be negative).

    Phases:
        - Filling: cache not yet full; tokens are written contiguously;
          ``cached_k()`` / ``cached_v()`` return only the valid prefix.
        - Steady-state: cache full; each new chunk triggers a left-roll of the
          local window and overwrites the rightmost positions;
          ``cached_k()`` / ``cached_v()`` return the full buffer.

    Per-step usage:
        1. before_update(chunk_idx) — prepare (roll local window if steady-state).
        2. update(k, v) — write the new chunk's keys/values into the cache.
        3. cached_k() / cached_v() — get cached keys/values for attention.
        4. after_update(chunk_idx) — update internal bookkeeping.
    """
```

Do **not** use an `Attributes:` section in the class docstring — document attributes individually (see next section).

## Dataclass / class field docstrings

PEP 257 attribute docstrings: a triple-quoted string placed **on the line(s) after** each field declaration. Not `#` comments. Applies to both `@dataclass` fields and regular class attributes.

```python
@dataclass
class BlockKVCache:
    k_shape: tuple[int, ...]
    """Shape of the keys. Must be the same as the values shape except for the last dimension."""

    seq_dim: int
    """Sequence dimension that will be rolled. Can be negative."""

    sink_size: int = 0
    """Number of sink tokens at the start of the cache that are never evicted. Defaults to 0."""

    _prev_chunk_idx: int = -1
    """Chunk index of the last written chunk; -1 when empty."""
```

Rules:

- One sentence per attribute unless it has real nuance. Wrap at ~88 chars.
- Include the default value's meaning when it's non-obvious (`"Defaults to 0."`, `"-1 when empty."`).
- Document private (`_prefix`) fields too when their invariants matter.

## Module-level constants

Use the same PEP 257 attribute-docstring convention for module-level constants whenever the constant has a non-obvious value, a tuning rationale, or a cross-file invariant. This way the rationale shows up as a hover tooltip in editors and is reachable by autodoc — not just in a `#` comment that no tool can find.

Good (rationale + tooltip-discoverable):

```python
_DEFAULT_MODEL_CHANNELS = 128
"""``head_dim = model_channels // num_heads`` must be a size cuDNN's
flash-attention supports; 64 is safe, 16/8 silently NaN."""

_DEFAULT_DTYPE: torch.dtype = torch.bfloat16
"""Encoder / decoder dtype — kept in lock-step with
``TemplateTransformerConfig.dtype`` so ``input_proj`` doesn't see
mismatched control + latent dtypes."""
```

Avoid the pure `#` form when there's a real reason for the value:

```python
# ``head_dim = model_channels // num_heads`` must be a size cuDNN's
# flash-attention supports; 64 is safe, 16/8 silently NaN.
_DEFAULT_MODEL_CHANNELS = 128
```

Skip the docstring entirely for self-evident constants — `_DEFAULT_NUM_HEADS = 2` doesn't need one.

When converting `#`-comments-above-constant into PEP 257 docstrings, leave one blank line before the next constant so each name's docstring is unambiguously attached.

## Inline comments

Comments earn their place when they make the next read of this code faster. Three legitimate roles:

1. **Why / invariant** — non-obvious motivation or a constraint the code can't express.
2. **Signpost a multi-line block** — a one-line lead-in that lets the reader skim past the next 3–10 lines instead of parsing them. This is *not* "narration"; it is structural orientation.
3. **Local landmark on a tricky one-liner** — a few words next to a line whose intent isn't obvious from the names alone.

When you're trimming or auditing a file, **be conservative about deleting existing comments**. The goal is "what helps the next reader", not "minimum comment count". A comment that would have helped you a moment ago when reading the block almost certainly helps the next person too — keep it.

Good — signpost on a block:

```python
# Normalize seq_dim to a non-negative index so downstream
# indexing math doesn't have to special-case negatives.
self.seq_dim = self.seq_dim if self.seq_dim >= 0 else self.seq_dim + tensor_dim
```

```python
# Hoist per-block KV pre-update out of the (graph-captured) network
# forward; predict_flow runs with eager_mode=False.
self.autoregressive_index = autoregressive_index
self.network_cache.before_update(autoregressive_index)
```

Good — local landmark:

```python
seq_dim = x.ndim + seq_dim  # bring it to positive dimension
```

Bad — pure restatement of the very next line:

```python
# Increment counter
self._n_cached += self.chunk_size

# Initialize k
self._k = torch.empty(self.k_shape, device=self.device, dtype=self.dtype)
```

Bad — restating an assertion's own message string immediately above / below it:

```python
# k and v must have the same shape except for the last dimension
assert self.k_shape[:-1] == self.v_shape[:-1], (
    "k and v must have the same shape except for the last dimension"
)
```

(The flashdreams code has occasional "initialize k and v" style comments — treat those as drift, not the standard; don't reproduce them in new code.)

Rule of thumb when deciding whether to keep an inline comment:

- Does it summarize the next *block* (≥ 2–3 lines)? → keep.
- Does it restate the line directly under it? → drop, or rename a variable instead.
- Does it duplicate an `assert` message word-for-word? → drop, the message is the comment.

Other conventions:

- TODO comments: `# TODO: short description` (no owner handle, no date).
- Don't annotate a change you're making — code is read later without that context. Example of what **not** to write: `# Fixed bug where X happened` or `# Changed to use Y for performance`.

## Section dividers within a file

Use `##` double-hash comments as visual dividers between logical sections. Seen throughout the codebase:

```python
## Default camera names / view-index mapping

DEFAULT_CAMERAS: tuple[str, ...] = (...)

## Per-rollout cache

@dataclass(kw_only=True)
class CosmosTransformerCache(TransformerAutoregressiveCache):
    ...
```

Place on its own line, one blank line before and after, short title. Don't use `# ---` rule lines or box-comments.

## Formatting / voice

- Line length: follow the file (most of the repo wraps around 88 chars; match what you see).
- Backticks: double backticks `` ``x`` `` inside docstrings for code, not single or triple. Single backticks in docstrings try to resolve as cross-references and will emit warnings when they can't.
- Shapes: annotate with double-backtick literals when it helps — `` ``[B, V, T, 1, H, W]`` ``. Use this liberally for tensor args.
- Voice: third person descriptive for docstring summaries of classes/attributes ("Long-lived AR cache…"); imperative for functions/methods ("Slice…", "Capture…").
- No emojis in docstrings or comments.

## Sphinx / Napoleon compatibility

Docs are built with `sphinx.ext.napoleon` (Google style) + `sphinx.ext.autodoc`, and `warningiserror = True` in `docs/source/conf.py` — **any malformed rST breaks CI**.

Practical rules:

- **Docstrings are reStructuredText, not Markdown.** Don't use `#` / `##` headings, `[text](url)` links, or triple-backtick code fences inside a docstring. For code blocks use `::` + an indented block, or the `Example:` / `Examples:` section (Napoleon renders it as a code block).
- **Cross-references use rST roles**, not double backticks, when you want a hyperlink:
  - `` :class:`CUDAGraphWrapper` ``, `` :meth:`BlockKVCache.update` ``, `` :func:`split_inputs_cp` ``, `` :attr:`BlockKVCache._k` ``, `` :mod:`flashdreams.infra.cuda_graph` ``, `` :obj:`None` ``.
  - Plain double backticks are still correct for *non-linked* literal code (argument names, shapes, small expressions).
- **Section labels Napoleon recognises** (and converts into rST admonitions / field lists):
  `Args`, `Arguments`, `Attention`, `Attributes`, `Caution`, `Danger`, `Error`, `Example`, `Examples`, `Hint`, `Important`, `Keyword Args`, `Keyword Arguments`, `Methods`, `Note`, `Notes`, `Other Parameters`, `Parameters`, `Return`, `Returns`, `Raise`, `Raises`, `References`, `See Also`, `Tip`, `Todo`, `Warning`, `Warnings`, `Warn`, `Warns`, `Yield`, `Yields`.
- **Custom section labels** like `Phases:`, `Per-step usage:`, `Typical usage example:` are **not** Napoleon-recognised. They render as plain paragraphs at best, and a stray blank-line can turn them into field-list warnings under `warningiserror`. Prefer `Note:` / `Example:` / `Examples:` (Napoleon-recognised) for anything callout-shaped. If a genuinely custom section is unavoidable, register it via `napoleon_custom_sections` in `conf.py` before using it in code.
- **`Attributes:` section is legal**, but we document fields with PEP 257 attribute docstrings (see above) and let autodoc discover them — don't duplicate.

## Developer voice — don't write a commit-log

Docstrings and comments are read by the next developer opening the file, not by a reviewer scanning a diff. Describe the *interface* and any invariant a caller must not violate. Everything else — why the change was made, what it used to be, kernel / tolerance archaeology — belongs in a commit message or a review thread.

Cut, on sight:

- **Meta-narration.** `"Previously we..."`, `"As of this change..."`, `"now reads runtime state populated by..."`, `"no longer baked into the config"`. The diff tells that story.
- **Decision logs that only restate the default.** `# Deliberately chosen so len_t != window_size_t / 2 != 1 — equal dims can mask off-by-one bugs.` The field's default is authoritative; if an invariant matters, state it once in one line, not as a paragraph.
- **Retelling what a called function does.** Let the `:meth:` / `:func:` cross-reference carry it. `"Delegates to :meth:`X`."` is enough.
- **"Callers who want Z patch on top via derive_config."** Obvious from the config module's contract — don't repeat in every builder.
- **Example rollouts written as prose.** A `.. code-block:: bash` or the test file is cheaper to read.
- **Tolerance archaeology.** One line for the floor (`"bf16 + TF32 ⇒ ~5e-2"`), not a bulleted breakdown per kernel.
- **"Programming error" when an `assert` already says so.** The assert's message *is* the docstring for that case; a one-line summary plus the assert is enough.
- **Restating the next 3 lines of code.** Rename a variable instead.

Two before/after pairs:

Bad (decision log + restated default):

```python
# Deliberately chosen so ``len_t != window_size_t / 2 != 1`` — equal
# temporal dims can mask off-by-one bugs in the KV cache's
# filling / steady bookkeeping. ``height`` and ``width`` are
# supplied per rollout (see :meth:`…initialize_autoregressive_cache`).
len_t: int = 2
"""Pre-flatten latent frames per AR chunk."""
```

Good:

```python
len_t: int = 2
"""Pre-flatten latent frames per AR chunk."""
```

Bad (inventory of what the function touches):

```python
"""Build a fully seeded cache for a new rollout.

Routes raw ``context`` (and optional ``negative_context``) through
:attr:`context_encoder` to produce the per-token context embeddings
consumed by every block forward. Also stashes the per-rollout
``(batch_size, height, width)`` and (when ``config.use_cuda_graph``
is set) builds fresh :class:`CUDAGraphWrapper` instances sized to
this rollout.
"""
```

Good:

```python
"""Build a fully seeded cache for a new rollout.

Runs ``context`` (and ``negative_context`` when CFG is on) through
:attr:`context_encoder`, stashes the per-rollout
``(batch_size, height, width)``, and — when ``config.use_cuda_graph``
is set — builds fresh :class:`CUDAGraphWrapper` instances sized to
this rollout.
"""
```

### Polishing pass order

When asked to tighten an existing file:

1. Strip meta-narration and decision logs (commit-log voice, "previously…", "now reads…").
2. Collapse any paragraph that restates adjacent field docstrings.
3. Promote shape annotations out of prose into ``[B, L, C]`` tokens.
4. Convert `"This method returns…"` → `"Return …"`; `"Will raise…"` → `"Raises…"`.
5. Cut any `Args:` entry that only repeats the parameter name and type.
6. Promote useful `#`-comment-on-a-constant lines into PEP 257 attribute docstrings on the constant.

**Do not** sweep through and delete every inline comment in the name of "no narration". Inline comments that signpost a multi-line block (see [Inline comments](#inline-comments)) are part of the house style — a tightening pass should keep them, only deleting the ones that genuinely restate the next line or duplicate an `assert` message.

## Anti-patterns — don't adopt

- reStructuredText `:param x:` / `:returns:` field lists — we use Google style. Napoleon tolerates mixing, but the rendered output is inconsistent.
- NumPy-style docstrings with `Parameters` / `----------` underlines — disabled via `napoleon_numpy_docstring = False`, renders wrong.
- Markdown inside docstrings (headings, fenced code blocks, hyperlink syntax) — not rST, will not render and may warn.
- Single backticks in docstrings for non-references — Sphinx treats them as unresolved cross-references under the default role and warns.
- Type info duplicated in `Args:` (`x (Tensor): …`) — the signature already has the type, and autodoc renders it.
- Boilerplate "This function does X" opener — start with the verb directly.
- Commit-log voice: `"Now uses…"`, `"Migrated from…"`, `"Originally…"`, `"Deliberately chosen so…"`. Describe the current contract, not the history.
