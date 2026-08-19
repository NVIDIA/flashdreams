# FlashDreams — AI Agent Context

## Purpose

FlashDreams is a GPU-heavy inference and serving library for autoregressive
video and world models. Researchers will modify the code for new models,
hardware, and experiments, so optimize for code that is easy to understand and
change rather than code that anticipates every possible use.

These rules apply to all newly written or modified code, even when nearby code
uses older patterns.

## Coding Principles

- **Readability over flexibility.** Add an abstraction only when it reduces net
  complexity. Prefer direct code over frameworks, registries, wrappers, and
  extension points created for hypothetical reuse.
- **Solve the present problem.** Implement the smallest complete behavior the
  request needs. Do not bundle adjacent features, compatibility layers, or
  speculative generality.
- **One way to do it.** Each concept should have one representation and one
  execution path. Remove superseded paths instead of keeping aliases, fallback
  branches, or parallel APIs.
- **Locality of behavior.** Keep related state and logic together. Every
  indirection must improve readability or provide genuine reuse across several
  call sites.
- **Configuration has one owner.** Do not add a CLI flag, environment variable,
  constant default, and config field for the same setting. Extend the existing
  authoritative configuration path.
- **Fail fast.** Reject malformed or unsupported input at the boundary. Do not
  silently substitute defaults, swallow exceptions, or continue through a
  partially initialized state.
- **Preserve dependency direction.** Generic runtime code must not know about a
  model integration. Model-specific behavior belongs with that integration.
- **Comments describe current code.** Never leave migration history, “temporary”
  notes without an issue, or commentary about what code used to do.
- **Google Python Style Guide** is the general baseline.

## Simplicity Rules

### Minimize abstraction layers

Add a class or wrapper only when it removes meaningful duplication across at
least three call sites or represents a real domain object.

```python
# Discouraged: one-use delegation.
class SessionStarter:
    def start(self, runtime, inputs):
        return runtime.start_session(inputs)

# Preferred.
session = runtime.start_session(inputs)
```

Use dataclasses for domain objects, protocol payloads, persisted artifacts, and
configuration schemas. Do not create dataclasses that merely bundle arguments
for one local call.

### Avoid defensive attribute probing

Do not use `getattr(...)` chains to guess which interface an object implements.
Define the expected interface and call it directly. Let missing attributes fail
at the actual programming error.

### Avoid unnecessary constants

Inline file-local literals by default. Introduce a named constant when multiple
components must agree on the value and drift would cause a subtle failure, such
as a wire-protocol key or environment variable name.

### No dead-code compatibility

FlashDreams is evolving quickly. Unless the user explicitly requires a staged
migration, delete old constructors, parameters, aliases, adapters, and branches
when replacing them. A second path is a maintenance cost, not free safety.

### Explicit signatures and useful types

Prefer explicit keyword parameters over argument bags. Add local annotations
when a boundary returns an untyped value and naming the concrete type improves
understanding. Do not add annotations that force casts or repeat obvious types.

## Error Handling

- Use normal exceptions and preserve their traceback.
- Validate external data, user input, checkpoint metadata, and transport
  messages at their boundary.
- Do not catch an exception unless the code can recover, add actionable context,
  or complete required cleanup.
- Never use a broad fallback to hide a missing dependency, unsupported backend,
  or failed optimization.
- “Auto” behavior must report what it selected and why; it must not silently
  change model semantics or output quality.

## Documentation and Comments

- Every public method needs a clear docstring describing behavior, arguments,
  return value, and important failure modes.
- Private methods need concise docstrings when their purpose is not obvious.
- Prefer plain language over architecture jargon.
- Explain invariants and non-obvious constraints, not line-by-line mechanics.
- Keep docs, examples, and launch commands synchronized with real entry points.
- Use the repository SPDX header on new source files.

## Tests Prove Behavior

Write a small number of high-signal tests for:

- user-visible behavior and regressions;
- numerical or state-machine correctness;
- validation at external boundaries;
- parity between two paths that claim identical semantics;
- failure and cleanup behavior that local reasoning cannot prove.

Do not write tests that:

- restate constants, config defaults, package metadata, or function signatures;
- verify that a mock was called without asserting the resulting behavior;
- snapshot implementation details instead of a public outcome;
- duplicate checks already guaranteed by static typing;
- require a GPU when a CPU/meta-tensor test proves the same contract.

Every pytest test must carry exactly one marker: `ci_cpu`, `ci_gpu`, or
`manual`. Use module-level `pytestmark = pytest.mark.ci_cpu` for CPU-safe test
modules. Plain `pytest` includes manual tests; normal validation should use a
marker expression.

## Source of Truth

- `SECURITY.md`: vulnerability reporting.
