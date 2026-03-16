---
description: "Use when editing Julia tests, TestItemRunner suites, JET coverage, Aqua checks, or doctests in AbstractOperators.jl. Covers tags, logging, and quality gates."
name: "Julia Testing And JET"
applyTo: "test/**/*.jl,docs/**/*.md"
---

# Julia Testing And JET

- Prefer `@testitem` with explicit tags and optional setup modules.
- Use type tags from: `:linearoperator`, `:nonlinearoperator`, `:batching`, `:calculus`, `:jet`, `:quality`, `:misc`.
- Operator tags must use exact CamelCase type names, for example `:MatrixOp`, `:FiniteDiff`, `:Compose`, `:SpreadingBatchOp`.
- Mixed tests may use multiple operator tags when the behavior genuinely spans operators.
- Use strict TestItemRunner filters when slicing the suite.
- Treat JET as mandatory for all public API:
  - `JET.test_package(...)`
  - `@test_opt`
  - `@test_call`
- Public API changes must update JET tests in the same change.
- Keep Aqua and doctests passing alongside functional tests.
- Never remove assertions to force green tests.
- All temporary test and benchmark outputs must go under `.temp/` only.
