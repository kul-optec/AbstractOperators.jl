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
- If GPU tests are backend-specific, keep them in separate `@testitem`s and use `:cuda` / `:amdgpu` tags.
- In non-FFTW/non-DSP operator tests, prefer JLArray backend checks over CUDA/AMDGPU device checks.
- Use direct `import CUDA` / `import AMDGPU` + `functional()` guards in testitems; avoid try/catch gating.
- When `VERB` is enabled, print each running testitem name at test-runner filter time.
- For local coverage, mirror CI with `julia --project=test --code-coverage=user test/runtests.jl`, then process `*.cov` / `*.info` artifacts into `lcov.info` if needed.
- Subpackages (DSPOperators, FFTWOperators, NFFTOperators, WaveletOperators) have no standalone `test/` directory; they are tested and their coverage is gathered exclusively through the parent package's `test/` project. Do not attempt a separate subpackage coverage run.
- Extension coverage should be gathered through the parent-package tests that load the relevant trigger packages; do not assume a separate extension-only coverage run exists.
- JET `@test_opt` flags `array_type::Type` (unparameterized keyword) as a source of runtime dispatch. Use `array_type::Type{<:AbstractArray}` and avoid kwarg-to-kwarg forwarding; use a typed positional-arg helper (e.g., `_make_eye(T, dims, S)`) so JET can resolve dispatch statically.
- When Aqua reports "Unexpected Pass" on a `@test_broken`/`broken=true` check, the underlying issue is now fixed — remove the workaround and use `Aqua.test_all(pkg)` unconditionally.
- Agent sub-tasks frequently generate `Eye(T, dims, array_type)` (3 positional args) instead of `Eye(T, dims; array_type=...)` (keyword). Always verify agent output for this pattern.
- Stochastic test assertions `op * randn(n) ≈ other_op * (op * randn(n))` are wrong when the two `randn` calls produce different vectors; always capture into a variable first.
- When testing GPU storage-type propagation, add `@test domain_array_type(op) <: CUDA.CuArray` / `<: AMDGPU.ROCArray` assertions directly in the per-operator CUDA/AMDGPU `@testitem`.
