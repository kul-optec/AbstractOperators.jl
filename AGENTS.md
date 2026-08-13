# AGENTS.md

Guidance for agents working in AbstractOperators.jl.

## Mission

Make changes to operators and their tests reliable and informative without weakening test
intent. Never remove assertions to force green tests. If a failure reflects a real
implementation bug, fix the source instead of loosening the test. Keep changes minimal and
localized; avoid unrelated refactors.

## Repository Layout & Operator Engineering

- For new or changed operators, keep the implementation complete:
  - constructors,
  - forward `mul!`,
  - adjoint `mul!` where applicable,
  - size/domain/codomain/storage traits,
  - property traits such as linearity, diagonal structure, and rank-related predicates.
- `check` utility function must be called in all effective `mul!` paths to ensure consistent
  argument validation and error messages.
- Preserve `domain_storage_type`/`codomain_storage_type` semantics and dispatch compatibility;
  keep them consistent with constructor-selected storage.
- Constructors should expose an `storage_type` keyword where storage backend selection is
  meaningful.
- When storage checks become stricter, fix operator traits and tests instead of relaxing
  `check`.
- Prefer behavior-preserving refactors: extract helpers, separate setup from kernels, reduce
  method size, but do not weaken checks.
- If modifying copy semantics, preserve the convention that immutable/read-only arrays are
  shared while mutable working buffers are copied deliberately (see
  `copy_operator(op; storage_type=nothing, threaded=nothing)`).
- Keep source formatted with Runic-compatible Julia style.

GPU extension conventions live in `ext/GpuExt/CLAUDE.md`.

## Performance

- Measure, don't guess: use `BenchmarkTools`, track allocations (`@time`, `@allocated`) and
  treat unexpected allocations as defects, use `@code_warntype` and JET to diagnose inference
  issues.
- Minimize allocations in inner loops: preallocate outputs, favor `mul!`/in-place APIs, use
  broadcast fusion (`@.`) when beneficial, unfuse broadcasts when repeated subexpressions are
  recomputed unnecessarily, use `@views` for slicing when copy cost dominates.
- For threaded Julia code that also calls BLAS, avoid oversubscription (often
  `OPENBLAS_NUM_THREADS=1` is best with multithreaded Julia; validate on workload).
- Use `@inbounds`/`@simd`/`@fastmath` only when correctness assumptions are explicitly
  validated.
- Benchmark setup code should normalize wrapped domain and codomain type traits to scalar
  element types before calling `randn`/`zeros`, use representative large inputs for GPU
  crossover studies, and keep the measurement setup deterministic (`Random.seed!(0)`).

## Testing & JET

- Prefer `@testitem` with explicit tags and optional setup modules; keep test files
  standalone-capable and aligned with TestItems setup modules.
- Use type tags from: `:linearoperator`, `:nonlinearoperator`, `:batching`, `:calculus`,
  `:jet`, `:quality`, `:misc`.
- Operator tags must use exact CamelCase type names, e.g. `:MatrixOp`, `:FiniteDiff`,
  `:Compose`, `:SpreadingBatchOp`. Mixed tests may use multiple operator tags when the
  behavior genuinely spans operators.
- Use `@run_package_tests filter=ti->...` / `TestItemRunner.run_tests(...)` for focused
  slices; use strict tag-exclusion filters for grouped runs (e.g.
  `ti -> !(:jet in ti.tags)`).
- Treat JET as mandatory for all public API, across all three modes in the same change:
  - `JET.test_package(...)` for package-level inference/type diagnostics,
  - `@test_opt` for representative public operations and constructors,
  - `@test_call` for key public call signatures and runtime-like call paths.
  Missing any of the three is an incomplete migration. Public API changes must update JET
  tests in the same change.
- JET `@test_opt` flags `storage_type::Type` (unparameterized keyword) as a source of runtime
  dispatch. Use `storage_type::Type{<:AbstractArray}` and avoid kwarg-to-kwarg forwarding; route
  through a typed positional-arg helper (e.g. `_make_eye(T, dims, S)`) so JET can resolve
  dispatch statically.
- Keep Aqua and doctests passing alongside functional tests. When Aqua reports "Unexpected
  Pass" on a `@test_broken`/`broken=true` check, the underlying issue is fixed — remove the
  workaround and use `Aqua.test_all(pkg)` unconditionally.
- If GPU tests are backend-specific, keep them in separate `@testitem`s with `:cuda`/`:amdgpu`
  tags. In non-FFTW/non-DSP operator tests, prefer JLArray backend checks over CUDA/AMDGPU
  device checks. Use direct `import CUDA`/`import AMDGPU` + `functional()` guards in
  testitems; avoid try/catch gating. Restrict GPU `GetIndex` test indices to ranges, colons,
  and scalar integers — bool-mask and integer-vector `view` forms are not universally
  supported across GPU backends. Add `domain_storage_type`/`codomain_storage_type` tests and
  verify `op * x` allocates on the active backend. Migrate GPU-backend storage-type assertions
  into each operator's own CUDA/AMDGPU `@testitem` (e.g.
  `@test domain_storage_type(op) <: CUDA.CuArray`) so they run with the functional tests.
- Stochastic test assertions like `op * randn(n) ≈ other_op * (op * randn(n))` are wrong when
  the two `randn` calls produce different vectors — always capture into a variable first.
- Agent sub-tasks frequently generate `Eye(T, dims, storage_type)` (3 positional args) instead
  of `Eye(T, dims; storage_type=...)` (keyword). Always verify agent output for this pattern.
- All temporary test and benchmark outputs must go under `.temp/` only.
- When `VERB` is enabled, print each running testitem name at test-runner filter time.

The long-running test/coverage/benchmark workflow (filtered TestItemRunner runs, coverage
capture, local and CI AirspeedVelocity comparisons) is documented in the
`test-coverage-benchmark-workflow` skill.

## Failure Triage

1. Read the exact failing assertion and stacktrace first.
2. Classify the failure: test setup/import/tagging issue, real source bug, or
   environment/performance instability.
3. For real bugs, patch source and keep/assert expected behavior in tests.
4. For flaky perf tests, stabilize methodology (workload, sampling, thresholds) without
   dropping coverage.
5. Re-run the smallest relevant filtered subset before broad reruns.

## Output Requirements

- Report what was changed and why; list files touched.
- Provide the exact filtered test commands used and state pass/fail counts for the final run.
- Call out remaining risks or follow-up items.
- Store all temporary run outputs only under `.temp/` inside the repository.
- When performance work is included, report allocation deltas and the exact benchmark commands
  used.
