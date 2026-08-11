# AGENTS.md

Guidance for agents working in AbstractOperators.jl.

## Mission

Make changes to operators and their tests reliable and informative without weakening test
intent. Never remove assertions to force green tests. If a failure reflects a real
implementation bug, fix the source instead of loosening the test. Keep changes minimal and
localized; avoid unrelated refactors.

## Repository Layout & Operator Engineering

- `src/linearoperators/` for concrete linear operators.
- `src/nonlinearoperators/` for nonlinear operators.
- `src/calculus/` for composition and operator calculus.
- `src/batching/` for batch operators.
- `ext/GpuExt/` for the GPU package extension (triggered by `GPUArrays`).
- For new or changed operators, keep the implementation complete:
  - constructors,
  - forward `mul!`,
  - adjoint `mul!` where applicable,
  - size/domain/codomain/storage traits,
  - property traits such as linearity, diagonal structure, and rank-related predicates.
- `check` utility function must be called in all effective `mul!` paths to ensure consistent
  argument validation and error messages.
- Preserve `domain_array_type`/`codomain_array_type` semantics and dispatch compatibility;
  keep them consistent with constructor-selected storage.
- Constructors should expose an `array_type` keyword where storage backend selection is
  meaningful.
- When storage checks become stricter, fix operator traits and tests instead of relaxing
  `check`.
- Prefer behavior-preserving refactors: extract helpers, separate setup from kernels, reduce
  method size, but do not weaken checks.
- If modifying copy semantics, preserve the convention that immutable/read-only arrays are
  shared while mutable working buffers are copied deliberately (see
  `copy_operator(op; array_type=nothing, threaded=nothing)`).
- Keep source formatted with Runic-compatible Julia style.

## GPU Implementation

- Julia package extensions can only `import` the parent package, trigger package(s), and
  stdlib; if extension code needs a parent dependency API, expose it from the parent module
  first.
- Override `mul!` in `ext/GpuExt/` for any operator whose base implementation uses scalar
  indexing loops (`@nloops`, `@nref`, `@inbounds y[i] = b[j]`); replace with broadcast-over-view
  (`y .= view(b, idx...)`).
- When overriding a threaded operator (e.g. `Variation{..., true}`) for GPU, delegate to the
  non-threaded variant (`Variation{..., false}`) — threading strategy is CPU-only.
- For FFT plans, prefer `inv(plan)` (AbstractFFTs-generic) over backend-specific
  `FFTW.plan_inv(...)` to keep CUDA/AMDGPU compatibility.
- With JLArrays/GPUArrays, avoid `copyto!(gpu, cpu_view)` where the source is a `SubArray`;
  materialize first (e.g. `src[1:n]`), or copy from a plain array.
- Keep CPU-only implementation details out of GPU overrides unless the backend truly supports
  them.
- For GPU `GetIndex` overrides, keep boolean-mask and integer-vector fancy indexing in CPU
  paths unless the backend support is verified.
- Prefer direct `CuArray(arr)` / `CUDA.zeros(...)` / `AMDGPU.ROCArray(arr)` / `AMDGPU.zeros(...)`
  calls over intermediate conversion variables.

## Performance

- Put performance-critical code in functions, not top-level scope.
- Avoid untyped globals in hot paths; use function arguments and `const` globals where
  appropriate.
- Prefer concrete field/container types; avoid abstract fields like `Function`,
  `AbstractArray`, or `Integer` in performance-sensitive structs.
- Maintain type stability: avoid variable type changes within loops, use `zero(x)`,
  `oneunit(T)`, stable return types, and function barriers for setup-vs-kernel separation.
- Measure, don't guess: use `BenchmarkTools`, track allocations (`@time`, `@allocated`) and
  treat unexpected allocations as defects, use `@code_warntype` and JET to diagnose inference
  issues.
- Minimize allocations in inner loops: preallocate outputs, favor `mul!`/in-place APIs, use
  broadcast fusion (`@.`) when beneficial, unfuse broadcasts when repeated subexpressions are
  recomputed unnecessarily, use `@views` for slicing when copy cost dominates.
- Iterate arrays in memory-friendly (column-major) order.
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
- JET `@test_opt` flags `array_type::Type` (unparameterized keyword) as a source of runtime
  dispatch. Use `array_type::Type{<:AbstractArray}` and avoid kwarg-to-kwarg forwarding; route
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
  supported across GPU backends. Add `domain_array_type`/`codomain_array_type` tests and
  verify `op * x` allocates on the active backend. Migrate GPU-backend storage-type assertions
  into each operator's own CUDA/AMDGPU `@testitem` (e.g.
  `@test domain_array_type(op) <: CUDA.CuArray`) so they run with the functional tests.
- Stochastic test assertions like `op * randn(n) ≈ other_op * (op * randn(n))` are wrong when
  the two `randn` calls produce different vectors — always capture into a variable first.
- Agent sub-tasks frequently generate `Eye(T, dims, array_type)` (3 positional args) instead
  of `Eye(T, dims; array_type=...)` (keyword). Always verify agent output for this pattern.
- All temporary test and benchmark outputs must go under `.temp/` only.
- When `VERB` is enabled, print each running testitem name at test-runner filter time.

## Long-Running Test / Coverage / Benchmark Workflow

1. Start from the smallest relevant test scope; prefer a persistent Julia REPL for repeated
   filtered `TestItemRunner.run_tests(...)` calls.
2. Fix real implementation bugs in source instead of weakening tests; rerun the same filtered
   slice until green, then expand to adjacent slices, then run the full suite.
3. Capture all run logs under `.temp/`.
4. For performance-sensitive changes, benchmark before and after; run focused ASV filters
   first, then a single full ASV comparison for final validation. Treat
   `speedup + uncertainty < 0.95` (master/dirty ratio) as a significant regression.
5. Prefer representative large inputs for linear and nonlinear operators to reduce
   microbenchmark noise, but wrap only fast operators in calculus operators to measure the
   calculus overhead itself.
6. Use AirspeedVelocity with an explicit script path when comparing against revisions that do
   not yet contain the benchmark file.

Recommended REPL pattern:

```julia
using TestItemRunner
run_tests("test"; filter = ti -> (:MatrixOp in ti.tags) && (:linearoperator in ti.tags))
```

Main package coverage (also exercises subpackage and extension code — DSPOperators,
FFTWOperators, NFFTOperators, and WaveletOperators have no standalone `test/` directory, and
extensions are exercised through the parent package's tests):

```sh
julia --project=test --code-coverage=user test/runtests.jl
```

Process coverage after a local run:

```sh
julia -e 'using Coverage; Coverage.LCOV.writefile("lcov.info", Coverage.process_folder())'
```

Filtered test run:

```julia
using TestItemRunner
TestItemRunner.run_tests(pwd(); filter = ti -> :MatrixOp in ti.tags) # by tag
TestItemRunner.run_tests(pwd(); filter = ti -> ti.name == "DCT")     # by test name
```

### Local benchmark comparison with AirspeedVelocity

```sh
mkdir -p .temp/asv
benchpkg \
  --path . \
  --rev master,dirty \
  --script benchmark/benchmarks.jl \
  --output-dir .temp/asv \
  --exeflags="--threads=4"
```

Filtered comparison for a single benchmark family:

```sh
mkdir -p .temp/asv
benchpkg \
  --path . \
  --rev master,dirty \
  --script benchmark/benchmarks.jl \
  --output-dir .temp/asv \
  --exeflags="--threads=4" \
  --add RecursiveArrayTools \
  --filter MIMOFilt
```

Render a comparison table:

```sh
benchpkgtable \
  --path . \
  --rev master,dirty \
  --input-dir .temp/asv \
  --ratio \
  --mode time,memory
```

### CI benchmark comparison (GitHub Actions)

The GitHub Actions benchmark CI does **not** use the AirspeedVelocity action because the
root-level Julia workspace (`[workspace]` in `Project.toml`) causes that action's
revision-management to mis-resolve the monorepo subprojects. Instead, two workflows implement
a fork-safe two-stage approach:

- **`benchmark.yml`** — unprivileged `pull_request` job that checks out both the base and head
  revisions, runs `benchmark/compare.jl` against explicit worktree paths, and uploads
  `body.md`, `pr_number.txt`, and `julia_version.txt` as an artifact.
- **`post_benchmark_comment.yml`** — privileged `workflow_run` job that downloads the artifact
  and creates or updates the PR comment.

The comparison table mirrors AirspeedVelocity output with separate Time and Memory sections,
base/head columns, a ratio column, and emoji indicators:

- 🚀 significant speedup: `ratio − ratio_err > 1.2` (time) or `ratio < 0.5` (memory)
- 🐢 significant slowdown: `ratio + ratio_err < 0.8` (time) or `ratio > 1.5` (memory)

To run the comparison locally with the same script used by CI:

```sh
git worktree add .temp/base master

julia --project=benchmark benchmark/compare.jl \
  --base-dir  .temp/base \
  --head-dir  . \
  --output-dir .temp/bench-compare \
  --pr        0 \
  --julia-version "$(julia -e 'print(VERSION)')"

cat .temp/bench-compare/body.md
```

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
