---
name: test-coverage-benchmark-workflow
description: Run AbstractOperators.jl's long-running test, coverage, and benchmark workflow — filtered TestItemRunner runs, coverage capture, and local/CI AirspeedVelocity benchmark comparisons. Use when running the full test suite, generating coverage, or comparing performance against master.
---

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
