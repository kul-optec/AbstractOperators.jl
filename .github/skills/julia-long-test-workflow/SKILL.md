---
name: julia-long-test-workflow
description: 'Use for long-running Julia test suites, TestItemRunner filtering, JET triage, benchmark-driven refactoring, and AirspeedVelocity branch-vs-master comparisons in AbstractOperators.jl.'
argument-hint: 'Describe the operator, test group, or benchmark comparison you want to run'
user-invocable: true
---

# Julia Long Test Workflow

## When To Use

- Iterating on a failing Julia test suite that is too slow to rerun wholesale.
- Narrowing failures with TestItemRunner tags or filenames.
- Verifying JET coverage for public API.
- Refactoring performance-sensitive operator code and checking for regressions.
- Comparing the current branch against `master` using AirspeedVelocity.

## Workflow

1. Start from the smallest relevant test scope.
2. Prefer a persistent Julia REPL for repeated filtered `TestItemRunner.run_tests(...)` calls.
3. Fix real implementation bugs in source instead of weakening tests.
4. Capture all run logs under `.temp/`.
5. For performance-sensitive changes, benchmark before and after.
6. Run focused ASV filters first, then a single full ASV comparison for final validation.
7. Treat `speedup + uncertainty < 0.95` (master/dirty ratio) as a significant regression.
8. Prefer representative large inputs for linear and nonlinear operators to reduce microbenchmark noise, but wrap only fast operators in calculus operators to measure the calculus overhead itself.
9. Use AirspeedVelocity with an explicit script path when comparing against revisions that do not yet contain the benchmark file.

## Common Commands

Filtered test run:

```julia
using TestItemRunner
TestItemRunner.run_tests(pwd(); filter = ti -> :MatrixOp in ti.tags) # example of filtering by tag
TestItemRunner.run_tests(pwd(); filter = ti -> ti.name == "DCT") # example of filtering by test name instead of tags
```

AirSpeedVelocity comparison:

```sh
benchpkg \
  --path . \
  --rev master,dirty \
  --script benchmark/benchmarks.jl \
  --output-dir .temp/asv \
  --exeflags="--threads=4"
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

## Done Criteria

- Targeted tests pass.
- JET coverage remains complete for public API touched.
- Benchmark deltas are measured and reported.
- Logs are saved under `.temp/`.
