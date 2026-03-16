---
description: "Use when optimizing Julia code Covers type stability, allocations, preallocation, views, threading, BLAS oversubscription, and measurement discipline."
name: "Julia Performance"
applyTo: "src/**/*.jl,benchmark/**/*.jl"
---

# Julia Performance

## Core Principles

- Keep hot paths inside functions, not top-level scope.
- Avoid untyped globals in performance-sensitive paths; use arguments and `const` globals when needed.
- Favor concrete field and container types; avoid abstractly typed hot fields.
- Preserve type stability:
  - avoid changing variable type in loops,
  - prefer stable return types,
  - use function barriers to separate setup from kernels.
- Prefer in-place APIs and preallocation over repeated temporary allocations.
- Use `@views` when slicing would otherwise allocate unnecessarily.
- Respect Julia's column-major memory order when writing loops.
- Use `@inbounds`, `@simd`, and `@fastmath` only when their correctness assumptions are justified.
- For threaded Julia code that also uses BLAS, avoid oversubscription and benchmark with explicit thread settings.
- Measure performance changes instead of guessing:
  - benchmark representative workloads,
  - inspect allocations,
  - use JET and `@code_warntype` for inference issues.
