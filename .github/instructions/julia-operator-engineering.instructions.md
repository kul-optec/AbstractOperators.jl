---
description: "Use when editing Julia operator implementations in AbstractOperators.jl. Covers operator completeness, package boundaries, storage traits, and behavior-preserving refactors."
name: "Julia Operator Engineering"
applyTo: "src/**/*.jl"
---

# Julia Operator Engineering

- Respect package boundaries:
  - `src/linearoperators/` for concrete linear operators.
  - `src/nonlinearoperators/` for nonlinear operators.
  - `src/calculus/` for composition and operator calculus.
  - `src/batching/` for batch operators.
- For new or changed operators, keep implementation complete:
  - constructors,
  - forward `mul!`,
  - adjoint `mul!` where applicable,
  - size/domain/codomain/storage traits,
  - property traits such as linearity, diagonal structure, and rank-related predicates.
- `check` utility function must be called in all effective `mul!` paths to ensure consistent argument validation and error messages.
- Preserve `domain_storage_type` and `codomain_storage_type` semantics and dispatch compatibility.
- Prefer behavior-preserving refactors: extract helpers, separate setup from kernels, reduce method size, but do not weaken checks.
- If modifying copy semantics, preserve the package convention that immutable/read-only arrays are shared while mutable working buffers are copied deliberately.
- Keep source formatted with Runic-compatible Julia style.
