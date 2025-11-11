# LinearMaps Wrapper Extension

The `LinearMapsExt` extension integrates `AbstractOperators` with the [`LinearMaps.jl`](https://github.com/Jutho/LinearMaps.jl) ecosystem, allowing any linear `AbstractOperator` with matching domain and codomain element types to be used as a `LinearMap`.

## Overview

`LinearMaps.jl` provides a lightweight interface for matrix-free linear operators acting on vectors. This extension wraps a linear `AbstractOperator` inside a `LinearMapWrapper`. The most significant difference between the two abstractions is that `AbstractOperators` support multi-dimensional array domains and codomains, while `LinearMaps.jl` assumes vector inputs and outputs, just like standard matrices. The wrapper performs reshaping between the multi-dimensional array domain/codomain of an `AbstractOperator` and the vector interface required by `LinearMaps.jl`, so that algorithms assuming vector inputs can operate transparently on multi-dimensional operators.

## Requirements

Only operators with identical `domain_type` and `codomain_type` can be wrapped. If the types differ (e.g. `Float64` → `Complex{Float64}`), an error is thrown, because `LinearMaps.jl` assumes a single scalar field for both input and output.

## Construction

Wrap an operator using the exported `LinearMaps.LinearMap` constructor:

```jldoctest
julia> using AbstractOperators, LinearMaps

julia> A = Eye(Float64, (4,4))
I  ℝ^(4, 4) -> ℝ^(4, 4)

julia> LinearMaps.LinearMap(A)
16×16 LinearMap{I  ℝ^(4, 4) -> ℝ^(4, 4)}
```

Attempting to wrap an incompatible operator:

```jldoctest
julia> using AbstractOperators, LinearMaps

julia> A = DiagOp(Float64, (4, 4), rand(ComplexF64, (4, 4)))
╲  ℝ^(4, 4) -> ℂ^(4, 4)

julia> LinearMaps.LinearMap(A)
ERROR: LinearMapsExt.LinearMap only supports operators with matching domain and codomain types
```

## Shape Semantics

- `size(L::LinearMapWrapper)` returns a 2-tuple `(m,n)` where `m = prod(size(A,1))` and `n = prod(size(A,2))`.
- Internally, `mul!` reshapes the input and output vectors to the original multidimensional shapes of the wrapped operator before delegating to `AbstractOperators.mul!`.

This means algorithms working on vectors (Krylov solvers, iterative methods) can operate transparently while the underlying operator still benefits from multidimensional structure.

## Multiplication

The extension implements the internal `LinearMaps._unsafe_mul!` methods:

```julia
LinearMaps._unsafe_mul!(y::AbstractVector, L::LinearMapWrapper, x::AbstractVector)
LinearMaps._unsafe_mul!(y::AbstractVector, L::TransposeOrAdjointWrapper, x::AbstractVector)
```

These perform:
1. Reshape `x` into `size(A,2)`.
2. Reshape `y` into `size(A,1)`.
3. Delegate to `mul!(..., A, ...)` or its adjoint.
4. Reshape `y` back into a vector.

## Forwarded Traits

All structural and algebraic properties defined for the underlying `AbstractOperator` are forwarded:

```juliajulia
is_linear(L::LinearMapWrapper) = is_linear(L.A)
is_symmetric(L::LinearMapWrapper) = is_symmetric(L.A)
is_orthogonal(L::LinearMapWrapper) = is_orthogonal(L.A)
# etc.
```

Additionally, the traits of `LinearAlgebra`, used by `LinearMaps.jl`, such as `isposdef`, `ishermitian`, etc., are also forwarded to the underlying operator:

```julia
LinearAlgebra.issymmetric(L::LinearMapWrapper) = is_symmetric(L.A) && domain_type(L) <: Real
LinearAlgebra.ishermitian(L::LinearMapWrapper) = is_symmetric(L.A)
LinearAlgebra.isposdef(L::LinearMapWrapper) = is_positive_definite(L.A)
```

This forwarding enables optimization-aware code relying on these predicates to continue working when using wrapped operators. For a full description of these traits, see the [Properties and Traits](@ref) page.
