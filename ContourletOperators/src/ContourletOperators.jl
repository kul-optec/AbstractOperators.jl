module ContourletOperators

export ContourletOp, NSCTOp, ContourletParams, parabolic_levels

using AbstractOperators
using Contourlets
using RecursiveArrayTools: ArrayPartition
import LinearAlgebra: mul!
import Base: size
import AbstractOperators: domain_type, codomain_type, fun_name, is_thread_safe
import OperatorCore: is_invertible

_threading_policy(threaded::Bool) = threaded ? Enabled() : Disabled()

# ── Shared coefficient <-> ArrayPartition flattening (in-place, allocation-free) ──

function _flatten!(y::ArrayPartition, coeffs)
    copyto!(y.x[1], coeffs.coarse)
    i = 1
    for scale in coeffs.subbands, sb in scale
        i += 1
        copyto!(y.x[i], sb)
    end
    return y
end

function _unflatten!(coeffs, x::ArrayPartition)
    copyto!(coeffs.coarse, x.x[1])
    i = 1
    for scale in coeffs.subbands, sb in scale
        i += 1
        copyto!(sb, x.x[i])
    end
    return coeffs
end

function _band_sizes(template)
    band_sizes = NTuple{2, Int}[size(template.coarse)]
    for scale in template.subbands, sb in scale
        push!(band_sizes, size(sb))
    end
    return band_sizes
end

# ── ContourletOp / NSCTOp: shared implementation, distinguished by a Kind trait ──
#
# ContourletOp and NSCTOp wrap the same Contourlets.jl calling convention (preallocated
# workspace + coefficient buffer, forward!/inverse! into them) and differ only in *which*
# Contourlets.jl functions they call. That difference is captured by a singleton `Kind`
# type parameter dispatching to the four `_make_ws`/`_similar_coeffs`/`_forward!`/`_inverse!`
# methods below, so the struct, constructor, dispatch barrier, `mul!` pair, and traits are
# implemented once and specialize per kind via `const` type aliases.

abstract type TransformKind end
struct ContourletKind <: TransformKind end
struct NSCTKind <: TransformKind end

_make_ws(::ContourletKind, params, dim_in, T, threading) = make_workspace(params, dim_in; T = T)
_make_ws(::NSCTKind, params, dim_in, T, threading) = make_nsct_workspace(params, dim_in; T = T, threading = threading)

_similar_coeffs(::ContourletKind, params, dim_in, Td) = similar_coefficients(params, dim_in; Td = Td)
_similar_coeffs(::NSCTKind, params, dim_in, Td) = similar_nsct_coefficients(params, dim_in; Td = Td)

_forward!(::ContourletKind, coeffs, x, params; workspace, threading) = ct_forward!(coeffs, x, params; workspace = workspace, threading = threading)
_forward!(::NSCTKind, coeffs, x, params; workspace, threading) = nsct_forward!(coeffs, x, params; workspace = workspace, threading = threading)

_inverse!(::ContourletKind, y, coeffs, params; workspace, threading) = ct_inverse!(y, coeffs, params; workspace = workspace, threading = threading)
_inverse!(::NSCTKind, y, coeffs, params; workspace, threading) = nsct_inverse!(y, coeffs, params; workspace = workspace, threading = threading)

_fun_name(::ContourletKind) = "𝒞𝒯"
_fun_name(::NSCTKind) = "𝒩𝒮𝒞𝒯"

struct ContourletTransformOp{K <: TransformKind, T, P <: ContourletParams, W, C, S <: Tuple, CT <: Tuple, TP <: ThreadingPolicy} <: LinearOperator
    kind::K
    params::P
    dim_in::NTuple{2, Int}
    band_sizes::Vector{NTuple{2, Int}}
    workspace::W
    coeffs::C
    codomain_size::S   # == Tuple(band_sizes), cached so size(L,1) is allocation-free
    codomain_types::CT  # == ntuple(_ -> T, length(band_sizes)), cached for codomain_type(L)
    threading::TP
end

function _transform_op(kind::K, T::Type, params::ContourletParams, dim_in::NTuple{2, Int}; threaded::Bool = true) where {K <: TransformKind}
    threading = _threading_policy(threaded)
    # make_workspace/make_nsct_workspace internally build their buffers at
    # Td = promote_type(eltype(params), T) (they cannot go below the filter precision).
    # _similar_coeffs must use that same Td, not the raw requested T, or coeffs/workspace
    # end up with mismatched element types and forward!/inverse! fail to dispatch at the
    # first mul! call.
    Td = promote_type(eltype(params), T)
    workspace = _make_ws(kind, params, dim_in, Td, threading)
    return _make_transform_op(kind, Td, params, dim_in, workspace, threading)
end

# Dispatch barrier: make_workspace's return type depends on a runtime filter-mode
# check inside Contourlets.jl (ladder vs modulation), so its inferred type at the
# call site above can be a Union of concrete types. Binding `workspace::W` as a
# method type parameter here re-concretizes W for whichever branch actually ran,
# so the ContourletTransformOp{...} inner constructor call resolves statically for JET.
# TODO(upstream): Contourlets.jl's make_workspace/make_nsct_workspace return type
# is itself unstable; if that gets fixed upstream this barrier becomes unnecessary.
@noinline function _make_transform_op(kind::K, T::Type, params::P, dim_in::NTuple{2, Int}, workspace::W, threading::TP) where {K <: TransformKind, P <: ContourletParams, W, TP <: ThreadingPolicy}
    coeffs = _similar_coeffs(kind, params, dim_in, T)
    band_sizes = _band_sizes(coeffs)
    codomain_size = Tuple(band_sizes)
    codomain_types = ntuple(_ -> T, length(band_sizes))
    return ContourletTransformOp{K, T, P, W, typeof(coeffs), typeof(codomain_size), typeof(codomain_types), TP}(
        kind, params, dim_in, band_sizes, workspace, coeffs, codomain_size, codomain_types, threading
    )
end

function mul!(y::ArrayPartition, L::ContourletTransformOp{K, T}, x::AbstractMatrix{T}) where {K, T}
    AbstractOperators.check(y, L, x)
    _forward!(L.kind, L.coeffs, x, L.params; workspace = L.workspace, threading = L.threading)
    _flatten!(y, L.coeffs)
    return y
end

function mul!(
        y::AbstractMatrix{T}, L::AdjointOperator{<:ContourletTransformOp{K, T}}, x::ArrayPartition
    ) where {K, T}
    AbstractOperators.check(y, L, x)
    _unflatten!(L.A.coeffs, x)
    _inverse!(L.A.kind, y, L.A.coeffs, L.A.params; workspace = L.A.workspace, threading = L.A.threading)
    return y
end

fun_name(L::ContourletTransformOp) = _fun_name(L.kind)

size(L::ContourletTransformOp) = (L.codomain_size, L.dim_in)

domain_type(::ContourletTransformOp{K, T}) where {K, T} = T
codomain_type(L::ContourletTransformOp) = L.codomain_types

is_invertible(::ContourletTransformOp) = true

AbstractOperators.is_thread_safe(::ContourletTransformOp) = false

# ── ContourletOp ────────────────────────────────────────────────────────────

"""
    ContourletOp(T::Type, params::ContourletParams, dim_in::NTuple{2,Int})
    ContourletOp(params::ContourletParams, dim_in::NTuple{2,Int})
    ContourletOp(A::AbstractMatrix, params::ContourletParams)

Creates a `LinearOperator` which, when multiplied with a matrix `x`, returns the discrete
Contourlet Transform (nearly critically sampled) coefficients of `x` as a flat
`ArrayPartition` (coarse band followed by each directional subband). `'` (`AdjointOperator`)
applies the inverse Contourlet Transform, giving a perfect-reconstruction left inverse.

The operator owns a preallocated `Contourlets.ContourletWorkspace` and coefficient buffer
(built via `make_workspace`/`similar_coefficients`), so both `mul!(y, C, x)` and
`mul!(y, C', x)` reuse the same scratch memory across calls instead of allocating fresh
buffers on every call. Because that scratch state is mutated in place, a single `ContourletOp`
instance is **not thread-safe** — use one instance per thread/task for concurrent use.

The `threaded::Bool` keyword (default `true`) controls whether the directional filter bank
stage inside Contourlets.jl runs multithreaded; pass `threaded = false` to force single-threaded
execution, e.g. to avoid oversubscription alongside multithreaded BLAS.

!!! note
    With the default `CDF97`/`Q2345` biorthogonal filters, the Contourlet Transform is
    *not* self-adjoint/orthogonal: `C'` is the declared **inverse** transform (`ct_inverse`),
    not the literal linear-algebra transpose of `C`. `C' * (C * x) ≈ x` holds to numerical
    precision, but `dot(C * x, y) == dot(x, C' * y)` does not in general.

```jldoctest
julia> using ContourletOperators, Contourlets

julia> params = ContourletParams(J = 2, L_array = [1, 2]);

julia> C = ContourletOp(params, (64, 64));

julia> y = C * ones(64, 64);

julia> x_rec = C' * y;

julia> maximum(abs, x_rec .- ones(64, 64)) < 1.0e-8
true
```
"""
const ContourletOp{T, P, W, C, S, CT, TP} = ContourletTransformOp{ContourletKind, T, P, W, C, S, CT, TP}

ContourletOp(T::Type, params::ContourletParams, dim_in::NTuple{2, Int}; kwargs...) = _transform_op(ContourletKind(), T, params, dim_in; kwargs...)
ContourletOp(params::ContourletParams, dim_in::NTuple{2, Int}; kwargs...) = ContourletOp(Float64, params, dim_in; kwargs...)
ContourletOp(A::AbstractMatrix, params::ContourletParams; kwargs...) = ContourletOp(eltype(A), params, size(A); kwargs...)

# ── NSCTOp ───────────────────────────────────────────────────────────────────

"""
    NSCTOp(T::Type, params::ContourletParams, dim_in::NTuple{2,Int})
    NSCTOp(params::ContourletParams, dim_in::NTuple{2,Int})
    NSCTOp(A::AbstractMatrix, params::ContourletParams)

Creates a `LinearOperator` which, when multiplied with a matrix `x`, returns the
Nonsubsampled Contourlet Transform (shift-invariant) coefficients of `x` as a flat
`ArrayPartition` (coarse band followed by each directional subband, all with the same
spatial size as `x`). `'` (`AdjointOperator`) applies the inverse NSCT, giving a
perfect-reconstruction left inverse.

The operator owns a preallocated `Contourlets.ContourletWorkspace` and coefficient buffer
(built via `make_nsct_workspace`/`similar_nsct_coefficients`), so both `mul!(y, N, x)` and
`mul!(y, N', x)` reuse the same scratch memory across calls instead of allocating fresh
buffers on every call. Because that scratch state is mutated in place, a single `NSCTOp`
instance is **not thread-safe** — use one instance per thread/task for concurrent use.

The `threaded::Bool` keyword (default `true`) controls threading for both the workspace's
FFTW plans (fixed at construction time) and the directional filter bank stage; pass
`threaded = false` to force single-threaded execution, e.g. to avoid oversubscription alongside
multithreaded BLAS.

!!! note
    With the default `CDF97`/`Q2345` biorthogonal filters, the NSCT is *not*
    self-adjoint/orthogonal: `N'` is the declared **inverse** transform (`nsct_inverse`),
    not the literal linear-algebra transpose of `N`. `N' * (N * x) ≈ x` holds to numerical
    precision, but `dot(N * x, y) == dot(x, N' * y)` does not in general.

```jldoctest
julia> using ContourletOperators, Contourlets

julia> params = ContourletParams(J = 2, L_array = [1, 2]);

julia> N = NSCTOp(params, (32, 32));

julia> y = N * ones(32, 32);

julia> x_rec = N' * y;

julia> maximum(abs, x_rec .- ones(32, 32)) < 1.0e-8
true
```
"""
const NSCTOp{T, P, W, C, S, CT, TP} = ContourletTransformOp{NSCTKind, T, P, W, C, S, CT, TP}

NSCTOp(T::Type, params::ContourletParams, dim_in::NTuple{2, Int}; kwargs...) = _transform_op(NSCTKind(), T, params, dim_in; kwargs...)
NSCTOp(params::ContourletParams, dim_in::NTuple{2, Int}; kwargs...) = NSCTOp(Float64, params, dim_in; kwargs...)
NSCTOp(A::AbstractMatrix, params::ContourletParams; kwargs...) = NSCTOp(eltype(A), params, size(A); kwargs...)

end # module
