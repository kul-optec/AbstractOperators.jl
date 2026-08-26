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

# ── ContourletTransform: shared field layout and traits for ContourletOp/NSCTOp ──
#
# ContourletOp and NSCTOp wrap the same Contourlets.jl calling convention (preallocated
# workspace + coefficient buffer, forward!/inverse! into them) and share an identical field
# layout, so their traits (size/domain_type/codomain_type/is_invertible/is_thread_safe) are
# defined once here, mirroring FFTWOperators' DCT/IDCT `CosineTransform` supertype. Each
# concrete type keeps its own constructor, dispatch barrier, `mul!` pair, and `fun_name`
# because those call different Contourlets.jl functions (ct_* vs nsct_*).
abstract type ContourletTransform{T, P <: ContourletParams, W, C, S <: Tuple, CT <: Tuple, TP <: ThreadingPolicy} <: LinearOperator end

size(L::ContourletTransform) = (L.codomain_size, L.dim_in)

domain_type(::ContourletTransform{T}) where {T} = T
codomain_type(L::ContourletTransform) = L.codomain_types

is_invertible(::ContourletTransform) = true

AbstractOperators.is_thread_safe(::ContourletTransform) = false

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
struct ContourletOp{T, P <: ContourletParams, W, C, S <: Tuple, CT <: Tuple, TP <: ThreadingPolicy} <: ContourletTransform{T, P, W, C, S, CT, TP}
    params::P
    dim_in::NTuple{2, Int}
    band_sizes::Vector{NTuple{2, Int}}
    workspace::W
    coeffs::C
    codomain_size::S   # == Tuple(band_sizes), cached so size(L,1) is allocation-free
    codomain_types::CT  # == ntuple(_ -> T, length(band_sizes)), cached for codomain_type(L)
    threading::TP
end

function ContourletOp(T::Type, params::ContourletParams, dim_in::NTuple{2, Int}; threaded::Bool = true)
    # make_workspace internally builds its buffers at Td = promote_type(eltype(params), T)
    # (it cannot go below the filter precision). similar_coefficients must use that same Td,
    # not the raw requested T, or coeffs/workspace end up with mismatched element types and
    # ct_forward!/ct_inverse! fail to dispatch at the first mul! call.
    Td = promote_type(eltype(params), T)
    workspace = make_workspace(params, dim_in; T = Td)
    return _make_contourlet_op(Td, params, dim_in, workspace, _threading_policy(threaded))
end

# Dispatch barrier: make_workspace's return type depends on a runtime filter-mode
# check inside Contourlets.jl (ladder vs modulation), so its inferred type at the
# call site above can be a Union of concrete types. Binding `workspace::W` as a
# method type parameter here re-concretizes W for whichever branch actually ran,
# so the ContourletOp{...} inner constructor call resolves statically for JET.
# TODO(upstream): Contourlets.jl's make_workspace/make_nsct_workspace return type
# is itself unstable; if that gets fixed upstream this barrier becomes unnecessary.
@noinline function _make_contourlet_op(T::Type, params::P, dim_in::NTuple{2, Int}, workspace::W, threading::TP) where {P <: ContourletParams, W, TP <: ThreadingPolicy}
    coeffs = similar_coefficients(params, dim_in; Td = T)
    band_sizes = _band_sizes(coeffs)
    codomain_size = Tuple(band_sizes)
    codomain_types = ntuple(_ -> T, length(band_sizes))
    return ContourletOp{T, P, W, typeof(coeffs), typeof(codomain_size), typeof(codomain_types), TP}(
        params, dim_in, band_sizes, workspace, coeffs, codomain_size, codomain_types, threading
    )
end

ContourletOp(params::ContourletParams, dim_in::NTuple{2, Int}; kwargs...) = ContourletOp(Float64, params, dim_in; kwargs...)
ContourletOp(A::AbstractMatrix, params::ContourletParams; kwargs...) = ContourletOp(eltype(A), params, size(A); kwargs...)

function mul!(y::ArrayPartition, L::ContourletOp{T}, x::AbstractMatrix{T}) where {T}
    AbstractOperators.check(y, L, x)
    ct_forward!(L.coeffs, x, L.params; workspace = L.workspace, threading = L.threading)
    _flatten!(y, L.coeffs)
    return y
end

function mul!(
        y::AbstractMatrix{T}, L::AdjointOperator{<:ContourletOp{T}}, x::ArrayPartition
    ) where {T}
    AbstractOperators.check(y, L, x)
    _unflatten!(L.A.coeffs, x)
    ct_inverse!(y, L.A.coeffs, L.A.params; workspace = L.A.workspace, threading = L.A.threading)
    return y
end

fun_name(::ContourletOp) = "𝒞𝒯"

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
struct NSCTOp{T, P <: ContourletParams, W, C, S <: Tuple, CT <: Tuple, TP <: ThreadingPolicy} <: ContourletTransform{T, P, W, C, S, CT, TP}
    params::P
    dim_in::NTuple{2, Int}
    band_sizes::Vector{NTuple{2, Int}}
    workspace::W
    coeffs::C
    codomain_size::S   # == Tuple(band_sizes), cached so size(L,1) is allocation-free
    codomain_types::CT  # == ntuple(_ -> T, length(band_sizes)), cached for codomain_type(L)
    threading::TP
end

function NSCTOp(T::Type, params::ContourletParams, dim_in::NTuple{2, Int}; threaded::Bool = true)
    # threading is also passed to make_nsct_workspace: its FFTW plans are threaded at
    # construction time, and Contourlets.jl warns if a later call's threading kwarg
    # implies a different thread count than the workspace was built with.
    threading = _threading_policy(threaded)
    # make_nsct_workspace internally builds its buffers at Td = promote_type(eltype(params), T)
    # (it cannot go below the filter precision). similar_nsct_coefficients must use that same
    # Td, not the raw requested T, or coeffs/workspace end up with mismatched element types and
    # nsct_forward!/nsct_inverse! fail to dispatch at the first mul! call.
    Td = promote_type(eltype(params), T)
    workspace = make_nsct_workspace(params, dim_in; T = Td, threading = threading)
    return _make_nsct_op(Td, params, dim_in, workspace, threading)
end

# Dispatch barrier: see the comment on `_make_contourlet_op` above.
@noinline function _make_nsct_op(T::Type, params::P, dim_in::NTuple{2, Int}, workspace::W, threading::TP) where {P <: ContourletParams, W, TP <: ThreadingPolicy}
    coeffs = similar_nsct_coefficients(params, dim_in; Td = T)
    band_sizes = _band_sizes(coeffs)
    codomain_size = Tuple(band_sizes)
    codomain_types = ntuple(_ -> T, length(band_sizes))
    return NSCTOp{T, P, W, typeof(coeffs), typeof(codomain_size), typeof(codomain_types), TP}(
        params, dim_in, band_sizes, workspace, coeffs, codomain_size, codomain_types, threading
    )
end

NSCTOp(params::ContourletParams, dim_in::NTuple{2, Int}; kwargs...) = NSCTOp(Float64, params, dim_in; kwargs...)
NSCTOp(A::AbstractMatrix, params::ContourletParams; kwargs...) = NSCTOp(eltype(A), params, size(A); kwargs...)

function mul!(y::ArrayPartition, L::NSCTOp{T}, x::AbstractMatrix{T}) where {T}
    AbstractOperators.check(y, L, x)
    nsct_forward!(L.coeffs, x, L.params; workspace = L.workspace, threading = L.threading)
    _flatten!(y, L.coeffs)
    return y
end

function mul!(
        y::AbstractMatrix{T}, L::AdjointOperator{<:NSCTOp{T}}, x::ArrayPartition
    ) where {T}
    AbstractOperators.check(y, L, x)
    _unflatten!(L.A.coeffs, x)
    nsct_inverse!(y, L.A.coeffs, L.A.params; workspace = L.A.workspace, threading = L.A.threading)
    return y
end

fun_name(::NSCTOp) = "𝒩𝒮𝒞𝒯"

end # module
