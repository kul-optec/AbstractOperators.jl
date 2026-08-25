module WaveletOperators

export WaveletOp, wavelet, WT

using AbstractOperators
using Wavelets
import LinearAlgebra: mul!, opnorm
import Base: size
import AbstractOperators:
    domain_type,
    codomain_type,
    domain_array_type,
    codomain_array_type,
    fun_name,
    is_thread_safe,
    supports_threading,
    is_threaded,
    has_fast_opnorm,
    _normalize_array_type,
    _array_wrapper_type,
    _copy_operator_impl
import OperatorCore:
    is_AcA_diagonal,
    is_AAc_diagonal,
    diag_AcA,
    diag_AAc,
    is_invertible,
    is_full_row_rank,
    is_full_column_rank

"""
	WaveletOp(wavelet::DiscreteWavelet, dim_in::Integer)
	WaveletOp(wavelet::DiscreteWavelet, dim_in::Tuple)

Creates a `LinearOperator` which, when multiplied with a vector `x::AbstractVector`, returns the wavelet
transform of `x` using the given `wavelet` and `levels`.

```jldoctest
julia> using WaveletOperators

julia> W = WaveletOp(wavelet(WT.db4), 4)
𝒲  ℝ^4 -> ℝ^4

julia> W * ones(4)
4-element Vector{Float64}:
  2.0
 -5.551115123125783e-17
 -8.326672684688674e-17
 -8.326672684688674e-17

```
"""
struct WaveletOp{T, N, W <: DiscreteWavelet, S <: AbstractArray{T}} <: LinearOperator
    wavelet::W
    dim_in::NTuple{N, Int}
    levels::Int
end

# Constructors

function WaveletOp(wavelet::DiscreteWavelet, dim_in, levels = nothing; array_type::Type{<:AbstractArray} = Array{Float64})
    if isnothing(levels)
        levels = get_max_transform_levels(dim_in)
    end
    return WaveletOp(Float64, wavelet, dim_in, levels; array_type)
end

function WaveletOp(A::AbstractArray, wavelet::DiscreteWavelet, levels::Int = get_max_transform_levels(size(A)))
    return WaveletOp(eltype(A), wavelet, size(A), levels; array_type = typeof(A isa SubArray ? parent(A) : A))
end

function WaveletOp(
        T::Type, wavelet::DiscreteWavelet, dim_in::Integer, levels::Int = get_max_transform_levels(dim_in);
        array_type::Type{<:AbstractArray} = Array{T}
    )
    if isodd(dim_in)
        throw(ArgumentError("The input dimension $dim_in is not suitable for wavelet transform: only even dimensions are allowed."))
    end
    if levels > get_max_transform_levels(dim_in)
        throw(ArgumentError("The number of levels $levels exceeds the maximum allowed for dimension $dim_in: $(get_max_transform_levels(dim_in))."))
    end
    S = _normalize_array_type(array_type, T)
    return WaveletOp{T, 1, typeof(wavelet), S}(wavelet, (dim_in,), levels)
end

function WaveletOp(
        T::Type, wavelet::DiscreteWavelet, dim_in::NTuple{N, Int}, levels::Int = get_max_transform_levels(dim_in);
        array_type::Type{<:AbstractArray} = Array{T}
    ) where {N}
    if any(isodd.(dim_in))
        throw(ArgumentError("The input dimension $dim_in is not suitable for wavelet transform: only even dimensions are allowed."))
    end
    if levels > get_max_transform_levels(dim_in)
        throw(ArgumentError("The number of levels $levels exceeds the maximum allowed for dimensions $dim_in: $(get_max_transform_levels(dim_in))."))
    end
    S = _normalize_array_type(array_type, T)
    return WaveletOp{T, N, typeof(wavelet), S}(wavelet, dim_in, levels)
end

# Mappings

function mul!(y::AbstractArray{T}, L::WaveletOp{T}, x::AbstractArray{T}) where {T}
    AbstractOperators.check(y, L, x)
    return dwt!(y, x, L.wavelet, L.levels)
end

function mul!(
        y::AbstractArray{T}, L::AdjointOperator{<:WaveletOp{T}}, x::AbstractArray{T}
    ) where {T}
    AbstractOperators.check(y, L, x)
    return idwt!(y, x, L.A.wavelet, L.A.levels)
end

# Properties

fun_name(::WaveletOp) = "𝒲"

size(L::WaveletOp) = (L.dim_in, L.dim_in)

domain_type(::WaveletOp{T}) where {T} = T
codomain_type(::WaveletOp{T}) where {T} = T
domain_array_type(::WaveletOp{T, N, W, S}) where {T, N, W, S} = S
codomain_array_type(::WaveletOp{T, N, W, S}) where {T, N, W, S} = S

is_AcA_diagonal(L::WaveletOp) = true
is_AAc_diagonal(L::WaveletOp) = true
is_invertible(L::WaveletOp) = true
is_full_row_rank(L::WaveletOp) = true
is_full_column_rank(L::WaveletOp) = true

diag_AcA(::WaveletOp{T}) where {T} = real(T(1))
diag_AAc(::WaveletOp{T}) where {T} = real(T(1))

AbstractOperators.is_thread_safe(::WaveletOp) = true

has_fast_opnorm(::WaveletOp) = true
has_fast_opnorm(::AdjointOperator{<:WaveletOp}) = true
opnorm(::WaveletOp{T}) where {T} = one(T)
opnorm(L::AdjointOperator{<:WaveletOp}) = one(eltype(domain_type(L.A)))

# Utils

get_max_transform_levels(dim_in::Integer) = maxtransformlevels(dim_in)
get_max_transform_levels(dim_in::Tuple) = minimum(maxtransformlevels.(dim_in))


# ─── Threading ────────────────────────────────────────────────────────────────
#
# No Julia-level threaded path: the work is done inside Wavelets.jl, which manages its own
# parallelism. `threaded` is therefore accepted by `copy_operator` (so a threaded batch
# operator can ask for a serial child) but changes nothing here, which is exactly what
# `supports_threading = false` states.
AbstractOperators.is_threaded(::WaveletOp) = false
AbstractOperators.supports_threading(::WaveletOp) = false

# No buffers to deep-copy (only `wavelet`/`dim_in`/`levels`, all immutable), so this method
# exists purely to honour `storage_type` requests by rebuilding the `S` type parameter;
# `threaded` is accepted for uniform forwarding but has no effect (see above).
function _copy_operator_impl(
        op::WaveletOp{T, N, W, S}; storage_type = nothing, threaded = nothing
    ) where {T, N, W, S}
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return WaveletOp{T, N, W, _normalize_array_type(new_at, T)}(op.wavelet, op.dim_in, op.levels)
end

end # module
