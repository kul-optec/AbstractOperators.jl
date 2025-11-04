module WaveletOperators

export WaveletOp, wavelet, WT

using AbstractOperators
using Wavelets
import LinearAlgebra: mul!, opnorm
import Base: size
import AbstractOperators:
	domain_type,
	codomain_type,
	fun_name,
	is_thread_safe,
	has_fast_opnorm
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
julia> W = WaveletOp(wavelet(WT.db4), 4)
〜  ℝ^4 -> ℝ^4

julia> W * ones(4)
4-element Vector{Float64}:
  2.0
 -5.551115123125783e-17
 -8.326672684688674e-17
 -8.326672684688674e-17

```
"""
struct WaveletOp{T} <: LinearOperator
	wavelet::DiscreteWavelet
	dim_in::Tuple
	levels::Int
end

# Constructors

function WaveletOp(wavelet::DiscreteWavelet, dim_in, levels=nothing)
	if isnothing(levels)
		levels = get_max_transform_levels(dim_in)
	end
	return WaveletOp(Float64, wavelet, dim_in, levels)
end

function WaveletOp(A::AbstractArray, wavelet::DiscreteWavelet, levels::Int=get_max_transform_levels(size(A)))
	return WaveletOp(eltype(A), wavelet, size(A), levels)
end

function WaveletOp(T::Type, wavelet::DiscreteWavelet, dim_in::Integer, levels::Int=get_max_transform_levels(dim_in))
	if isodd(dim_in)
		throw(ArgumentError("The input dimension $dim_in is not suitable for wavelet transform: only even dimensions are allowed."))
	end
	if levels > get_max_transform_levels(dim_in)
		throw(ArgumentError("The number of levels $levels exceeds the maximum allowed for dimension $dim_in: $(get_max_transform_levels(dim_in))."))
	end
	return WaveletOp{T}(wavelet, (dim_in,), levels)
end

function WaveletOp(T::Type, wavelet::DiscreteWavelet, dim_in::Tuple, levels::Int=get_max_transform_levels(dim_in))
	if any(isodd.(dim_in))
		throw(ArgumentError("The input dimension $dim_in is not suitable for wavelet transform: only even dimensions are allowed."))
	end
	if levels > get_max_transform_levels(dim_in)
		throw(ArgumentError("The number of levels $levels exceeds the maximum allowed for dimensions $dim_in: $(get_max_transform_levels(dim_in))."))
	end
	return WaveletOp{T}(wavelet, dim_in, levels)
end

# Mappings

function mul!(y::AbstractArray{T}, L::WaveletOp{T}, x::AbstractArray{T}) where {T}
	return dwt!(y, x, L.wavelet, L.levels)
end

function mul!(
	y::AbstractArray{T}, L::AdjointOperator{WaveletOp{T}}, x::AbstractArray{T}
) where {T}
	return idwt!(y, x, L.A.wavelet, L.A.levels)
end

# Properties

fun_name(::WaveletOp) = "𝒲"

size(L::WaveletOp) = (L.dim_in, L.dim_in)

domain_type(::WaveletOp{T}) where {T} = T
codomain_type(::WaveletOp{T}) where {T} = T

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
opnorm(::AdjointOperator{<:WaveletOp}) = one(eltype(domain_type(L.A)))

# Utils

get_max_transform_levels(dim_in::Integer) = maxtransformlevels(dim_in)
get_max_transform_levels(dim_in::Tuple) = minimum(maxtransformlevels.(dim_in))

end # module
