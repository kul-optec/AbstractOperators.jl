module WaveletOperators

export WaveletOp, wavelet, WT

using AbstractOperators
using Wavelets
import LinearAlgebra: mul!
import Base: size
import AbstractOperators:
	domainType,
	codomainType,
	fun_name,
	diag_AcA,
	diag_AAc
import OperatorCore:
	is_AcA_diagonal,
	is_AAc_diagonal,
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
end

# Constructors

function WaveletOp(T::Type, wavelet::DiscreteWavelet, dim_in::Integer)
	return WaveletOp{T}(wavelet, (dim_in,))
end

function WaveletOp(T::Type, wavelet::DiscreteWavelet, dim_in::Tuple)
	return WaveletOp{T}(wavelet, dim_in)
end

function WaveletOp(A::AbstractArray, wavelet::DiscreteWavelet)
	return WaveletOp{eltype(A)}(wavelet, size(A))
end

# Mappings

function mul!(y::AbstractArray{T}, L::WaveletOp{T}, x::AbstractArray{T}) where {T}
	return dwt!(y, x, L.wavelet)
end

function mul!(
	y::AbstractArray{T}, L::AdjointOperator{WaveletOp{T}}, x::AbstractArray{T}
) where {T}
	return idwt!(y, x, L.A.wavelet)
end

# Properties

fun_name(::WaveletOp) = "𝒲"

size(L::WaveletOp) = (L.dim_in, L.dim_in)

domainType(::WaveletOp{T}) where {T} = T
codomainType(::WaveletOp{T}) where {T} = T

is_AcA_diagonal(L::WaveletOp) = true
is_AAc_diagonal(L::WaveletOp) = true
is_invertible(L::WaveletOp) = true
is_full_row_rank(L::WaveletOp) = true
is_full_column_rank(L::WaveletOp) = true

diag_AcA(::WaveletOp{T}) where {T} = real(T(1))
diag_AAc(::WaveletOp{T}) where {T} = real(T(1))

end # module
