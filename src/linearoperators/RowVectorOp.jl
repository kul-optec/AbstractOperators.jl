export RowVectorOp

immutable RowVectorOp{T, M <: AbstractVector{T}} <: LinearOperator
	A::M
	n_col_in::Integer
end

# Constructors

function RowVectorOp{N, M <: AbstractVector}(DomainType::Type, DomainDim::NTuple{N,Int}, A::M)  
	N > 2 && error("cannot multiply a RowVector by a n-dimensional Variable with n > 2") 
	length(A) != DomainDim[1] && error("wrong input dimensions")
	if N == 1
		RowVectorOp{DomainType, M}(A, 1)
	else
		RowVectorOp{DomainType, M}(A, DomainDim[2])
	end
end
###

RowVectorOp{M <: AbstractVector}(A::M) = RowVectorOp{eltype(A), M}(A, 1)
RowVectorOp{M <: AbstractVector}(T::Type, A::M) = RowVectorOp{T, M}(A, 1)
RowVectorOp{M <: AbstractVector}(A::M, n::Integer) = RowVectorOp{eltype(A), M}(A, n)
RowVectorOp{M <: AbstractVector}(T::Type, A::M, n::Integer) = RowVectorOp{T, M}(A, n)

import Base: convert
convert{T,M<:AbstractVector{T}}(::Type{LinearOperator}, L::M) = Transpose(RowVectorOp{T,M}(L,1))
convert{T,M<:AbstractVector{T}}(::Type{LinearOperator}, L::M, n::Integer) = 
Transpose(RowVectorOp{T,M}(L, n))
convert(::Type{LinearOperator}, L::RowVector)             = RowVectorOp(L.vec, 1)
convert(::Type{LinearOperator}, L::RowVector, n::Integer) = RowVectorOp(L.vec, n)

# Mappings

A_mul_B!{M, T}(y::AbstractArray, L::RowVectorOp{M, T}, b::AbstractArray) = Ac_mul_B!(y, L.A, b)
Ac_mul_B!{M, T}(y::AbstractArray, L::RowVectorOp{M, T}, b::AbstractArray) = A_mul_B!(y, L.A, b)

# Properties

domainType{T, M}(L::RowVectorOp{T, M}) = T
codomainType{T, M}(L::RowVectorOp{T, M}) = T

size(L::RowVectorOp) = L.n_col_in == 1 ? ((1,),(length(L.A),)) : ((1,L.n_col_in),(length(L.A),L.n_col_in))

fun_name(L::RowVectorOp) = "━"

is_full_row_rank(L::RowVectorOp) = all(L.A .!= 0.)
is_full_column_rank(L::RowVectorOp) = true
