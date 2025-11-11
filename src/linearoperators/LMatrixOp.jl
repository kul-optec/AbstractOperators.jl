export LMatrixOp

"""
	LMatrixOp(domain_type=Float64::Type, dim_in::Tuple, b::Union{AbstractVector,AbstractMatrix})
	LMatrixOp(b::AbstractVector, number_of_rows::Int)

Creates a `LinearOperator` which, when multiplied with a matrix `X::AbstractMatrix`, returns the product `X*b`.

```jldoctest
julia> op = LMatrixOp(Float64,(3,4),ones(4))
(⋅)b  ℝ^(3, 4) -> ℝ^3

julia> op = LMatrixOp(ones(4),3)
(⋅)b  ℝ^(3, 4) -> ℝ^3

julia> op*ones(3,4)
3-element Vector{Float64}:
 4.0
 4.0
 4.0
	
```
"""
struct LMatrixOp{T,A<:Union{AbstractVector,AbstractMatrix},B<:AbstractMatrix} <:
	   LinearOperator
	b::A
	bt::B
	n_row_in::Integer
end

##TODO decide what to do when domain_type is given, with conversion one loses pointer to data...
# Constructors
function LMatrixOp(
	domain_type::Type, DomainDim::Tuple{Int,Int}, b::A
) where {A<:Union{AbstractVector,AbstractMatrix}}
	bt = b'
	return LMatrixOp{domain_type,A,typeof(bt)}(b, bt, DomainDim[1])
end

function LMatrixOp(
	b::A, n_row_in::Int
) where {T,A<:Union{AbstractVector{T},AbstractMatrix{T}}}
	return LMatrixOp(T, (n_row_in, size(b, 1)), b)
end

# Mappings
function mul!(
	y::C, L::LMatrixOp{T,A,B}, X::AbstractMatrix{T}
) where {T,A,B,C<:Union{AbstractVector,AbstractMatrix}}
	return mul!(y, X, L.b)
end

function mul!(
	y::AbstractMatrix{T}, L::AdjointOperator{LMatrixOp{T,A,B}}, Y::AbstractVector{T}
) where {T,A,B}
	return y .= L.A.bt .* Y
end

function mul!(
	y::AbstractMatrix{T}, L::AdjointOperator{LMatrixOp{T,A,B}}, Y::AbstractMatrix{T}
) where {T,A,B}
	return mul!(y, Y, L.A.b')
end

# Properties
domain_type(::LMatrixOp{T}) where {T} = T
codomain_type(::LMatrixOp{T}) where {T} = T
is_thread_safe(::LMatrixOp) = true

fun_name(L::LMatrixOp) = "(⋅)b"

function size(L::LMatrixOp{T,A,B}) where {T,A<:AbstractVector,B<:Adjoint}
	return (L.n_row_in,), (L.n_row_in, length(L.b))
end
function size(L::LMatrixOp{T,A,B}) where {T,A<:AbstractMatrix,B<:AbstractMatrix}
	return (L.n_row_in, size(L.b, 2)), (L.n_row_in, size(L.b, 1))
end

#TODO

#is_full_row_rank(L::LMatrixOp) =
#is_full_column_rank(L::MatrixOp) =
