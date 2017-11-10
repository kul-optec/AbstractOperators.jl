export LMatrixOp


"""
`LMatrixOp(domainType=Float64::Type, dim_in::Tuple, b::Union{AbstractVector,AbstractMatrix})`

`LMatrixOp(b::AbstractVector, number_of_rows::Int)`

Creates a `LinearOperator` which, when multiplied with a matrix `X::AbstractMatrix`, returns the product `X*b`.

```julia
julia> op = LMatrixOp(Float64,(3,4),ones(4))
(⋅)b  ℝ^(3, 4) -> ℝ^3 

julia> op = LMatrixOp(ones(4),3)
(⋅)b  ℝ^(3, 4) -> ℝ^3

julia> op*ones(3,4)
3-element Array{Float64,1}:
 4.0
 4.0
 4.0

```

"""

struct LMatrixOp{T, A <: Union{AbstractVector,AbstractMatrix}, 
		 B <:Union{RowVector,AbstractMatrix}} <: LinearOperator
	b::A
	bt::B
	n_row_in::Integer
end

##TODO decide what to do when domainType is given, with conversion one loses pointer to data...
# Constructors
function LMatrixOp(DomainType::Type,
		   DomainDim::Tuple{Int,Int}, b::A) where {A <: Union{AbstractVector,AbstractMatrix}} 
	bt = b'
	LMatrixOp{DomainType, A, typeof(bt)}(b,bt,DomainDim[1])
end

LMatrixOp(b::A, n_row_in::Int) where {T,A<:Union{AbstractVector{T},AbstractMatrix{T}}} = 
LMatrixOp(T,(n_row_in,size(b,1)),b) 

# Mappings
A_mul_B!(y::C, L::LMatrixOp{T,A,B}, X::AbstractMatrix{T} ) where {T,A,B,C<:Union{AbstractVector,AbstractMatrix}} = 
A_mul_B!(y,X,L.b)

function Ac_mul_B!(y::AbstractMatrix{T}, L::LMatrixOp{T,A,B}, Y::AbstractVector{T} ) where {T,A,B} 
	y .= L.bt.*Y
end

function Ac_mul_B!(y::AbstractMatrix{T}, L::LMatrixOp{T,A,B}, Y::AbstractMatrix{T} ) where {T,A,B} 
	A_mul_Bc!(y,Y,L.b)
end

# Properties
domainType(L::LMatrixOp{T, A}) where {T, A} = T
codomainType(L::LMatrixOp{T, A}) where {T, A} = T

fun_name(L::LMatrixOp) = "(⋅)b"

size(L::LMatrixOp{T,A,B}) where {T,A <: AbstractVector,B <: RowVector} = (L.n_row_in,),(L.n_row_in, length(L.b))
size(L::LMatrixOp{T,A,B}) where {T,A <: AbstractMatrix,B <: AbstractMatrix} = (L.n_row_in,size(L.b,2)),(L.n_row_in, size(L.b,1))

#TODO

#is_full_row_rank(L::LMatrixOp) = 
#is_full_column_rank(L::MatrixOp) =
