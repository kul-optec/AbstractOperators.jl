export MatrixMul

immutable MatrixMul{T, A <: AbstractVector, B <:RowVector} <: LinearOperator
	b::A
	bt::B
	n_row_in::Integer
end

##TODO decide what to do when domainType is given, with conversion one loses pointer to data...
# Constructors
function MatrixMul{N, A <: AbstractVector}(DomainType::Type,
					   DomainDim::NTuple{N,Int}, b::A, n_row_in::Integer)  
	N != 2 && error("length(DomainDim) must be equal to 2 ")
	(n_row_in,length(b)) != (DomainDim[1],DomainDim[2]) && error("wrong input dimensions")
	bt = b'
	MatrixMul{DomainType, A, typeof(bt)}(b,bt,n_row_in)
end

MatrixMul{T,A<:AbstractVector{T}}(b::A, n_row_in::Int) = MatrixMul(T,(n_row_in,length(b)),b,n_row_in) 

# Mappings
A_mul_B!{T,A}(y::AbstractVector{T}, L::MatrixMul{T,A}, b::AbstractMatrix{T} ) = A_mul_B!(y,b,L.b)
function Ac_mul_B!{T,A,B}(y::AbstractMatrix{T}, L::MatrixMul{T,A,B}, b::AbstractVector{T} ) 
	y .= L.bt.*b
end

# Properties
domainType{T, A}(L::MatrixMul{T, A}) = T
codomainType{T, A}(L::MatrixMul{T, A}) = T

fun_name(L::MatrixMul) = "(⋅)b"

size(L::MatrixMul) = (L.n_row_in,),(L.n_row_in, length(L.b))

#TODO

#is_full_row_rank(L::MatrixMul) = 
#is_full_column_rank(L::MatrixOp) =
