export MatrixOp

immutable MatrixOp{T, M <: AbstractMatrix{T}} <: LinearOperator
	A::M
	n_columns_input::Integer
end

# Constructors

##TODO decide what to do when domainType is given, with conversion one loses pointer to data...
###standard constructor Operator{N}(DomainType::Type, DomainDim::NTuple{N,Int})
function MatrixOp{N, M <: AbstractMatrix}(DomainType::Type, DomainDim::NTuple{N,Int}, A::M)  
	N > 2 && error("cannot multiply a Matrix by a n-dimensional Variable with n > 2") 
	size(A,2) != DomainDim[1] && error("wrong input dimensions")
	if N == 1
		MatrixOp{DomainType, M}(A, 1)
	else
		MatrixOp{DomainType, M}(A, DomainDim[2])
	end
end
###

MatrixOp{M <: AbstractMatrix}(A::M) = MatrixOp{eltype(A), M}(A, 1)
MatrixOp{M <: AbstractMatrix}(T::Type, A::M) = MatrixOp{T, M}(A, 1)
MatrixOp{M <: AbstractMatrix}(A::M, n::Integer) = MatrixOp{eltype(A), M}(A, n)
MatrixOp{M <: AbstractMatrix}(T::Type, A::M, n::Integer) = MatrixOp{T, M}(A, n)

# Mappings

A_mul_B!{M, T}(y::AbstractArray, L::MatrixOp{M, T}, b::AbstractArray) = A_mul_B!(y, L.A, b)
Ac_mul_B!{M, T}(y::AbstractArray, L::MatrixOp{M, T}, b::AbstractArray) = Ac_mul_B!(y, L.A, b)

# Properties

domainType{T, M}(L::MatrixOp{T, M}) = T
codomainType{T, M}(L::MatrixOp{T, M}) = T

function size(L::MatrixOp)
	if L.n_columns_input == 1
		( (size(L.A, 1),), (size(L.A, 2),) )
	else
		( (size(L.A, 1), L.n_columns_input), (size(L.A, 2), L.n_columns_input) )
	end
end

fun_name(L::MatrixOp) = "▒"

is_diagonal(L::MatrixOp) = isdiag(L.A)
is_full_row_rank(L::MatrixOp) = rank(L.A) == size(L.A, 1)
is_full_column_rank(L::MatrixOp) = rank(L.A) == size(L.A, 2)
