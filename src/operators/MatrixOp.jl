export MatrixOp

immutable MatrixOp{M <: AbstractMatrix, T} <: LinearOperator
	A::M
	n_columns_input::Integer
end

# Constructors

MatrixOp{M <: AbstractMatrix}(A::M) = MatrixOp{M, eltype(A)}(A, 1)
MatrixOp{M <: AbstractMatrix}(A::M, T::Type) = MatrixOp{M, T}(A, 1)
MatrixOp{M <: AbstractMatrix}(A::M, n::Integer) = MatrixOp{M, eltype(A)}(A, n)
MatrixOp{M <: AbstractMatrix}(A::M, T::Type, n::Integer) = MatrixOp{M, T}(A, n)

MatrixOp{M <: AbstractMatrix}(x::AbstractArray, A::M) = ndims(x) > 2 ?
error("cannot multiply a Matrix by a n-dimensional Variable with n > 2") : MatrixOp{M,eltype(x)}(A,size(x,2))

# Mappings

A_mul_B!{M, T}(y::AbstractArray, L::MatrixOp{M, T}, b::AbstractArray) = A_mul_B!(y, L.A, b)
Ac_mul_B!{M, T}(y::AbstractArray, L::MatrixOp{M, T}, b::AbstractArray) = Ac_mul_B!(y, L.A, b)

# Properties

domainType{M, T}(L::MatrixOp{M, T}) = T
codomainType{M, T}(L::MatrixOp{M, T}) = T

function size(L::MatrixOp)
	if L.n_columns_input == 1
		( (size(L.A, 1),), (size(L.A, 2),) )
	else
		( (size(L.A, 1), L.n_columns_input), (size(L.A, 2), L.n_columns_input) )
	end
end

fun_name(L::MatrixOp) = "Matrix operator"

is_diagonal(L::MatrixOp) = isdiag(L.A)
is_full_row_rank(L::MatrixOp) = rank(L.A) == size(L.A, 1)
is_full_column_rank(L::MatrixOp) = rank(L.A) == size(L.A, 2)
# the following is O(n^3): I would assume for now no matrix is Gram diagonal
# is_gram_diagonal(L::MatrixOp)   = isdiag(L.A'*L.A)
