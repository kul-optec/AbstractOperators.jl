export MatrixOp

"""
	MatrixOp(domain_type=Float64::Type, dim_in::Tuple, A::AbstractMatrix)
	MatrixOp(A::AbstractMatrix)
	MatrixOp(A::AbstractMatrix, n_colons)

Creates a `LinearOperator` which, when multiplied with a vector `x::AbstractVector`, returns the product `A*x`.

The input `x` can be also a matrix: the number of columns must be given either in the second entry of `dim_in::Tuple` or using the constructor `MatrixOp(A::AbstractMatrix, n_colons)`.

```jldoctest
julia> MatrixOp(Float64,(10,),randn(20,10))
▒  ℝ^10 -> ℝ^20

julia> MatrixOp(randn(20,10))
▒  ℝ^10 -> ℝ^20

julia> MatrixOp(Float64,(10,20),randn(20,10))
▒  ℝ^(10, 20) -> ℝ^(20, 20)

julia> MatrixOp(randn(20,10),4)
▒  ℝ^(10, 4) -> ℝ^(20, 4)
	
```
"""
struct MatrixOp{D,T,M<:AbstractMatrix{T}} <: LinearOperator
	A::M
	n_col_in::Integer
end

# Constructors

##TODO decide what to do when domain_type is given, with conversion one loses pointer to data...
###standard constructor Operator{N}(domain_type::Type, DomainDim::NTuple{N,Int})
function MatrixOp(
	domain_type::Type, DomainDim::NTuple{N,Int}, A::M
) where {N,T,M<:AbstractMatrix{T}}
	N > 2 && error("cannot multiply a Matrix by a n-dimensional Variable with n > 2")
	size(A, 2) != DomainDim[1] && error("wrong input dimensions")
	if N == 1
		MatrixOp{domain_type,T,M}(A, 1)
	else
		MatrixOp{domain_type,T,M}(A, DomainDim[2])
	end
end
###

MatrixOp(A::M) where {M<:AbstractMatrix} = MatrixOp(eltype(A), (size(A, 2),), A)
MatrixOp(D::Type, A::M) where {M<:AbstractMatrix} = MatrixOp(D, (size(A, 2),), A)
function MatrixOp(A::M, n::Integer) where {M<:AbstractMatrix}
	return MatrixOp(eltype(A), (size(A, 2), n), A)
end
function MatrixOp(D::Type, A::M, n::Integer) where {M<:AbstractMatrix}
	return MatrixOp(D, (size(A, 2), n), A)
end

function Scale(coeff::Number, A::MatrixOp)
	if coeff == 1
		return A
	end
	cT = codomain_type(A)
	isCodomainReal = typeof(cT) <: Tuple ? all([t <: Real for t in cT]) : cT <: Real
	if isCodomainReal && typeof(coeff) <: Complex
		error(
			"Cannot Scale AbstractOperator with real codomain with complex scalar. Use `DiagOp` instead.",
		)
	end
	return MatrixOp(coeff * A.A, A.n_col_in)
end
function Scale(coeff::Number, A::AdjointOperator{MatrixOp})
	if coeff == 1
		return A
	end
	cT = codomain_type(A)
	isCodomainReal = typeof(cT) <: Tuple ? all([t <: Real for t in cT]) : cT <: Real
	if isCodomainReal && typeof(coeff) <: Complex
		error(
			"Cannot Scale AbstractOperator with real codomain with complex scalar. Use `DiagOp` instead.",
		)
	end
	return AdjointOperator(Scale(conj(coeff), A.A))
end

import Base: convert
convert(::Type{LinearOperator}, L::M) where {T,M<:AbstractMatrix{T}} = MatrixOp{T,T,M}(L, 1)
function convert(::Type{LinearOperator}, L::M, n::Integer) where {T,M<:AbstractMatrix{T}}
	return MatrixOp{T,T,M}(L, n)
end
function convert(::Type{LinearOperator}, dom::Type, dim_in::Tuple, L::AbstractMatrix)
	return MatrixOp(dom, dim_in, L)
end

# Mappings

mul!(y::AbstractArray, L::MatrixOp{D,T,M}, b::AbstractArray) where {D,T,M} = mul!(y, L.A, b)
function mul!(
	y::AbstractArray, L::AdjointOperator{MatrixOp{D,T,M}}, b::AbstractArray
) where {D,T,M}
	return mul!(y, L.A.A', b)
end

# Special Case, real b, complex matrix
function mul!(
	y::AbstractArray, L::AdjointOperator{MatrixOp{D,T,M}}, b::AbstractArray
) where {D<:Real,T<:Complex,M}
	yc = similar(y, T, size(y))
	mul!(yc, L.A.A', b)
	return y .= real.(yc)
end

# Properties

domain_type(::MatrixOp{D}) where {D} = D
codomain_type(::MatrixOp{D,T}) where {D,T} = D <: Real && T <: Complex ? T : D
is_thread_safe(::MatrixOp) = true

function size(L::MatrixOp)
	if L.n_col_in == 1
		((size(L.A, 1),), (size(L.A, 2),))
	else
		((size(L.A, 1), L.n_col_in), (size(L.A, 2), L.n_col_in))
	end
end

fun_name(L::MatrixOp) = "▒"

is_diagonal(L::MatrixOp) = isdiag(L.A)
is_symmetric(L::MatrixOp) = issymmetric(L.A)
is_AAc_diagonal(L::MatrixOp) = isdiag(L.A * L.A')
is_AcA_diagonal(L::MatrixOp) = isdiag(L.A' * L.A)
is_null(L::MatrixOp) = L.A == 0 * I
is_eye(L::MatrixOp) = L.A == I
is_invertible(L::MatrixOp) =
	size(L.A, 1) == size(L.A, 2) &&
	!isapprox(det(BigFloat.(L.A)), 0, atol = eps(eltype(L.A)) * 10)
is_orthogonal(L::MatrixOp) = size(L.A, 1) == size(L.A, 2) && all(<(eps(eltype(L.A)) * 10), L.A' * L.A - I)
is_full_row_rank(L::MatrixOp) = rank(L.A) == size(L.A, 1)
is_full_column_rank(L::MatrixOp) = rank(L.A) == size(L.A, 2)
is_positive_definite(L::MatrixOp) = isposdef(L.A)
is_positive_semidefinite(L::MatrixOp) = issymmetric(L.A) && all(eigvals(Symmetric(L.A)) .>= 0)

has_optimized_normalop(::MatrixOp) = true
get_normal_op(L::MatrixOp) = MatrixOp(domain_type(L), size(L, 2), L.A' * L.A)

LinearAlgebra.opnorm(L::MatrixOp) = opnorm(L.A)
