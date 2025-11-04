export AdjointOperator

"""
	AdjointOperator(A::AbstractOperator)

Shorthand constructor:

	'(A::AbstractOperator)

Returns the adjoint operator of `A`.

```jldoctest
julia> AdjointOperator(ZeroPad((2,2),(0,2)))
[I;0]ᵃ  ℝ^(2, 4) -> ℝ^(2, 2)

julia> [Eye(10); FiniteDiff((10,))]'
[I;δx]ᵃ  ℝ^10  ℝ^9 -> ℝ^10
	
```
"""
struct AdjointOperator{T<:AbstractOperator} <: AbstractOperator
	A::T
	function AdjointOperator(A::T) where {T<:AbstractOperator}
		is_linear(A) == false &&
			error("Cannot transpose a nonlinear operator. You might use `jacobian`")
		return new{T}(A)
	end
end

# Constructors

AdjointOperator(L::AdjointOperator) = L.A

# Properties

has_fast_opnorm(L::AdjointOperator) = has_fast_opnorm(L.A)
LinearAlgebra.opnorm(L::AdjointOperator) = opnorm(L.A)
estimate_opnorm(L::AdjointOperator) = estimate_opnorm(L.A)

Base.:(==)(L1::AdjointOperator{T}, L2::AdjointOperator{T}) where {T} = L1.A == L2.A
size(L::AdjointOperator) = size(L.A, 2), size(L.A, 1)

domain_type(L::AdjointOperator) = codomain_type(L.A)
codomain_type(L::AdjointOperator) = domain_type(L.A)
domain_storage_type(L::AdjointOperator) = codomain_storage_type(L.A)
codomain_storage_type(L::AdjointOperator) = domain_storage_type(L.A)
is_thread_safe(L::AdjointOperator) = is_thread_safe(L.A)

fun_name(L::AdjointOperator) = fun_name(L.A) * "ᵃ"

is_linear(L::AdjointOperator) = is_linear(L.A)
is_null(L::AdjointOperator) = is_null(L.A)
is_eye(L::AdjointOperator) = is_eye(L.A)
is_diagonal(L::AdjointOperator) = is_diagonal(L.A)
is_AcA_diagonal(L::AdjointOperator) = is_AAc_diagonal(L.A)
is_AAc_diagonal(L::AdjointOperator) = is_AcA_diagonal(L.A)
is_orthogonal(L::AdjointOperator) = is_orthogonal(L.A)
is_invertible(L::AdjointOperator) = is_invertible(L.A)
is_full_row_rank(L::AdjointOperator) = is_full_column_rank(L.A)
is_full_column_rank(L::AdjointOperator) = is_full_row_rank(L.A)

diag(L::AdjointOperator) = diag(L.A)
diag_AcA(L::AdjointOperator) = diag_AAc(L.A)
diag_AAc(L::AdjointOperator) = diag_AcA(L.A)
