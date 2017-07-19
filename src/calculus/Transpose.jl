export Transpose

immutable Transpose{T <: AbstractOperator} <: AbstractOperator
	A::T
	function Transpose(A::T) where {T<:AbstractOperator} 
		is_linear(A) == false && error("Cannot transpose a nonlinear operator. You might use `jacobian`")
		new{T}(A)
	end
end

# Constructors

Transpose(L::Transpose) = L.A

# Mappings

A_mul_B!{T<:AbstractOperator}(y, L::Transpose{T}, x) = Ac_mul_B!(y, L.A, x)
Ac_mul_B!{T<:AbstractOperator}(y, L::Transpose{T}, x) = A_mul_B!(y, L.A, x)

# Properties

size(L::Transpose) = size(L.A,2), size(L.A,1)

domainType(L::Transpose) = codomainType(L.A)
codomainType(L::Transpose) = domainType(L.A)

fun_name(L::Transpose)  = fun_name(L.A)*"ᵃ"

is_linear(L::Transpose) = is_linear(L.A)
is_null(L::Transpose) = is_null(L.A)
is_eye(L::Transpose) = is_eye(L.A)
is_diagonal(L::Transpose) = is_diagonal(L.A)
is_AcA_diagonal(L::Transpose) = is_AAc_diagonal(L.A)
is_AAc_diagonal(L::Transpose) = is_AcA_diagonal(L.A)
is_orthogonal(L::Transpose) = is_orthogonal(L.A)
is_invertible(L::Transpose) = is_invertible(L.A)
is_full_row_rank(L::Transpose) = is_full_column_rank(L.A)
is_full_column_rank(L::Transpose) = is_full_row_rank(L.A)

diag(L::Transpose) = diag(L.A)
diag_AcA(L::Transpose) = diag_AAc(L.A)
diag_AAc(L::Transpose) = diag_AcA(L.A)
