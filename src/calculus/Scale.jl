export Scale

"""
	Scale(α::Number,A::AbstractOperator)

Shorthand constructor:

	*(α::Number,A::AbstractOperator)

Scale an `AbstractOperator` by a factor of `α`.

```jldoctest
julia> A = FiniteDiff((10,2))
δx  ℝ^(10, 2) -> ℝ^(9, 2)

julia> S = Scale(10,A)
αδx  ℝ^(10, 2) -> ℝ^(9, 2)

julia> 10*A         #shorthand
αδx  ℝ^(10, 2) -> ℝ^(9, 2)
	
```
"""
struct Scale{T<:Number,L<:AbstractOperator} <: AbstractOperator
	coeff::T
	coeff_conj::T
	A::L
	function Scale(coeff, coeff_conj, L)
		cT = codomainType(L)
		isCodomainReal = typeof(cT) <: Tuple ? all([t <: Real for t in cT]) : cT <: Real
		if isCodomainReal && typeof(coeff) <: Complex
			error(
				"Cannot Scale AbstractOperator with real codomain with complex scalar. Use `DiagOp` instead.",
			)
		end
		return new{typeof(coeff),typeof(L)}(coeff, coeff_conj, L)
	end
end

# Constructors
function Scale(coeff, L)
	coeff_conj = conj(coeff)
	coeff, coeff_conj = promote(coeff, coeff_conj)
	return Scale(coeff, coeff_conj, L)
end

# Special Constructors
# scale of scale
function Scale(coeff::Number, L::Scale)
	return Scale(*(promote(coeff, L.coeff)...), L.A)
end

# Mappings

function mul!(y::C, L::Scale{T,A}, x::D) where {T,C<:AbstractArray,D,A<:AbstractOperator}
	mul!(y, L.A, x)
	return y .*= L.coeff
end

function mul!(y::C, L::Scale{T,A}, x::D) where {T,C<:Tuple,D,A<:AbstractOperator}
	mul!(y, L.A, x)
	for k in eachindex(y)
		y[k] .*= L.coeff
	end
end

function mul!(
	y::D, S::AdjointOperator{Scale{T,A}}, x::C
) where {T,C,D<:AbstractArray,A<:AbstractOperator}
	L = S.A
	mul!(y, L.A', x)
	return y .*= L.coeff_conj
end

function mul!(
	y::D, S::AdjointOperator{Scale{T,A}}, x::C
) where {T,C,D<:Tuple,A<:AbstractOperator}
	L = S.A
	mul!(y, L.A', x)
	for k in eachindex(y)
		y[k] .*= L.coeff_conj
	end
end

has_optimized_normalop(L::Scale) = is_linear(L.A) && has_optimized_normalop(L.A)
function get_normal_op(L::Scale)
	if is_linear(L.A)
		return Scale(L.coeff*coeff_conj, L.coeff*coeff_conj, get_normal_op(L.A))
	else
		return L' * L
	end
end

# Properties

size(L::Scale) = size(L.A)

domainType(L::Scale) = domainType(L.A)
codomainType(L::Scale) = codomainType(L.A)
is_thread_safe(L::Scale) = is_thread_safe(L.A)

is_linear(L::Scale) = is_linear(L.A)
is_sliced(L::Scale) = is_sliced(L.A)
get_slicing_expr(L::Scale) = get_slicing_expr(L.A)
get_slicing_mask(L::Scale) = get_slicing_mask(L.A)
remove_slicing(L::Scale) = L.coeff * remove_slicing(L.A)
is_null(L::Scale) = is_null(L.A)
is_eye(L::Scale) = is_diagonal(L.A)
is_diagonal(L::Scale) = is_diagonal(L.A)
is_invertible(L::Scale) = L.coeff == 0 ? false : is_invertible(L.A)
is_AcA_diagonal(L::Scale) = is_AcA_diagonal(L.A)
is_AAc_diagonal(L::Scale) = is_AAc_diagonal(L.A)
is_full_row_rank(L::Scale) = is_full_row_rank(L.A)
is_full_column_rank(L::Scale) = is_full_column_rank(L.A)

fun_name(L::Scale) = "α$(fun_name(L.A))"

diag(L::Scale) = L.coeff * diag(L.A)
diag_AcA(L::Scale) = (L.coeff)^2 * diag_AcA(L.A)
diag_AAc(L::Scale) = (L.coeff)^2 * diag_AAc(L.A)
remove_displacement(S::Scale) = Scale(S.coeff, S.coeff_conj, remove_displacement(S.A))

LinearAlgebra.opnorm(L::Scale) = L.coeff * LinearAlgebra.opnorm(L.A)

# utils

function permute(S::Scale, p::AbstractVector{Int})
	A = permute(S.A, p)
	return Scale(S.coeff, S.coeff_conj, A)
end
