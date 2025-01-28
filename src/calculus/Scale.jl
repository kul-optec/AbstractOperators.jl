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
struct Scale{T<:RealOrComplex,L<:AbstractOperator} <: AbstractOperator
	coeff::T
	coeff_conj::T
	A::L
end

# Constructors

function Scale(coeff::T, L::R) where {T<:RealOrComplex,R<:AbstractOperator}
	coeff_conj = conj(coeff)
	coeff, coeff_conj = promote(coeff, coeff_conj)
	cT = codomainType(L)
	isCodomainReal = typeof(cT) <: Tuple ? all([t <: Real for t in cT]) : cT <: Real
	if isCodomainReal && T <: Complex
		error(
			"Cannot Scale AbstractOperator with real codomain with complex scalar. Use `DiagOp` instead.",
		)
	end
	return Scale{typeof(coeff),R}(coeff, coeff_conj, L)
end

# Special Constructors
# scale of scale
function Scale(
	coeff::T2, L::S
) where {T1<:RealOrComplex,T2<:RealOrComplex,R<:AbstractOperator,S<:Scale{T1,R}}
	return Scale(*(promote(coeff, L.coeff)...), L.A)
end
# scale of DiagOp
Scale(coeff::T, L::DiagOp) where {T<:RealOrComplex} = DiagOp(coeff * diag(L))

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

# Properties

size(L::Scale) = size(L.A)

domainType(L::Scale) = domainType(L.A)
codomainType(L::Scale) = codomainType(L.A)
is_thread_safe(L::Scale) = is_thread_safe(L.A)

is_linear(L::Scale) = is_linear(L.A)
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
