export AffineAdd

"""
	AffineAdd(A::AbstractOperator, d, [sign = true])

Affine addition to `AbstractOperator` with an array or scalar `d`.

Use `sign = false` to perform subtraction.

```jldoctest
julia> A = AffineAdd(Sin(3),[1.;2.;3.])
sin+d  ℝ^3 -> ℝ^3

julia> A*[3.;4.;5.] == sin.([3.;4.;5.]).+[1.;2.;3.]
true

julia> A = AffineAdd(Exp(3),[1.;2.;3.],false)
e-d  ℝ^3 -> ℝ^3

julia> A*[3.;4.;5.] == exp.([3.;4.;5.]).-[1.;2.;3.]
true
	
```
"""
struct AffineAdd{L<:AbstractOperator,D<:Union{AbstractArray,Number},S} <: AbstractOperator
	A::L
	d::D
	function AffineAdd(
		A::L, d::D, sign::Bool=true
	) where {L<:AbstractOperator,D<:AbstractArray}
		if size(d) != size(A, 1)
			throw(
				DimensionMismatch(
					"codomain size of $A not compatible with array `d` of size $(size(d))"
				),
			)
		end
		if eltype(d) != codomain_type(A)
			error(
				"cannot tilt opertor having codomain type $(codomain_type(A)) with array of type $(eltype(d))",
			)
		end
		return new{L,D,sign}(A, d)
	end
	# scalar
	function AffineAdd(A::L, d::D, sign::Bool=true) where {L<:AbstractOperator,D<:Number}
		if typeof(d) <: Complex && codomain_type(A) <: Real
			error(
				"cannot tilt opertor having codomain type $(codomain_type(A)) with array of type $(eltype(d))",
			)
		end
		return new{L,D,sign}(A, d)
	end
end

function Scale(coeff::Number, T::AffineAdd{L,D,S}) where {L,D,S}
	coeff == 1 ? T : AffineAdd(Scale(coeff, T.A), coeff * T.d, S)
end

# Mappings
# array
function mul!(y::DD, T::AffineAdd{L,D,true}, x) where {L<:AbstractOperator,DD,D}
	mul!(y, T.A, x)
	return @.. y += T.d
end

function mul!(y::DD, T::AffineAdd{L,D,false}, x) where {L<:AbstractOperator,DD,D}
	mul!(y, T.A, x)
	return @.. y -= T.d
end

function mul!(y, T::AdjointOperator{AffineAdd{L,D,S}}, x) where {L<:AbstractOperator,D,S}
	return mul!(y, T.A.A', x)
end

# Properties

function Base.:(==)(L1::AffineAdd{L,D,S}, L2::AffineAdd{L,D,S}) where {L,D,S}
	L1.A == L2.A && L1.d == L2.d
end
size(L::AffineAdd) = size(L.A)

domain_type(L::AffineAdd) = domain_type(L.A)
codomain_type(L::AffineAdd) = codomain_type(L.A)
domain_storage_type(L::AffineAdd) = domain_storage_type(L.A)
codomain_storage_type(L::AffineAdd) = codomain_storage_type(L.A)
is_thread_safe(L::AffineAdd) = is_thread_safe(L.A)

is_linear(L::AffineAdd) = is_linear(L.A)
is_null(L::AffineAdd) = is_null(L.A) && all(displacement(L) .== 0)
is_eye(L::AffineAdd) = is_eye(L.A) && all(displacement(L) .== 0)
is_diagonal(L::AffineAdd) = is_diagonal(L.A)
is_invertible(L::AffineAdd) = is_invertible(L.A)
is_AcA_diagonal(L::AffineAdd) = is_AcA_diagonal(L.A)
is_AAc_diagonal(L::AffineAdd) = is_AAc_diagonal(L.A)
is_full_row_rank(L::AffineAdd) = is_full_row_rank(L.A)
is_full_column_rank(L::AffineAdd) = is_full_column_rank(L.A)

is_sliced(L::AffineAdd) = is_sliced(L.A)
get_slicing_expr(L::AffineAdd) = get_slicing_expr(L.A)
get_slicing_mask(L::AffineAdd) = get_slicing_mask(L.A)
remove_slicing(L::AffineAdd) = remove_slicing(L.A)

fun_name(T::AffineAdd{L,D,S}) where {L,D,S} = "$(fun_name(T.A))" * (S ? "+" : "-") * "d"

diag(L::AffineAdd) = diag(L.A)
diag_AcA(L::AffineAdd) = diag_AcA(L.A)
diag_AAc(L::AffineAdd) = diag_AAc(L.A)

has_optimized_normalop(T::AffineAdd) = has_optimized_normalop(T.A)
function get_normal_op(T::AffineAdd{L,D,S}) where {L,D,S}
	AffineAdd(get_normal_op(T.A), T.A' * T.d, S)
end

# utils
import Base: sign
sign(T::AffineAdd{L,D,false}) where {L,D} = -1
sign(T::AffineAdd{L,D,true}) where {L,D} = 1

function permute(T::AffineAdd{L,D,S}, p::AbstractVector{Int}) where {L,D,S}
	A = permute(T.A, p)
	return AffineAdd(A, T.d, S)
end

displacement(A::AffineAdd{L,D,true}) where {L,D} = A.d .+ displacement(A.A)
displacement(A::AffineAdd{L,D,false}) where {L,D} = -A.d .+ displacement(A.A)

remove_displacement(A::AffineAdd) = remove_displacement(A.A)
