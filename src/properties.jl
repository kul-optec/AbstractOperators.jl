
import Base: size, ndims, similar, copy
import LinearAlgebra: diag, opnorm

export ndoms,
	domain_type,
	codomain_type,
	domain_storage_type,
	codomain_storage_type,
	is_linear,
	is_eye,
	is_null,
	is_diagonal,
	is_AcA_diagonal,
	is_AAc_diagonal,
	diag_AcA,
	diag_AAc,
	is_orthogonal,
	is_invertible,
	is_full_row_rank,
	is_full_column_rank,
	is_positive_definite,
	is_positive_semidefinite,
	is_symmetric,
	is_sliced,
	remove_slicing,
	displacement,
	remove_displacement,
	is_thread_safe,
	estimate_opnorm

"""
	domain_type(A::AbstractOperator)

Returns the type of the domain.

```jldoctest
julia> domain_type(DiagOp(rand(10)))
Float64

julia> domain_type(hcat(Eye(Complex{Float64},(10,)),DiagOp(rand(ComplexF64, 10))))
(ComplexF64, ComplexF64)
```
"""
domain_type

"""
	codomain_type(A::AbstractOperator)

Returns the type of the codomain.

```jldoctest
julia> codomain_type(DiagOp(rand(ComplexF64, 10)))
ComplexF64 (alias for Complex{Float64})

julia> codomain_type(vcat(Eye(Complex{Float64},(10,)),DiagOp(rand(ComplexF64, 10))))
(ComplexF64, ComplexF64)
```
"""
codomain_type

"""
	domain_storage_type(L::AbstractOperator)

Returns the type of the storage for the domain of the operator.

```jldoctest
julia> domain_storage_type(DiagOp(rand(10)))
Array{Float64}

julia> domain_storage_type(hcat(Eye(Complex{Float64},(10,)),DiagOp(rand(ComplexF64, 10))))
RecursiveArrayTools.ArrayPartition{ComplexF64, Tuple{Array{ComplexF64}, Array{ComplexF64}}}
```
"""
function domain_storage_type(L::AbstractOperator)
	dt = domain_type(L)
	return if dt isa Tuple
		arrayTypes = Tuple{[Array{t} for t in dt]...}
		ArrayPartition{promote_type(dt...),arrayTypes}
	else
		Array{dt}
	end
end

"""
	codomain_storage_type(L::AbstractOperator)

Returns the type of the storage of for the codomain of the operator.

```jldoctest
julia> codomain_storage_type(DiagOp(rand(ComplexF64,10)))
Array{ComplexF64}

julia> codomain_storage_type(vcat(Eye(Complex{Float64},(10,)),DiagOp(rand(ComplexF64,10))))
RecursiveArrayTools.ArrayPartition{ComplexF64, Tuple{Array{ComplexF64}, Array{ComplexF64}}}
```
"""
function codomain_storage_type(L::AbstractOperator)
	dt = codomain_type(L)
	return if dt isa Tuple
		arrayTypes = Tuple{[Array{t} for t in dt]...}
		ArrayPartition{promote_type(dt...),arrayTypes}
	else
		Array{dt}
	end
end

function allocate_in_domain(L::AbstractOperator, dims...=size(L, 2)...)
	dS = domain_storage_type(L)
	if dS <: ArrayPartition
		S = dS.parameters[2]
		return ArrayPartition([similar(s, d...) for (s, d) in zip(S.parameters, dims)]...)
	else
		return similar(dS, dims...)
	end
end

function allocate_in_codomain(L::AbstractOperator, dims...=size(L, 1)...)
	cS = codomain_storage_type(L)
	if cS <: ArrayPartition
		S = cS.parameters[2]
		return ArrayPartition([similar(s, d...) for (s, d) in zip(S.parameters, dims)]...)
	else
		return similar(cS, dims...)
	end
end

storage_type_display_string(::Type{T}) where {T<:AbstractArray} = ""
function storage_display_string(L::AbstractOperator)
	return storage_type_display_string(codomain_storage_type(L))
end

"""
	is_thread_safe(L::AbstractOperator)

Returns whether the operator is thread safe (i.e. it can be used on multiple arrays simulaneously).
"""
is_thread_safe(L::AbstractOperator) = false

"""
	size(A::AbstractOperator, [dom,])

Returns the size of an `AbstractOperator`. Type `size(A,1)` for the size of the codomain and `size(A,2)` for the size of the codomain.

Note that the size is always returned as a `Tuple`, so for a 2D operator the size of the codomain will be `(m,)`
and the size of the domain will be `(n,)` for an `m x n` operator.

```jldoctest
julia> size(FiniteDiff((10,20), 1))
((9, 20), (10, 20))

julia> size(FiniteDiff((10,20), 1),1)
(9, 20)

julia> size(FiniteDiff((10,20), 1),2)
(10, 20)
```
"""
size(L::AbstractOperator, i::Int) = size(L)[i]

"""
	ndims(A::AbstractOperator, [dom,])

Returns a `Tuple` with the number of dimensions of the codomain and domain of an `AbstractOperator`.  Type `ndims(A,1)` for the number of dimensions of the codomain and `ndims(A,2)` for the number of dimensions of the codomain.

```jldoctest
julia> V = Variation((2,3,4))
Ʋ  ℝ^(2, 3, 4) -> ℝ^(24, 3)

julia> ndims(V)
(2, 3)

julia> ndims(V,1)
2

julia> ndims(V,2)
3
```
"""
ndims(L::AbstractOperator) = count_dims(size(L, 1)), count_dims(size(L, 2))
ndims(L::AbstractOperator, i::Int) = ndims(L)[i]

count_dims(::Tuple{}) = 0
count_dims(::NTuple{N,Int}) where {N} = N
count_dims(dims::Tuple) = count_dims.(dims)

"""
	ndoms(L::AbstractOperator, [dom::Int]) -> (number of codomains, number of domains)

Returns the number of codomains and domains  of a `AbstractOperator`. Optionally you can specify the codomain (with `dom = 1`) or the domain (with `dom = 2`)

```jldoctest
julia> ndoms(Eye(10,10))
(1, 1)

julia> ndoms(hcat(Eye(10,10),Eye(10,10)))
(1, 2)

julia> ndoms(hcat(Eye(10,10),Eye(10,10)),2)
2

julia> ndoms(DCAT(Eye(10,10),Eye(10,10)))
(2, 2)
```
"""
ndoms(L::AbstractOperator) = length.(ndims(L))
ndoms(L::AbstractOperator, i::Int) = ndoms(L)[i]

is_linear(L::LinearOperator) = true

"""
	is_sliced(A)

Returns true if `A` is a sliced operator.
Operator `A` is sliced if it applies to only a subset of the input values.

```jldoctest
julia> is_sliced(DiagOp(rand(10)))
false

julia> is_sliced(DiagOp(rand(10)) * GetIndex((20,), 1:10))
true
```
"""
is_sliced(L) = false

"""
	get_slicing_expr(A)

Returns the slicing expression of `A`.
Operator `A` is sliced if it applies to only a subset of the input values.
The slicing expression is either a tuple of indices or a bit array that specifies the subset of input values that `A` applies to.
"""
get_slicing_expr(L) = is_null(L) ? nothing : Colon()

"""
	get_slicing_mask(A)

Returns the slicing mask of `A`.
Operator `A` is sliced if it applies to only a subset of the input values.
"""
get_slicing_mask(L) = error("cannot get slicing mask of operator of type $(typeof(L))")

"""
	remove_slicing(A)

Returns the operator `A` without slicing.
Operator `A` is sliced if it applies to only a subset of the input values.
"""
remove_slicing(L) = L

has_fast_opnorm(L) = false

"""
	displacement(A::AbstractOperator)

Returns the displacement of the operator.

```jldoctest
julia> A = AffineAdd(Eye(4),[1.;2.;3.;4.])
I+d  ℝ^4 -> ℝ^4

julia> displacement(A)
4-element Vector{Float64}:
 1.0
 2.0
 3.0
 4.0

```
"""
function displacement(S::AbstractOperator)
	x = allocate_in_domain(S)
	fill!(x, 0)
	d = S * x
	if all(y -> y == d[1], d)
		return d[1]
	else
		return d
	end
end

"""
	remove_displacement(A::AbstractOperator)

Removes the displacement of the operator.

"""
remove_displacement(A::AbstractOperator) = A

import Base: convert
function convert(::Type{T}, dom::Type, dim_in::Tuple, L::T) where {T<:AbstractOperator}
	domain_type(L) != dom && error(
		"cannot convert operator with domain $(domain_type(L)) to operator with domain $dom ",
	)
	size(L, 1) != dim_in && error(
		"cannot convert operator with size $(size(L,1)) to operator with domain $dim_in ",
	)
	return L
end

"""
	can_be_combined(L::AbstractOperator, R::AbstractOperator) = false
	can_be_combined(L::AbstractOperator, M::AbstractOperator, R::AbstractOperator) = false

Returns whether the operators `L` and `R` can be merged when they are multiplied.

Examples:
```jldoctest
julia> AbstractOperators.can_be_combined(DiagOp(rand(10)), FiniteDiff((10,)))
false

julia> AbstractOperators.can_be_combined(Eye(10), FiniteDiff((11,)))
true
```
"""
can_be_combined(L, R) = is_eye(L) || is_eye(R) || is_null(L) || (is_null(R) && is_linear(L) && all(displacement(L) .== 0))
can_be_combined(L, M, R) = false

"""
	combine(L::AbstractOperator, R::AbstractOperator)
Returns the combined operator of `L` and `R`. The combined operator is defined as `L * R` where `L` and `R` are the operators to be combined.

Examples:
```jldoctest
julia> AbstractOperators.combine(Eye(10), DiagOp(rand(10)))
╲  ℝ^10 -> ℝ^10
```
"""
function combine(L, R)
	if is_eye(L)
		return R
	elseif is_eye(R)
		return L
	elseif is_null(L)
		if size(R, 1) == size(R, 2) && domain_type(R) == codomain_type(R)
			return L
		else
			return Zeros(domain_type(R), size(R, 2), codomain_type(L), size(L, 1))
		end
	elseif is_null(R) && is_linear(L) && all(displacement(L) .== 0)
		if size(L, 1) == size(L, 2) && domain_type(L) == codomain_type(L)
			return R
		else
			return Zeros(domain_type(R), size(R, 2), codomain_type(L), size(L, 1))
		end
	else
		error("cannot combine operators")
	end
end

function combine(L, M, R)
	error("cannot combine operators")
end

"""
	has_optimized_normalop(L::AbstractOperator)

Returns whether the operator `L` has an optimized normal operator.
The normal operator is defined as `L' * L` where `L'` is the adjoint of `L`.
"""
has_optimized_normalop(L::AbstractOperator) = false

"""
	get_normal_op(L::AbstractOperator)

Returns the normal operator of the operator `L`. The normal operator is defined as `L' * L` where `L'` is the adjoint of `L`.
"""
function get_normal_op(L::AbstractOperator)
	return L' * L
end

"""
	LinearAlgebra.diag(A::AbstractOperator)

Returns the diagonal of `A`. If `A` is not diagonal, an error is thrown.

The diagonal is defined as the vector `d` such that `A * x = d .* x` for all `x` in the domain of `A`, where `.*` is the element-wise multiplication.
"""
LinearAlgebra.diag(L::AbstractOperator) = error("cannot get diagonal of operator of type $(typeof(L))")

"""
	LinearAlgebra.opnorm(A::AbstractOperator)

Returns the operator norm of `A`. The operator norm is defined as the maximum singular value of `A`.
It is computed using the power method by default, unless the operator has a fast implementation.

The operator norm is defined as: `‖A‖ = sup_{x != 0} ‖A*x‖ / ‖x‖`.

Parameters of power iteration:
- Maximum number of iterations: 100
- Tolerance for convergence: 1e-6
These parameters can be adjusted in the [estimate_opnorm](@ref) function.
"""
function LinearAlgebra.opnorm(A::AbstractOperator)
	return powerit(A)
end

has_fast_opnorm(::AbstractOperator) = false

"""
	estimate_opnorm(A::AbstractOperator)

Estimates the operator norm of `A`. The operator norm is defined as the maximum singular value of `A`.
It is computed using the power method with reduced iterations unless the operator has a fast implementation.

The operator norm is defined as: `‖A‖ = sup_{x != 0} ‖A*x‖ / ‖x‖`.

Parameters of power iteration:
- Maximum number of iterations: 20
- Tolerance for convergence: 0.01
These parameters can be adjusted by passing `maxit` and `tol` keyword arguments. E.g.:
```julia
julia> estimate_opnorm(A; maxit=50, tol=1e-6)
```
"""
function estimate_opnorm(A::AbstractOperator; maxit=20, tol=1e-3)
	if has_fast_opnorm(A)
		return opnorm(A)
	else
		return powerit(A; maxit, tol)
	end
end

function powerit(A::AbstractOperator; maxit=100, tol=1e-6)
	# Power method for estimating the operator norm
	AHA = A' * A
    x = allocate_in_domain(A)
	y = similar(x)
	Random.randn!(x)
    normalize!(x)
    λ = zero(real(eltype(x)))
    λ_old = real(eltype(x))(Inf)

    for _ in 1:maxit
        mul!(y, AHA, x)
        λ = norm(y)
        if abs(λ - λ_old) < max(tol * λ, tol)
            break
        end
        λ_old = λ
        @.. thread=true x = y / λ
    end

    return sqrt(λ)
end

#printing
function Base.show(io::IO, L::AbstractOperator)
	return print(io, fun_name(L) * storage_display_string(L) * " " * fun_space(L))
end

function fun_space(L::AbstractOperator)
	dom = fun_dom(L, 2)
	codom = fun_dom(L, 1)
	return dom * "->" * codom
end

function fun_dom(L::AbstractOperator, n::Int)
	dm = n == 2 ? domain_type(L) : codomain_type(L)
	sz = size(L, n)
	return string_dom(dm, sz)
end

function string_dom(dm::Type, sz::Tuple)
	dm_st = dm <: Complex ? " ℂ" : " ℝ"
	sz_st = length(sz) == 1 ? "$(sz[1]) " : "$sz "
	return dm_st * "^" * sz_st
end

function string_dom(dm::Tuple, sz::Tuple)
	s = string_dom.(dm, sz)
	return length(s) > 3 ? s[1] * "..." * s[end] : *(s...)
end
