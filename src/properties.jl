
import Base: size, ndims
import LinearAlgebra: diag

export ndoms,
	domainType,
	codomainType,
	domain_storage_type,
	codomain_storage_type,
	is_linear,
	is_eye,
	is_null,
	is_diagonal,
	is_AcA_diagonal,
	is_AAc_diagonal,
	is_orthogonal,
	is_invertible,
	is_full_row_rank,
	is_full_column_rank,
	is_sliced,
	diag_AcA,
	diag_AAc,
	displacement,
	remove_displacement,
	is_thread_safe

"""
	domainType(A::AbstractOperator)

Returns the type of the domain.

```jldoctest
julia> domainType(DFT(10))
Float64

julia> domainType(hcat(Eye(Complex{Float64},(10,)),DFT(Complex{Float64},10)))
(ComplexF64, ComplexF64)
```
"""
domainType

"""
	codomainType(A::AbstractOperator)

Returns the type of the codomain.

```jldoctest
julia> codomainType(DFT(10))
ComplexF64 (alias for Complex{Float64})

julia> codomainType(vcat(Eye(Complex{Float64},(10,)),DFT(Complex{Float64},10)))
(ComplexF64, ComplexF64)
```
"""
codomainType

"""
	domain_storage_type(L::AbstractOperator)

Returns the type of the storage for the domain of the operator.

```jldoctest
julia> domain_storage_type(DFT(10))
Vector{Float64} (alias for Array{Float64, 1})

julia> domain_storage_type(hcat(Eye(Complex{Float64},(10,)),DFT(Complex{Float64},10)))
RecursiveArrayTools.ArrayPartition{ComplexF64, Tuple{Vector{ComplexF64}, Vector{ComplexF64}}}
```
"""
function domain_storage_type(L::AbstractOperator)
	dt = domainType(L)
	return if dt isa Tuple
		arrayTypes = Tuple{[Array{t,d} for (t, d) in zip(dt, length.(size(L, 2)))]...}
		ArrayPartition{promote_type(dt...),arrayTypes}
	else
		Array{dt,length(size(L, 2))}
	end
end

"""
	codomain_storage_type(L::AbstractOperator)

Returns the type of the storage of for the codomain of the operator.

```jldoctest
julia> codomain_storage_type(DFT(10))
Vector{ComplexF64} (alias for Array{Complex{Float64}, 1})

julia> codomain_storage_type(vcat(Eye(Complex{Float64},(10,)),DFT(Complex{Float64},10)))
RecursiveArrayTools.ArrayPartition{ComplexF64, Tuple{Vector{ComplexF64}, Vector{ComplexF64}}}
```
"""
function codomain_storage_type(L::AbstractOperator)
	dt = codomainType(L)
	return if dt isa Tuple
		arrayTypes = Tuple{[Array{t,d} for (t, d) in zip(dt, length.(size(L, 1)))]...}
		ArrayPartition{promote_type(dt...),arrayTypes}
	else
		Array{dt,length(size(L, 1))}
	end
end

function allocate_in_domain(L::AbstractOperator, dims...=size(L, 2)...)
	return allocate(domain_storage_type(L), dims...)
end
function allocate_in_codomain(L::AbstractOperator, dims...=size(L, 1)...)
	return allocate(codomain_storage_type(L), dims...)
end

allocate(::Type{T}, dims...) where {T<:AbstractArray} = T(undef, dims...)
function allocate(::Type{ArrayPartition{T,S}}, dims...) where {T,S}
	return ArrayPartition([allocate(s, d...) for (s, d) in zip(S.parameters, dims)]...)
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
"""
size(L::AbstractOperator, i::Int) = size(L)[i]

"""
	ndims(A::AbstractOperator, [dom,])

Returns a `Tuple` with the number of dimensions of the codomain and domain of an `AbstractOperator`.  Type `ndims(A,1)` for the number of dimensions of the codomain and `ndims(A,2)` for the number of dimensions of the codomain.

"""
ndims(L::AbstractOperator) = count_dims(size(L, 1)), count_dims(size(L, 2))
ndims(L::AbstractOperator, i::Int) = ndims(L)[i]

count_dims(::Tuple{}) = 0
count_dims(::NTuple{N,Int}) where {N} = N
count_dims(dims::NTuple) = count_dims.(dims)

"""
	ndoms(L::AbstractOperator, [dom::Int]) -> (number of codomains, number of domains)

Returns the number of codomains and domains  of a `AbstractOperator`. Optionally you can specify the codomain (with `dom = 1`) or the domain (with `dom = 2`)

```jldoctest
julia> ndoms(DFT(10,10))
(1, 1)

julia> ndoms(hcat(DFT(10,10),DFT(10,10)))
(1, 2)

julia> ndoms(hcat(DFT(10,10),DFT(10,10)),2)
2

julia> ndoms(DCAT(DFT(10,10),DFT(10,10)))
(2, 2)
```
"""
ndoms(L::AbstractOperator) = length.(ndims(L))
ndoms(L::AbstractOperator, i::Int) = ndoms(L)[i]

diag_AcA(L::AbstractOperator) = error("is_AAc_diagonal($L) == false")
diag_AAc(L::AbstractOperator) = error("is_AcA_diagonal($L) == false")

is_linear(L::LinearOperator) = true

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
	domainType(L) != dom && error(
		"cannot convert operator with domain $(domainType(L)) to operator with domain $dom ",
	)
	size(L, 1) != dim_in && error(
		"cannot convert operator with size $(size(L,1)) to operator with domain $dim_in ",
	)
	return L
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
	dm = n == 2 ? domainType(L) : codomainType(L)
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
