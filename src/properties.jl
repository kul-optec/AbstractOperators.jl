
import Base: size, ndims
import LinearAlgebra: diag

export ndoms,
       domainType,
       codomainType,
       domainStorageType,
       codomainStorageType,
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
       remove_displacement


"""
`domainType(A::AbstractOperator)`

Returns the type of the domain.

```julia
julia> domainType(DFT(10))
Float64

julia> domainType(hcat(Eye(Complex{Float64},(10,)),DFT(Complex{Float64},10)))
(Complex{Float64}, Complex{Float64})
```
"""
domainType

"""
`codomainType(A::AbstractOperator)`

Returns the type of the codomain.

```julia
julia> codomainType(DFT(10))
Complex{Float64}

julia> codomainType(vcat(Eye(Complex{Float64},(10,)),DFT(Complex{Float64},10)))
(Complex{Float64}, Complex{Float64})
```
"""
codomainType

"""
`inputStorageType(A::AbstractOperator)`

Returns the type of the storage for the input of the operator.

```julia
julia> inputStorageType(DFT(10))
Array{Complex{Float64},1}

julia> inputStorageType(hcat(Eye(Complex{Float64},(10,)),DFT(Complex{Float64},10)))
ArrayPartition{Complex{Float64},1,Array{Complex{Float64},1},Array{Complex{Float64},1}}
```
"""
function domainStorageType(L::AbstractOperator)
    dt = domainType(L)
    return if dt isa Tuple
        arrayTypes = Tuple{[Array{t, d} for (t, d) in zip(dt, length.(size(L,2)))]...}
        ArrayPartition{promote_type(dt...), arrayTypes}
    else
        Array{dt, length(size(L,2))}
    end
end

"""
`outputStorageType(A::AbstractOperator)`

Returns the type of the storage of for the output of the operator.

```julia
julia> outputStorageType(DFT(10))
Array{Complex{Float64},1}

julia> outputStorageType(vcat(Eye(Complex{Float64},(10,)),DFT(Complex{Float64},10)))
ArrayPartition{Complex{Float64},1,Array{Complex{Float64},1},Array{Complex{Float64},1}}
```
"""
function codomainStorageType(L::AbstractOperator)
    dt = codomainType(L)
    return if dt isa Tuple
        arrayTypes = Tuple{[Array{t, d} for (t, d) in zip(dt, length.(size(L,1)))]...}
        ArrayPartition{promote_type(dt...), arrayTypes}
    else
        Array{dt, length(size(L,1))}
    end
end

allocateInDomain(L::AbstractOperator, dims...=size(L,2)...) = allocate(domainStorageType(L), dims...)
allocateInCodomain(L::AbstractOperator, dims...=size(L,1)...) = allocate(codomainStorageType(L), dims...)

allocate(::Type{T}, dims...) where {T <: AbstractArray} = T(undef, dims...)
allocate(::Type{ArrayPartition{T,S}}, dims...) where {T,S} =
    ArrayPartition([allocate(s, d...) for (s,d) in zip(S.parameters, dims)]...)

storageTypeDisplayString(::Type{T}) where {T <: AbstractArray} = ""
storageDisplayString(L::AbstractOperator) = storageTypeDisplayString(codomainStorageType(L))

"""
`size(A::AbstractOperator, [dom,])`

Returns the size of an `AbstractOperator`. Type `size(A,1)` for the size of the codomain and `size(A,2)` for the size of the codomain.
"""
size(L::AbstractOperator, i::Int) = size(L)[i]

"""
`ndims(A::AbstractOperator, [dom,])`

Returns a `Tuple` with the number of dimensions of the codomain and domain of an `AbstractOperator`.  Type `ndims(A,1)` for the number of dimensions of the codomain and `ndims(A,2)` for the number of dimensions of the codomain.

"""
ndims(L::AbstractOperator) = count_dims(size(L,1)), count_dims(size(L,2))
ndims(L::AbstractOperator, i::Int) = ndims(L)[i]

count_dims(dims::NTuple{N,Int}) where N = N
count_dims(dims::NTuple{N,Tuple}) where N = count_dims.(dims)

"""
`ndoms(L::AbstractOperator, [dom::Int]) -> (number of codomains, number of domains)`

Returns the number of codomains and domains  of a `AbstractOperator`. Optionally you can specify the codomain (with `dom = 1`) or the domain (with `dom = 2`)

```julia
julia > ndoms(DFT(10,10))
(1,1)

julia> ndoms(hcat(DFT(10,10),DFT(10,10)))
(1, 2)

julia> ndoms(hcat(DFT(10,10),DFT(10,10)),2)
2

julia> ndoms(DCAT(DFT(10,10),DFT(10,10)))
(2,2)
```
"""
ndoms(L::AbstractOperator) = length.(ndims(L))
ndoms(L::AbstractOperator, i::Int) = ndoms(L)[i]

diag_AcA(L::AbstractOperator) = error("is_AAc_diagonal($L) == false")
diag_AAc(L::AbstractOperator) = error("is_AcA_diagonal($L) == false")

"""
`is_linear(A::AbstractOperator)`

Test whether `A` is a `LinearOperator`

```julia
julia> is_linear(Eye(2))
true

julia> is_linear(Sigmoid(Float64,(2,)))
false
```

"""
is_linear(          L::LinearOperator  ) = true
is_linear(          L::AbstractOperator) = false

"""
`is_null(A::AbstractOperator)`

Test whether `A` is null.

```julia
julia> is_null(Zeros(Float64,(10,),(10,)))
true

julia> is_null(Eye(10))
false

```
"""
is_null(            L::AbstractOperator) = false

"""
`is_eye(A::AbstractOperator)`

Test whether `A` is an Identity operator

```julia
julia> is_eye(Eye(10))
true

julia> is_eye(Zeros(Float64,(10,),(10,)))
false

```
"""
is_eye(             L::AbstractOperator) = false

"""
`is_diagonal(A::AbstractOperator)`

Test whether `A` is diagonal.

```julia
julia> is_diagonal(Eye(10))
true

julia> is_diagonal(FiniteDiff((10,)))
false

```
"""
is_diagonal(        L::AbstractOperator) = false

"""
`is_AcA_diagonal(A::AbstractOperator)`

Test whether `A'*A` is diagonal.

```julia
julia> is_AcA_diagonal(Eye(10))
true

julia> is_AcA_diagonal(GetIndex((10,),1:3))
false

```
"""
is_AcA_diagonal(    L::AbstractOperator) = is_diagonal(L)

"""
`is_AAc_diagonal(A::AbstractOperator)`

Test whether `A*A'` is diagonal.

```julia
julia> is_AAc_diagonal(Eye(10))
true

julia> is_AAc_diagonal(GetIndex((10,),1:3))
false

```
"""
is_AAc_diagonal(    L::AbstractOperator) = is_diagonal(L)

"""
`is_orthogonal(A::AbstractOperator)`

Test whether `A` is orthogonal.

```julia
julia> is_orthogonal(DCT(10))
true

julia> is_orthogonal(MatrixOp(randn(3,4)))
false

```
"""
is_orthogonal(      L::AbstractOperator) = false

"""
`is_invertible(A::AbstractOperator)`

Test whether `A` is easily invertible.

```julia
julia> is_invertible(DFT(10))
true

julia> is_invertible(MatrixOp(randn(3,4)))
false

```
"""
is_invertible(      L::AbstractOperator) = false

"""
`is_full_row_rank(A::AbstractOperator)`

Test whether `A` is easily invertible.

```julia
julia> is_full_row_rank(MatrixOp(randn(3,4)))
true

julia> is_full_row_rank(MatrixOp(randn(4,3)))
false
```
"""
is_full_row_rank(   L::AbstractOperator) = false

"""
`is_full_row_rank(A::AbstractOperator)`

Test whether `A` is easily invertible.

```julia
julia> is_full_column_rank(MatrixOp(randn(4,3)))
true

julia> is_full_column_rank(MatrixOp(randn(3,4)))
false
```
"""
is_full_column_rank(L::AbstractOperator) = false

"""
`is_sliced(A::AbstractOperator)`

Test whether `A` is a sliced operator.

```julia
julia> is_sliced(GetIndex((10,), 1:5))
true

```
"""
is_sliced(L::AbstractOperator) = false

"""
`displacement(A::AbstractOperator)`

Returns the displacement of the operator.

```julia
julia> A = AffineAdd(Eye(4),[1.;2.;3.;4.])
I+d  ℝ^4 -> ℝ^4

julia> displacement(A)
4-element Array{Float64,1}:
 1.0
 2.0
 3.0
 4.0

```
"""
function displacement(S::AbstractOperator)
    x = allocateInDomain(S)
    fill!(x, 0)
    d = S*x
    if all(y -> y == d[1], d)
        return d[1]
    else
        return d
    end
end

"""
`remove_displacement(A::AbstractOperator)`

Removes the displacement of the operator.

"""
remove_displacement(A::AbstractOperator) = A

import Base: convert
function convert(::Type{T}, dom::Type, dim_in::Tuple, L::T) where {T <: AbstractOperator}
	domainType(L) != dom && error("cannot convert operator with domain $(domainType(L)) to operator with domain $dom ")
	size(L,1) != dim_in && error("cannot convert operator with size $(size(L,1)) to operator with domain $dim_in ")
	return L
end


#printing
function Base.show(io::IO, L::AbstractOperator)
	print(io, fun_name(L)*storageDisplayString(L)*" "*fun_space(L))
end

function fun_space(L::AbstractOperator)
	dom = fun_dom(L,2)
	codom = fun_dom(L,1)
	return dom*"->"*codom
end

function fun_dom(L::AbstractOperator,n::Int)
	dm = n == 2 ? domainType(L) : codomainType(L)
	sz = size(L,n)
	return string_dom(dm,sz)
end

function string_dom(dm::Type,sz::Tuple)
	dm_st = dm <: Complex ? " ℂ" : " ℝ"
	sz_st = length(sz) == 1 ? "$(sz[1]) " : "$sz "
	return dm_st*"^"*sz_st
end

function string_dom(dm::Tuple,sz::Tuple)
	s = string_dom.(dm,sz)
	length(s) > 3 ? s[1]*"..."*s[end]  : *(s...)
end
