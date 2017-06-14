__precompile__()

module AbstractOperators

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

abstract type LinearOperator end

import Base: A_mul_B!, Ac_mul_B!, size, ndims 
export ndoms

# deep stuff

include("utilities/deep.jl")

# Basic operators

include("operators/MyOperator.jl")
include("operators/Zeros.jl")
include("operators/Eye.jl")
include("operators/DiagOp.jl")
include("operators/GetIndex.jl")
include("operators/MatrixOp.jl")
include("operators/DFT.jl")
include("operators/DCT.jl")
include("operators/FiniteDiff.jl")
include("operators/Variation.jl")
include("operators/Conv.jl")
include("operators/Filt.jl")
include("operators/MIMOFilt.jl")
include("operators/ZeroPad.jl")
include("operators/Xcorr.jl")
include("operators/LBFGS.jl")
include("operators/BlkDiagLBFGS.jl")
include("operators/utils.jl")

# Calculus rules

include("calculus/Scale.jl")
include("calculus/DCAT.jl")
include("calculus/HCAT.jl")
include("calculus/VCAT.jl")
include("calculus/Compose.jl")
include("calculus/Reshape.jl")
include("calculus/Sum.jl")
include("calculus/Transpose.jl")

# Syntax
include("syntax.jl")

size(L::LinearOperator, i::Int) = size(L)[i]
ndims(L::LinearOperator) = count_dims(size(L,1)), count_dims(size(L,2))
ndims(L::LinearOperator, i::Int) = ndims(L)[i]

count_dims{N}(dims::NTuple{N,Int}) = N
count_dims{N}(dims::NTuple{N,Tuple}) = count_dims.(dims)

"""
`ndoms(L::LinearOperator, [dom::Int]) -> (number of codomains, number of domains)`

Returns the number of codomains and domains  of a `LinearOperator`. Optionally you can specify the codomain (with `dom = 1`) or the domain (with `dom = 2`)

```julia
julia > ndoms(DFT(10,10))
(1,1)

julia> ndoms(hcat(DFT(10,10),DFT(10,10)))
(1, 2)

julia> ndoms(hcat(DFT(10,10),DFT(10,10)),2)
2

julia> ndoms(blkdiag(DFT(10,10),DFT(10,10))
(2,2)
```
"""
ndoms(L::LinearOperator) = length.(ndims(L))
ndoms(L::LinearOperator, i::Int) = ndoms(L)[i]

is_null(L::LinearOperator) = false
is_eye(L::LinearOperator) = false
is_diagonal(L::LinearOperator) = false
is_gram_diagonal(L::LinearOperator) = is_diagonal(L)
is_invertible(L::LinearOperator) = false
is_full_row_rank(L::LinearOperator) = false
is_full_column_rank(L::LinearOperator) = false

#printing
function Base.show(io::IO, L::LinearOperator)
	print(io, fun_name(L)" "*fun_space(L) )
end

function fun_space(L::LinearOperator)  
	dom = fun_dom(L,2)
	codom = fun_dom(L,1)
	return dom*"->"*codom  
end

function fun_dom(L::LinearOperator,n::Int)
	dm = n == 2? domainType(L) : codomainType(L)
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
	length(s) > 3 ? s[1]*"..."s[end]  : *(s...)
end

export LinearOperator,
       domainType, 
       codomainType,
       is_eye,
       is_null,
       is_diagonal,
       is_gram_diagonal,
       is_invertible,
       is_full_row_rank,
       is_full_column_rank


end
