__precompile__()

module AbstractOperators

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

abstract type AbstractOperator end

abstract type LinearOperator    <: AbstractOperator end
abstract type NonLinearOperator <: AbstractOperator end

import Base: A_mul_B!, Ac_mul_B!, size, ndims, diag 
export ndoms, diag_AcA, diag_AAc

# deep stuff

include("utilities/deep.jl")

# Linear operators

include("linearoperators/MyLinOp.jl")
include("linearoperators/Zeros.jl")
include("linearoperators/Eye.jl")
include("linearoperators/DiagOp.jl")
include("linearoperators/GetIndex.jl")
include("linearoperators/MatrixOp.jl")
include("linearoperators/DFT.jl")
include("linearoperators/DCT.jl")
include("linearoperators/FiniteDiff.jl")
include("linearoperators/Variation.jl")
include("linearoperators/Conv.jl")
include("linearoperators/Filt.jl")
include("linearoperators/MIMOFilt.jl")
include("linearoperators/ZeroPad.jl")
include("linearoperators/Xcorr.jl")
include("linearoperators/LBFGS.jl")
include("linearoperators/BlkDiagLBFGS.jl")
include("linearoperators/utils.jl")

# Non-Linear operators
include("nonlinearoperators/Sigmoid.jl")

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

size(L::AbstractOperator, i::Int) = size(L)[i]
ndims(L::AbstractOperator) = count_dims(size(L,1)), count_dims(size(L,2))
ndims(L::AbstractOperator, i::Int) = ndims(L)[i]

count_dims{N}(dims::NTuple{N,Int}) = N
count_dims{N}(dims::NTuple{N,Tuple}) = count_dims.(dims)

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

julia> ndoms(blkdiag(DFT(10,10),DFT(10,10))
(2,2)
```
"""
ndoms(L::AbstractOperator) = length.(ndims(L))
ndoms(L::AbstractOperator, i::Int) = ndoms(L)[i]

diag_AcA(L::AbstractOperator) = error("is_AAc_diagonal($L) == false")
diag_AAc(L::AbstractOperator) = error("is_AcA_diagonal($L) == false")

is_linear(          L::LinearOperator  ) = true

is_linear(          L::AbstractOperator) = false
is_null(            L::AbstractOperator) = false
is_eye(             L::AbstractOperator) = false
is_diagonal(        L::AbstractOperator) = false
is_AcA_diagonal(    L::AbstractOperator) = is_diagonal(L)
is_AAc_diagonal(    L::AbstractOperator) = is_diagonal(L)
is_orthogonal(      L::AbstractOperator) = false
is_invertible(      L::AbstractOperator) = false
is_full_row_rank(   L::AbstractOperator) = false
is_full_column_rank(L::AbstractOperator) = false

export jacobian

jacobian(A::LinearOperator, x::AbstractArray) = A

#printing
function Base.show(io::IO, L::AbstractOperator)
	print(io, fun_name(L)" "*fun_space(L) )
end

function fun_space(L::AbstractOperator)  
	dom = fun_dom(L,2)
	codom = fun_dom(L,1)
	return dom*"->"*codom  
end

function fun_dom(L::AbstractOperator,n::Int)
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
       NonLinearOperator,
       AbstractOperator,
       domainType, 
       codomainType,
       is_linear,
       is_eye,
       is_null,
       is_diagonal,
       is_AcA_diagonal,
       is_AAc_diagonal,
       is_orthogonal,
       is_invertible,
       is_full_row_rank,
       is_full_column_rank


end
