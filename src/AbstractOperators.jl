__precompile__()

module AbstractOperators

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

export LinearOperator

abstract type LinearOperator end

import Base: A_mul_B!, Ac_mul_B!, size 

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
# include("operators/LBFGS.jl")
# include("operators/BlkDiagLBFGS.jl")
include("operators/utils.jl")

# Calculus rules

include("calculus/DCAT.jl")
include("calculus/HCAT.jl")
include("calculus/VCAT.jl")
include("calculus/Compose.jl")
include("calculus/Reshape.jl")
include("calculus/Scale.jl")
include("calculus/Sum.jl")
include("calculus/Transpose.jl")

# Syntax
include("syntax.jl")

size(L::LinearOperator, i::Int) = size(L)[i]
ndims(L::LinearOperator) = length(size(L,1)), length(size(L,2))
ndims(L::LinearOperator, i::Int) = ndims(L)[i]

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
	return *(s...)
end


end
