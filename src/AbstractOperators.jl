__precompile__()

module AbstractOperators

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

abstract type AbstractOperator end

abstract type LinearOperator    <: AbstractOperator end
abstract type NonLinearOperator <: AbstractOperator end

import Base: A_mul_B!, Ac_mul_B! 

export LinearOperator,
       NonLinearOperator,
       AbstractOperator

# deep stuff

include("utilities/deep.jl")

# predicates and properties
include("properties.jl")

# Linear operators

include("linearoperators/MyLinOp.jl")
include("linearoperators/Zeros.jl")
include("linearoperators/ZeroPad.jl")
include("linearoperators/Eye.jl")
include("linearoperators/DiagOp.jl")
include("linearoperators/GetIndex.jl")
include("linearoperators/MatrixOp.jl")
include("linearoperators/MatrixMul.jl")
include("linearoperators/DFT.jl")
include("linearoperators/RDFT.jl")
include("linearoperators/IRDFT.jl")
include("linearoperators/DCT.jl")
include("linearoperators/FiniteDiff.jl")
include("linearoperators/Variation.jl")
include("linearoperators/Conv.jl")
include("linearoperators/Filt.jl")
include("linearoperators/MIMOFilt.jl")
include("linearoperators/Xcorr.jl")
include("linearoperators/LBFGS.jl")
include("linearoperators/BlkDiagLBFGS.jl")

# Calculus rules

include("calculus/Scale.jl")
include("calculus/DCAT.jl")
include("calculus/HCAT.jl")
include("calculus/VCAT.jl")
include("calculus/Compose.jl")
include("calculus/Reshape.jl")
include("calculus/BroadCast.jl")
include("calculus/Sum.jl")
include("calculus/Transpose.jl")
include("calculus/Jacobian.jl")
include("calculus/NonLinearCompose.jl")

# Non-Linear operators
include("nonlinearoperators/Sigmoid.jl")
include("nonlinearoperators/SoftMax.jl")
include("nonlinearoperators/SoftPlus.jl")

# Syntax
include("syntax.jl")




end
