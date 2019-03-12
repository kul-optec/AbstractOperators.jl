__precompile__()

module AbstractOperators

using LinearAlgebra, AbstractFFTs, DSP, FFTW, RecursiveArrayTools


const RealOrComplex{R} = Union{R, Complex{R}}
abstract type AbstractOperator end

abstract type LinearOperator    <: AbstractOperator end
abstract type NonLinearOperator <: AbstractOperator end

import LinearAlgebra: mul!

export LinearOperator,
       NonLinearOperator,
       AbstractOperator
export mul!

# Predicates and properties

include("properties.jl")
include("calculus/AdjointOperator.jl")

## Linear operators

include("linearoperators/MyLinOp.jl")
include("linearoperators/Zeros.jl")
include("linearoperators/ZeroPad.jl")
include("linearoperators/Eye.jl")
include("linearoperators/DiagOp.jl")
include("linearoperators/GetIndex.jl")
include("linearoperators/MatrixOp.jl")
include("linearoperators/LMatrixOp.jl")
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

# Calculus rules

include("calculus/Scale.jl")
include("calculus/DCAT.jl")
include("calculus/HCAT.jl")
include("calculus/VCAT.jl")
include("calculus/Compose.jl")
include("calculus/Reshape.jl")
include("calculus/BroadCast.jl")
include("calculus/Sum.jl")
include("calculus/AffineAdd.jl")
include("calculus/Jacobian.jl")
include("calculus/NonLinearCompose.jl")
include("calculus/SelfCompose.jl")
include("calculus/Hadamard.jl")

# Non-Linear operators
include("nonlinearoperators/Pow.jl")
include("nonlinearoperators/Exp.jl")
include("nonlinearoperators/Sin.jl")
include("nonlinearoperators/Cos.jl")
include("nonlinearoperators/Atan.jl")
include("nonlinearoperators/Tanh.jl")
include("nonlinearoperators/Sigmoid.jl")
include("nonlinearoperators/SoftMax.jl")
include("nonlinearoperators/SoftPlus.jl")

# Syntax
include("syntax.jl")

end
