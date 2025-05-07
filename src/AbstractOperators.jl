module AbstractOperators

using LinearAlgebra, DSP, FFTW, RecursiveArrayTools, FastBroadcast

const RealOrComplex{R} = Union{R,Complex{R}}
abstract type AbstractOperator end

abstract type LinearOperator <: AbstractOperator end
abstract type NonLinearOperator <: AbstractOperator end

import LinearAlgebra: mul!
import Base: size, ndims, AbstractLock, @lock
import Base.Threads: @spawn, @threads, nthreads

import OperatorCore: 
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
    get_slicing_expr,
    get_slicing_mask,
    remove_slicing

export LinearOperator, NonLinearOperator, AbstractOperator
export mul!

# Predicates and properties

include("properties.jl")
include("calculus/AdjointOperator.jl")
include("calculus/Scale.jl")

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

include("calculus/DCAT.jl")
include("calculus/HCAT.jl")
include("calculus/VCAT.jl")
include("calculus/Compose.jl")
include("calculus/Reshape.jl")
include("calculus/BroadCast.jl")
include("calculus/Sum.jl")
include("calculus/AffineAdd.jl")
include("calculus/Jacobian.jl")
include("calculus/Axt_mul_Bx.jl")
include("calculus/Ax_mul_Bxt.jl")
include("calculus/Ax_mul_Bx.jl")
include("calculus/HadamardProd.jl")

# Batch operators
include("batching/BatchOp.jl")
include("batching/SimpleBatchOp.jl")
include("batching/SpreadingBatchOp.jl")
include("batching/threading_utils.jl")

# Non-Linear operators
include("nonlinearoperators/Pow.jl")
include("nonlinearoperators/Exp.jl")
include("nonlinearoperators/Sin.jl")
include("nonlinearoperators/Cos.jl")
include("nonlinearoperators/Atan.jl")
include("nonlinearoperators/Tanh.jl")
include("nonlinearoperators/Sech.jl")
include("nonlinearoperators/Sigmoid.jl")
include("nonlinearoperators/SoftMax.jl")
include("nonlinearoperators/SoftPlus.jl")

# Others
include("syntax.jl")
include("combination_rules.jl")

end
