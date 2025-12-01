module AbstractOperators

using LinearAlgebra, Random
using Base.Cartesian: @ncall, @ntuple, @nloops, @nref
using Polyester: @batch, disable_polyester_threads
using FastBroadcast: FastBroadcast, @..
using RecursiveArrayTools: ArrayPartition

abstract type AbstractOperator end

abstract type LinearOperator <: AbstractOperator end
abstract type NonLinearOperator <: AbstractOperator end

import LinearAlgebra: mul!
import Base: size, ndims, @lock
import Base.Threads: @spawn, @threads, nthreads

import OperatorCore: 
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
    is_symmetric,
    is_positive_definite,
    is_positive_semidefinite

export LinearOperator, NonLinearOperator, AbstractOperator
export mul!

const DEBUG_COMPOSE = Ref{Bool}(false)

# Predicates and properties

include("utils.jl")
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
include("linearoperators/FiniteDiff.jl")
include("linearoperators/Variation.jl")
include("linearoperators/LBFGS.jl")

# Batch operators
include("batching/BatchOp.jl")
include("batching/SimpleBatchOp.jl")
include("batching/SpreadingBatchOp.jl")

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
