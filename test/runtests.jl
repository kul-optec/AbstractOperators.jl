using AbstractOperators
using LinearAlgebra, FFTW, DSP, SparseArrays
using Printf
using Random
using Test
using Profile

include("utils.jl")
Random.seed!(0)

verb = true

@testset "AbstractOperators" begin

@testset "Block Arrays" begin
  include("test_block.jl")
end

@testset "Linear operators" begin
  include("test_linear_operators.jl")
end
#
#@testset "Non-Linear operators" begin
#  include("test_nonlinear_operators.jl")
#end
#
@testset "Linear Calculus rules" begin
  include("test_linear_operators_calculus.jl")
end
#
#@testset "Nonlinear Calculus rules" begin
#  include("test_nonlinear_operators_calculus.jl")
#end
#
#@testset "L-BFGS" begin
#  include("test_lbfgs.jl")
#end
#
#@testset "Syntax shorthands" begin
#  include("test_syntax.jl")
#end

end

