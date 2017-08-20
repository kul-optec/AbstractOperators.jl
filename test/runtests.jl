using AbstractOperators
using Base.Test
using Base.Profile

include("utils.jl")
srand(0)

verb = true

@testset "AbstractOperators" begin

@testset "Tuple operations" begin
  include("test_deep.jl")
end

@testset "Linear operators" begin
  include("test_linear_operators.jl")
end

@testset "Non-Linear operators" begin
  include("test_nonlinear_operators.jl")
end

@testset "Calculus rules" begin
  include("test_linear_operators_calculus.jl")
  include("test_nonlinear_operators_calculus.jl")
end

@testset "Syntax shorthands" begin
  include("test_syntax.jl")
end

@testset "L-BFGS" begin
  include("test_lbfgs.jl")
  include("test_lbfgs_larger.jl")
end

end









