using AbstractOperators
using Base.Test
using Base.Profile

srand(0)

include("utils.jl")

include("test_deep.jl")
include("test_operators.jl")
include("test_operators_calculus.jl")
include("test_syntax.jl")
include("test_lbfgs.jl")
include("test_lbfgs_larger.jl")

