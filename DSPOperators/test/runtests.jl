using DSPOperators, AbstractOperators
using Test, LinearAlgebra, DSP, RecursiveArrayTools
using Aqua

const verb = false
include(joinpath(@__DIR__, "..", "..", "test", "utils.jl"))

@testset "DSPOperators" begin
    include("test_dsp_operators.jl")

    @testset "Aqua" begin
        Aqua.test_all(DSPOperators)
    end
end
