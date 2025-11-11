using WaveletOperators, AbstractOperators
using Test, LinearAlgebra, Wavelets, RecursiveArrayTools
using Aqua

include(joinpath(@__DIR__, "..", "..", "test", "utils.jl"))

const verb = false

@testset "WaveletOperators" begin
    include("test_wavelet_operators.jl")

    @testset "Aqua" begin
        Aqua.test_all(WaveletOperators)
    end
end
