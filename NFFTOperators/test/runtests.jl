using NFFTOperators, AbstractOperators
using Test, LinearAlgebra, FFTW, NFFT, RecursiveArrayTools
using Aqua

include(joinpath(@__DIR__, "..", "..", "test", "utils.jl"))

@testset "NFFTOperators" begin
    include("test_NfftOps.jl")

    @testset "Aqua" begin
        Aqua.test_all(NFFTOperators)
    end
end
