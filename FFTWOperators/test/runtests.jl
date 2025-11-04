using FFTWOperators, AbstractOperators
using Test, LinearAlgebra, FFTW, RecursiveArrayTools
using AbstractOperators: can_be_combined, combine
using Aqua

include(joinpath(@__DIR__, "..", "..", "test", "utils.jl"))

const verb = false

@testset "FFTWOperators" begin
    include("test_fftw_operators.jl")
    include("test_shift_operators.jl")

    @testset "Transform Combinations" begin
        n = 8  # Power of 2 for DCT
        
        # Test DCT combinations
        dct_op = DCT(n)
        idct_op = IDCT(n)
        
        @test can_be_combined(dct_op, idct_op)
        @test can_be_combined(idct_op, dct_op)
        
        combined_dct = combine(dct_op, idct_op)
        @test combined_dct isa Eye
        
        # Test DFT combinations
        dft_op = DFT(ComplexF64, n)
        idft_op = IDFT(n)
        
        @test can_be_combined(dft_op, idft_op)
        @test can_be_combined(idft_op, dft_op)
        
        combined_dft = combine(dft_op, idft_op)
        @test combined_dft isa Eye
    end

    @testset "Aqua" begin
        Aqua.test_all(FFTWOperators, piracies=false)
        Aqua.test_piracies(FFTWOperators; treat_as_own=[FFTW, AbstractOperators])
    end
end
