@testitem "WaveletOp" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, LinearAlgebra, Random, WaveletOperators

    ########## WaveletOp ############
    n = 8
    op = WaveletOp(Float64, wavelet(WT.db4), (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = dwt(x1, wavelet(WT.db4))

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n = 8
    op = WaveletOp(ComplexF64, wavelet(WT.db4), (n,))
    x1 = randn(ComplexF64, n)
    y1 = test_op(op, x1, randn(ComplexF64, n), verb)
    y2 = dwt(x1, wavelet(WT.db4))

    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "WaveletOp constructor errors" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, WaveletOperators
    wt = wavelet(WT.db4)

    # 1D: odd dimension
    @test_throws ArgumentError WaveletOp(Float64, wt, 5)
    # 1D: too many levels
    @test_throws ArgumentError WaveletOp(Float64, wt, 8, 100)

    # ND: odd dimension in tuple
    @test_throws ArgumentError WaveletOp(Float64, wt, (5, 8))
    # ND: too many levels
    @test_throws ArgumentError WaveletOp(Float64, wt, (8, 8), 100)
end
