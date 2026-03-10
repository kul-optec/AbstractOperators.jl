# JET static analysis for WaveletOperators mul! methods
@testitem "@test_opt WaveletOperators mul!" tags = [:jet, :wavelet] begin
    using JET, Wavelets, WaveletOperators, AbstractOperators
    wt = wavelet(WT.db4)

    # ── WaveletOp (1D) ────────────────────────────────────────────────────────
    let n = 8, x = randn(n), y = zeros(n), op = WaveletOp(wt, n)
        @test_opt target_modules = (WaveletOperators,) mul!(y, op, x)
        @test_opt target_modules = (WaveletOperators,) mul!(y, AdjointOperator(op), x)
    end

    # ── WaveletOp (2D) ────────────────────────────────────────────────────────
    let n = 8, x = randn(n, n), y = zeros(n, n), op = WaveletOp(wt, (n, n))
        @test_opt target_modules = (WaveletOperators,) mul!(y, op, x)
        @test_opt target_modules = (WaveletOperators,) mul!(y, AdjointOperator(op), x)
    end
end

# JET call analysis for WaveletOperators constructors
@testitem "@test_call WaveletOperators" tags = [:jet, :wavelet] begin
    using JET, Wavelets, WaveletOperators
    wt = wavelet(WT.db4)
    n = 8

    @test_call target_modules = (WaveletOperators,) WaveletOp(wt, n)
    @test_call target_modules = (WaveletOperators,) WaveletOp(wt, (n, n))
end

# JET package-level analysis for WaveletOperators
@testitem "JET test_package WaveletOperators" tags = [:jet, :wavelet] begin
    using JET, WaveletOperators
    JET.test_package(WaveletOperators; target_modules = (WaveletOperators,))
end
