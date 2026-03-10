# JET static analysis for FFTWOperators mul! methods
@testitem "@test_opt FFTWOperators mul!" tags = [:jet, :DCT, :IDCT, :DFT, :RDFT, :FFTShift, :IFFTShift] begin
    using JET, FFTW, FFTWOperators, LinearAlgebra, AbstractOperators
    n = 8   # power of 2 for DCT/RDFT compatibility

    # ── DCT ───────────────────────────────────────────────────────────────────
    let x = randn(n), y = zeros(n), op = DCT(Float64, (n,))
        @test_opt target_modules = (FFTWOperators,) mul!(y, op, x)
        @test_opt target_modules = (FFTWOperators,) mul!(y, AdjointOperator(op), y)
    end

    # ── IDCT ──────────────────────────────────────────────────────────────────
    let x = randn(n), y = zeros(n), op = IDCT(Float64, (n,))
        @test_opt target_modules = (FFTWOperators,) mul!(y, op, x)
        @test_opt target_modules = (FFTWOperators,) mul!(y, AdjointOperator(op), y)
    end

    # ── DFT (real domain: ℝ^n → ℂ^n) ─────────────────────────────────────────
    let x = randn(n), yc = zeros(ComplexF64, n), op = DFT(Float64, (n,))
        @test_opt target_modules = (FFTWOperators,) mul!(yc, op, x)
        @test_opt target_modules = (FFTWOperators,) mul!(x, AdjointOperator(op), yc)
    end

    # ── DFT (complex domain: ℂ^n → ℂ^n) ──────────────────────────────────────
    let xc = randn(ComplexF64, n), yc = zeros(ComplexF64, n), op = DFT(ComplexF64, (n,))
        @test_opt target_modules = (FFTWOperators,) mul!(yc, op, xc)
        @test_opt target_modules = (FFTWOperators,) mul!(xc, AdjointOperator(op), yc)
    end

    # ── RDFT (ℝ^n → ℂ^(n/2+1)) ───────────────────────────────────────────────
    let x = randn(n), yc = zeros(ComplexF64, n ÷ 2 + 1), op = RDFT(Float64, (n,))
        @test_opt target_modules = (FFTWOperators,) mul!(yc, op, x)
        @test_opt target_modules = (FFTWOperators,) mul!(x, AdjointOperator(op), yc)
    end

    # ── FFTShift / IFFTShift ──────────────────────────────────────────────────
    let x = randn(n), y = zeros(n), op = FFTShift(Float64, (n,))
        @test_opt target_modules = (FFTWOperators,) mul!(y, op, x)
        @test_opt target_modules = (FFTWOperators,) mul!(y, AdjointOperator(op), x)
    end
    let x = randn(n), y = zeros(n), op = IFFTShift(Float64, (n,))
        @test_opt target_modules = (FFTWOperators,) mul!(y, op, x)
        @test_opt target_modules = (FFTWOperators,) mul!(y, AdjointOperator(op), x)
    end

    # ── IDFT (complex → complex) ─────────────────────────────────────────────
    let xc = randn(ComplexF64, n), yc = zeros(ComplexF64, n), op = IDFT(ComplexF64, (n,))
        @test_opt target_modules = (FFTWOperators,) mul!(yc, op, xc)
        @test_opt target_modules = (FFTWOperators,) mul!(xc, AdjointOperator(op), yc)
    end

    # ── IRDFT (ℂ^(n/2+1) → ℝ^n) ─────────────────────────────────────────────
    let xc = zeros(ComplexF64, n ÷ 2 + 1), y_real = zeros(n), op = IRDFT(ComplexF64, (n ÷ 2 + 1,), n)
        @test_opt target_modules = (FFTWOperators,) mul!(y_real, op, xc)
        @test_opt target_modules = (FFTWOperators,) mul!(xc, AdjointOperator(op), y_real)
    end

    # ── SignAlternation ───────────────────────────────────────────────────────
    let x = randn(n), y = zeros(n), op = SignAlternation(Float64, (n,), (1,); threaded = false)
        @test_opt target_modules = (FFTWOperators,) mul!(y, op, x)
        @test_opt target_modules = (FFTWOperators,) mul!(y, AdjointOperator(op), x)
    end
end

# JET call analysis for FFTWOperators constructors
@testitem "@test_call FFTWOperators" tags = [:jet, :DCT, :IDCT, :DFT, :RDFT, :FFTShift, :IFFTShift] begin
    using JET, FFTW, FFTWOperators, AbstractOperators
    n = 8

    @test_call target_modules = (FFTWOperators,) DCT(Float64, (n,))
    @test_call target_modules = (FFTWOperators,) IDCT(Float64, (n,))
    @test_call target_modules = (FFTWOperators,) DFT(Float64, (n,))
    @test_call target_modules = (FFTWOperators,) DFT(ComplexF64, (n,))
    @test_call target_modules = (FFTWOperators,) IDFT(ComplexF64, (n,))
    @test_call target_modules = (FFTWOperators,) RDFT(Float64, (n,))
    @test_call target_modules = (FFTWOperators,) IRDFT(ComplexF64, (n ÷ 2 + 1,), n)
    @test_call target_modules = (FFTWOperators,) FFTShift(Float64, (n,))
    @test_call target_modules = (FFTWOperators,) IFFTShift(Float64, (n,))
    @test_call target_modules = (FFTWOperators,) SignAlternation(Float64, (n,), (1,); threaded = false)
end

# JET package-level analysis for FFTWOperators
@testitem "JET test_package FFTWOperators" tags = [:jet, :DCT, :IDCT, :DFT, :RDFT, :FFTShift, :IFFTShift] begin
    using JET, FFTWOperators
    JET.test_package(FFTWOperators; target_modules = (FFTWOperators,))
end
