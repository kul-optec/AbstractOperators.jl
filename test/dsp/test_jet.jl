# JET static analysis for DSPOperators mul! methods
@testitem "@test_opt DSPOperators mul!" tags = [:jet, :dsp] begin
    using DSPOperators, JET, DSP, AbstractOperators
    n = 10

    # ── Xcorr ─────────────────────────────────────────────────────────────────
    h = randn(3)
    let x = randn(n), op = Xcorr(Float64, (n,), h)
        m = 2 * max(n, length(h)) - 1
        y = zeros(m)
        @test_opt target_modules = (DSPOperators,) mul!(y, op, x)
        @test_opt target_modules = (DSPOperators,) mul!(x, AdjointOperator(op), y)
    end

    # ── Conv ──────────────────────────────────────────────────────────────────
    let x = randn(n), h_conv = randn(5)
        op = Conv(Float64, (n,), h_conv)
        m = n + length(h_conv) - 1
        y = zeros(m)
        @test_opt target_modules = (DSPOperators,) mul!(y, op, x)
        @test_opt target_modules = (DSPOperators,) mul!(x, AdjointOperator(op), y)
    end

    # ── Filt (FIR) ────────────────────────────────────────────────────────────
    let x = randn(n, 1)
        b_fir = Float64[1.0, 0.5, 0.25]
        op = Filt(Float64, (n,), b_fir)
        y = zeros(n, 1)
        @test_opt target_modules = (DSPOperators,) mul!(y, op, x)
        @test_opt target_modules = (DSPOperators,) mul!(x, AdjointOperator(op), y)
    end

    # ── Filt (IIR) ────────────────────────────────────────────────────────────
    let x = randn(n, 1)
        b_iir = Float64[1.0, 0.5]
        a_iir = Float64[1.0, -0.5]
        op = Filt(Float64, (n,), b_iir, a_iir)
        y = zeros(n, 1)
        @test_opt target_modules = (DSPOperators,) mul!(y, op, x)
        @test_opt target_modules = (DSPOperators,) mul!(x, AdjointOperator(op), y)
    end

    # ── MIMOFilt (2-input, 1-output MIMO system) ──────────────────────────────
    let
        m_mimo, n_mimo = 10, 2
        B = [[1.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0]]
        A = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        op = MIMOFilt(Float64, (m_mimo, n_mimo), B, A)
        x_mimo = randn(m_mimo, n_mimo)
        y_mimo = zeros(m_mimo, 1)
        @test_opt target_modules = (DSPOperators,) mul!(y_mimo, op, x_mimo)
        @test_opt target_modules = (DSPOperators,) mul!(x_mimo, AdjointOperator(op), y_mimo)
    end
end

# JET call analysis for DSPOperators constructors
@testitem "@test_call DSPOperators" tags = [:jet, :dsp] begin
    using DSPOperators, JET, DSP, AbstractOperators
    n = 10

    # ── Xcorr ─────────────────────────────────────────────────────────────────
    @test_call target_modules = (DSPOperators,) Xcorr(Float64, (n,), randn(3))

    # ── Conv ──────────────────────────────────────────────────────────────────
    @test_call target_modules = (DSPOperators,) Conv(Float64, (n,), randn(5))

    # ── Filt (FIR) ────────────────────────────────────────────────────────────
    @test_call target_modules = (DSPOperators,) Filt(Float64, (n,), Float64[1.0, 0.5, 0.25])

    # ── Filt (IIR) ────────────────────────────────────────────────────────────
    @test_call target_modules = (DSPOperators,) Filt(
        Float64, (n,), Float64[1.0, 0.5], Float64[1.0, -0.5]
    )

    # ── MIMOFilt ──────────────────────────────────────────────────────────────
    @test_call target_modules = (DSPOperators,) MIMOFilt(
        Float64,
        (10, 2),
        [[1.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
    )
end

# JET package-level analysis for DSPOperators
@testitem "JET test_package DSPOperators" tags = [:jet, :dsp] begin
    using JET, DSPOperators
    JET.test_package(DSPOperators; target_modules = (DSPOperators,))
end
