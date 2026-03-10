# Standalone: julia --project=test test/jet/test_call.jl
@testitem "@test_call" tags = [:jet, :base] begin
    using JET, RecursiveArrayTools, AbstractOperators
    const AO = AbstractOperators
    n, m = 8, 4
    d = randn(n)
    M = randn(n, n)
    x = randn(n)
    y = zeros(n)
    b = randn(n)

    # ── Constructors ─────────────────────────────────────────────────────────

    # Core linear operators
    @test_call target_modules = (AO,) DiagOp(d)
    @test_call target_modules = (AO,) Eye(n)
    @test_call target_modules = (AO,) ZeroPad((n,), m)
    @test_call target_modules = (AO,) GetIndex((n,), 1:4)
    @test_call target_modules = (AO,) Variation(n, 2)
    @test_call target_modules = (AO,) FiniteDiff((n,))
    @test_call target_modules = (AO,) Zeros(Float64, (n,), Float64, (n,))
    @test_call target_modules = (AO,) MatrixOp(M)
    @test_call target_modules = (AO,) LMatrixOp(randn(n), n)
    @test_call target_modules = (AO,) MyLinOp(
        Float64, (n,), Float64, (n,),
        (yy, xx) -> (yy .= xx),
        (yy, xx) -> (yy .= xx),
    )

    # Calculus / composition operators
    @test_call target_modules = (AO,) Compose(Eye(n), DiagOp(d))
    @test_call target_modules = (AO,) Sum(Eye(n), DiagOp(d))
    @test_call target_modules = (AO,) Scale(2.0, Eye(n))
    @test_call target_modules = (AO,) AdjointOperator(MatrixOp(M))
    @test_call target_modules = (AO,) HCAT(Eye(n), DiagOp(d))
    @test_call target_modules = (AO,) VCAT(Eye(n), DiagOp(d))
    @test_call target_modules = (AO,) DCAT(Eye(n), Eye(n))
    @test_call target_modules = (AO,) Reshape(Eye(n), 2, 4)
    @test_call target_modules = (AO,) HadamardProd(MatrixOp(M), MatrixOp(M))
    @test_call target_modules = (AO,) AffineAdd(MatrixOp(M), b)
    @test_call target_modules = (AO,) BroadCast(Eye(n), (n, 2))
    @test_call target_modules = (AO,) Ax_mul_Bx(MatrixOp(M, n), MatrixOp(M, n))
    @test_call target_modules = (AO,) Axt_mul_Bx(MatrixOp(M), MatrixOp(M))
    @test_call target_modules = (AO,) Ax_mul_Bxt(MatrixOp(M), MatrixOp(M))

    # Nonlinear operators
    @test_call target_modules = (AO,) Sigmoid(Float64, (n,), 2)
    @test_call target_modules = (AO,) SoftMax(Float64, (n,))
    @test_call target_modules = (AO,) SoftPlus(Float64, (n,))
    @test_call target_modules = (AO,) Exp(Float64, (n,))
    @test_call target_modules = (AO,) Sin(Float64, (n,))
    @test_call target_modules = (AO,) Cos(Float64, (n,))
    @test_call target_modules = (AO,) Atan(Float64, (n,))
    @test_call target_modules = (AO,) Tanh(Float64, (n,))
    @test_call target_modules = (AO,) Sech(Float64, (n,))
    @test_call target_modules = (AO,) Pow(Float64, (n,), 2.0)

    # Jacobian
    @test_call target_modules = (AO,) Jacobian(Sigmoid(Float64, (n,), 2), x)

    # Batching operators
    # SimpleBatchOp:
    @test_call target_modules = (AO,) BatchOp([DiagOp(randn(2)) for _ in 1:2], 3)
    # SpreadingBatchOp constructor — 3 known @test_opt issues in shape calculation
    # (symbol_mask_to_bool returns Vararg tuple; create_BatchOp kwcall dispatch):
    @test_call target_modules = (AO,) BatchOp([DiagOp(randn(2)) for _ in 1:3], 4)

    # ── mul! ─────────────────────────────────────────────────────────────────

    # Core / linear
    @test_call target_modules = (AO,) mul!(y, Eye(n), x)
    @test_call target_modules = (AO,) mul!(y, DiagOp(d), x)
    @test_call target_modules = (AO,) mul!(y, MatrixOp(M), x)
    @test_call target_modules = (AO,) mul!(y, AdjointOperator(MatrixOp(M)), x)

    # ZeroPad
    @test_call target_modules = (AO,) mul!(zeros(n + m), ZeroPad((n,), (m,)), x)
    @test_call target_modules = (AO,) mul!(x, AdjointOperator(ZeroPad((n,), (m,))), zeros(n + m))

    # GetIndex
    @test_call target_modules = (AO,) mul!(zeros(m), GetIndex((n,), 1:m), x)
    @test_call target_modules = (AO,) mul!(x, AdjointOperator(GetIndex((n,), 1:m)), zeros(m))

    # Zeros
    @test_call target_modules = (AO,) mul!(y, Zeros(Float64, (n,), Float64, (n,)), x)
    @test_call target_modules = (AO,) mul!(y, AdjointOperator(Zeros(Float64, (n,), Float64, (n,))), x)

    # Reshape
    @test_call target_modules = (AO,) mul!(zeros(2, n ÷ 2), Reshape(Eye(n), 2, n ÷ 2), x)
    @test_call target_modules = (AO,) mul!(x, AdjointOperator(Reshape(Eye(n), 2, n ÷ 2)), zeros(2, n ÷ 2))

    # Calculus
    @test_call target_modules = (AO,) mul!(y, Scale(2.0, Eye(n)), x)
    @test_call target_modules = (AO,) mul!(y, Compose(Eye(n), DiagOp(d)), x)
    @test_call target_modules = (AO,) mul!(y, Sum(Eye(n), DiagOp(d)), x)
    @test_call target_modules = (AO,) mul!(y, AffineAdd(MatrixOp(M), randn(n)), x)
    @test_call target_modules = (AO,) mul!(y, HadamardProd(MatrixOp(M), MatrixOp(M)), x)

    # BroadCast, LMatrixOp, Variation
    @test_call target_modules = (AO,) mul!(zeros(n, 2), BroadCast(Eye(n), (n, 2)), x)
    @test_call target_modules = (AO,) mul!(y, LMatrixOp(randn(n), n), randn(n, n))
    @test_call target_modules = (AO,) mul!(zeros(9, 2), Variation(3, 3), randn(3, 3))

    # FiniteDiff forward and adjoint
    @test_call target_modules = (AO,) mul!(zeros(n - 1), FiniteDiff((n,)), x)
    @test_call target_modules = (AO,) mul!(x, AdjointOperator(FiniteDiff((n,))), zeros(n - 1))

    # HCAT, VCAT, DCAT (ArrayPartition I/O)
    @test_call target_modules = (AO,) mul!(y, HCAT(Eye(n), DiagOp(d)), ArrayPartition(randn(n), randn(n)))
    @test_call target_modules = (AO,) mul!(ArrayPartition(zeros(n), zeros(n)), VCAT(Eye(n), DiagOp(d)), x)
    @test_call target_modules = (AO,) mul!(
        ArrayPartition(zeros(n), zeros(n)),
        DCAT(Eye(n), DiagOp(d)),
        ArrayPartition(randn(n), randn(n)),
    )

    # 2D / batch operators
    x2 = randn(n, n)
    y2 = zeros(n, n)
    @test_call target_modules = (AO,) mul!(y2, Ax_mul_Bx(MatrixOp(M, n), MatrixOp(M, n)), x2)

    # SpreadingBatchOp mul! (0 @test_opt issues)
    @test_call target_modules = (AO,) mul!(
        zeros(2, 3, 4),
        BatchOp([DiagOp(randn(2)) for _ in 1:3], 4; threaded = false),
        randn(2, 3, 4),
    )

    # ── Exported utility functions ────────────────────────────────────────────
    @test_call target_modules = (AO,) update!(LBFGS(zeros(n), 3), randn(n), randn(n), randn(n), randn(n))
    @test_call target_modules = (AO,) reset!(LBFGS(zeros(n), 3))
end
