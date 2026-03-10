# Standalone: julia --project=test test/jet/test_opt_mul.jl
@testitem "@test_opt mul!" tags = [:jet, :base] begin
    using JET, RecursiveArrayTools, AbstractOperators
    const AO = AbstractOperators
    n, m = 8, 4
    d = randn(n)
    M = randn(n, n)
    x = randn(n)
    y = zeros(n)

    # ── Linear operators ──────────────────────────────────────────────────────
    @test_opt target_modules = (AO,) mul!(y, Eye(n), x)
    @test_opt target_modules = (AO,) mul!(y, DiagOp(d), x)
    @test_opt target_modules = (AO,) mul!(y, MatrixOp(M), x)
    @test_opt target_modules = (AO,) mul!(y, AdjointOperator(MatrixOp(M)), x)
    # MatrixOp batched (NC=m columns): forward and adjoint
    @test_opt target_modules = (AO,) mul!(zeros(n, m), MatrixOp(M, m), randn(n, m))
    @test_opt target_modules = (AO,) mul!(zeros(n, m), AdjointOperator(MatrixOp(M, m)), randn(n, m))

    # ZeroPad (codomain = ℝ^(n+m))
    @test_opt target_modules = (AO,) mul!(zeros(n + m), ZeroPad((n,), (m,)), x)
    @test_opt target_modules = (AO,) mul!(x, AdjointOperator(ZeroPad((n,), (m,))), zeros(n + m))

    # GetIndex (codomain = ℝ^m)
    @test_opt target_modules = (AO,) mul!(zeros(m), GetIndex((n,), 1:m), x)
    @test_opt target_modules = (AO,) mul!(x, AdjointOperator(GetIndex((n,), 1:m)), zeros(m))

    # Zeros
    @test_opt target_modules = (AO,) mul!(y, Zeros(Float64, (n,), Float64, (n,)), x)
    @test_opt target_modules = (AO,) mul!(y, AdjointOperator(Zeros(Float64, (n,), Float64, (n,))), x)

    # Reshape (2 × n/2 ↔ n)
    @test_opt target_modules = (AO,) mul!(zeros(2, n ÷ 2), Reshape(Eye(n), 2, n ÷ 2), x)
    @test_opt target_modules = (AO,) mul!(x, AdjointOperator(Reshape(Eye(n), 2, n ÷ 2)), zeros(2, n ÷ 2))

    # Calculus: Scale, Compose, Sum
    @test_opt target_modules = (AO,) mul!(y, Scale(2.0, Eye(n)), x)
    @test_opt target_modules = (AO,) mul!(y, Compose(Eye(n), DiagOp(d)), x)
    @test_opt target_modules = (AO,) mul!(y, Sum(Eye(n), DiagOp(d)), x)
    # Nested Sum flattening path — type-stable via @generated constructor
    @test_opt target_modules = (AO,) mul!(y, Sum(Sum(Eye(n), DiagOp(d)), DiagOp(d)), x)

    # AffineAdd, HadamardProd
    @test_opt target_modules = (AO,) mul!(y, AffineAdd(MatrixOp(M), randn(n)), x)
    @test_opt target_modules = (AO,) mul!(y, HadamardProd(MatrixOp(M), MatrixOp(M)), x)

    # BroadCast (codomain = ℝ^(n×2))
    @test_opt target_modules = (AO,) mul!(zeros(n, 2), BroadCast(Eye(n), (n, 2)), x)

    # LMatrixOp: ℝ^(n×n) → ℝ^n
    @test_opt target_modules = (AO,) mul!(y, LMatrixOp(randn(n), n), randn(n, n))

    # Variation: ℝ^(3×3) → ℝ^(9×2)
    @test_opt target_modules = (AO,) mul!(zeros(9, 2), Variation(3, 3), randn(3, 3))

    # FiniteDiff forward (codomain = ℝ^(n-1)) and adjoint (domain = ℝ^(n-1))
    @test_opt target_modules = (AO,) mul!(zeros(n - 1), FiniteDiff((n,)), x)
    @test_opt target_modules = (AO,) mul!(x, AdjointOperator(FiniteDiff((n,))), zeros(n - 1))

    # HCAT: ArrayPartition → Vector (forward)
    @test_opt target_modules = (AO,) mul!(y, HCAT(Eye(n), DiagOp(d)), ArrayPartition(randn(n), randn(n)))
    # HCAT adjoint: Vector → ArrayPartition (backward)
    @test_opt target_modules = (AO,) mul!(ArrayPartition(zeros(n), zeros(n)), AdjointOperator(HCAT(Eye(n), DiagOp(d))), y)

    # DCAT: ArrayPartition ↔ ArrayPartition
    @test_opt target_modules = (AO,) mul!(
        ArrayPartition(zeros(n), zeros(n)),
        DCAT(Eye(n), DiagOp(d)),
        ArrayPartition(randn(n), randn(n)),
    )

    # VCAT: Vector → ArrayPartition (forward)
    @test_opt target_modules = (AO,) mul!(ArrayPartition(zeros(n), zeros(n)), VCAT(Eye(n), DiagOp(d)), x)
    # VCAT adjoint: ArrayPartition → Vector (backward)
    @test_opt target_modules = (AO,) mul!(x, AdjointOperator(VCAT(Eye(n), DiagOp(d))), ArrayPartition(randn(n), randn(n)))

    # ── 2D / batch operators ──────────────────────────────────────────────────
    x2 = randn(n, n)
    y2 = zeros(n, n)
    @test_opt target_modules = (AO,) mul!(y2, Ax_mul_Bx(MatrixOp(M, n), MatrixOp(M, n)), x2)
    @test_opt target_modules = (AO,) mul!(y2, Axt_mul_Bx(MatrixOp(M), MatrixOp(M)), x2)
    @test_opt target_modules = (AO,) mul!(y2, Ax_mul_Bxt(MatrixOp(M), MatrixOp(M)), x2)

    # MyLinOp: custom linear operator (identity)
    let my_op = MyLinOp(Float64, (n,), Float64, (n,), (yy, xx) -> (yy .= xx), (yy, xx) -> (yy .= xx))
        @test_opt target_modules = (AO,) mul!(y, my_op, x)
        @test_opt target_modules = (AO,) mul!(y, AdjointOperator(my_op), x)
    end

    # SpreadingBatchOp (single-threaded): ℝ^(2×3×4) → ℝ^(2×3×4)
    @test_opt target_modules = (AO,) mul!(
        zeros(2, 3, 4),
        BatchOp([DiagOp(randn(2)) for _ in 1:3], 4; threaded = false),
        randn(2, 3, 4),
    )

    # ── Nonlinear operators: forward ──────────────────────────────────────────
    @test_opt target_modules = (AO,) mul!(y, Sigmoid(Float64, (n,), 2.0), x)
    @test_opt target_modules = (AO,) mul!(y, Exp(Float64, (n,)), x)
    @test_opt target_modules = (AO,) mul!(y, Sin(Float64, (n,)), x)
    @test_opt target_modules = (AO,) mul!(y, Cos(Float64, (n,)), x)
    @test_opt target_modules = (AO,) mul!(y, Tanh(Float64, (n,)), x)
    @test_opt target_modules = (AO,) mul!(y, Atan(Float64, (n,)), x)
    @test_opt target_modules = (AO,) mul!(y, Sech(Float64, (n,)), x)
    @test_opt target_modules = (AO,) mul!(y, SoftMax(Float64, (n,)), x)
    @test_opt target_modules = (AO,) mul!(y, SoftPlus(Float64, (n,)), x)
    @test_opt target_modules = (AO,) mul!(y, Pow(Float64, (n,), 2), x)

    # ── Nonlinear operators: Jacobian adjoint ──────────────────────────────────
    # Pattern: create op, instantiate Jacobian at x, then mul!(grad, J', b)
    let op = Sigmoid(Float64, (n,), 2.0), J = Jacobian(op, x)
        @test_opt target_modules = (AO,) mul!(y, J', y)
    end
    let op = Exp(Float64, (n,)), J = Jacobian(op, x)
        @test_opt target_modules = (AO,) mul!(y, J', y)
    end
    let op = Sin(Float64, (n,)), J = Jacobian(op, x)
        @test_opt target_modules = (AO,) mul!(y, J', y)
    end
    let op = Cos(Float64, (n,)), J = Jacobian(op, x)
        @test_opt target_modules = (AO,) mul!(y, J', y)
    end
    let op = Tanh(Float64, (n,)), J = Jacobian(op, x)
        @test_opt target_modules = (AO,) mul!(y, J', y)
    end
    let op = Atan(Float64, (n,)), J = Jacobian(op, x)
        @test_opt target_modules = (AO,) mul!(y, J', y)
    end
    let op = Sech(Float64, (n,)), J = Jacobian(op, x)
        @test_opt target_modules = (AO,) mul!(y, J', y)
    end
    let op = SoftMax(Float64, (n,)), J = Jacobian(op, x)
        @test_opt target_modules = (AO,) mul!(y, J', y)
    end
    let op = SoftPlus(Float64, (n,)), J = Jacobian(op, x)
        @test_opt target_modules = (AO,) mul!(y, J', y)
    end
    let op = Pow(Float64, (n,), 2), J = Jacobian(op, x)
        @test_opt target_modules = (AO,) mul!(y, J', y)
    end
end
