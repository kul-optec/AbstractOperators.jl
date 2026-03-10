# Standalone: julia --project=test test/jet/test_opt_constructors.jl
@testitem "@test_opt constructors" tags = [:jet, :base] begin
    using JET, AbstractOperators
    const AO = AbstractOperators
    n = 8
    m = 4
    d = randn(n)
    M = randn(n, n)
    x = randn(n)
    y = zeros(n)
    b = randn(n)

    # Core linear operators
    @test_opt target_modules = (AO,) DiagOp(d)
    @test_opt target_modules = (AO,) Eye(n)
    @test_opt target_modules = (AO,) GetIndex((n,), 1:4)
    @test_opt target_modules = (AO,) ZeroPad((n,), m)
    @test_opt target_modules = (AO,) Variation(n, 2)
    @test_opt target_modules = (AO,) FiniteDiff((n,))
    @test_opt target_modules = (AO,) MatrixOp(M)

    # Calculus operators
    @test_opt target_modules = (AO,) Compose(Eye(n), DiagOp(d))
    @test_opt target_modules = (AO,) Sum(Eye(n), DiagOp(d))
    @test_opt target_modules = (AO,) Scale(2.0, Eye(n))
    @test_opt target_modules = (AO,) AdjointOperator(MatrixOp(M))
    @test_opt target_modules = (AO,) HCAT(Eye(n), DiagOp(d))
    @test_opt target_modules = (AO,) VCAT(Eye(n), DiagOp(d))
    @test_opt target_modules = (AO,) DCAT(Eye(n), Eye(n))
    @test_opt target_modules = (AO,) Reshape(Eye(n), 2, 4)
    @test_opt target_modules = (AO,) HadamardProd(MatrixOp(M), MatrixOp(M))
    @test_opt target_modules = (AO,) AffineAdd(MatrixOp(M), b)
    @test_opt target_modules = (AO,) BroadCast(Eye(n), (n, 2))
    @test_opt target_modules = (AO,) Ax_mul_Bx(MatrixOp(M, n), MatrixOp(M, n))
    @test_opt target_modules = (AO,) Axt_mul_Bx(MatrixOp(M), MatrixOp(M))
    @test_opt target_modules = (AO,) Ax_mul_Bxt(MatrixOp(M), MatrixOp(M))

    # Nonlinear operators
    @test_opt target_modules = (AO,) Sigmoid(Float64, (n,), 2)
    @test_opt target_modules = (AO,) SoftMax(Float64, (n,))
    @test_opt target_modules = (AO,) SoftPlus(Float64, (n,))
    @test_opt target_modules = (AO,) Exp(Float64, (n,))
    @test_opt target_modules = (AO,) Sin(Float64, (n,))
    @test_opt target_modules = (AO,) Cos(Float64, (n,))
    @test_opt target_modules = (AO,) Atan(Float64, (n,))
    @test_opt target_modules = (AO,) Tanh(Float64, (n,))
    @test_opt target_modules = (AO,) Sech(Float64, (n,))
    @test_opt target_modules = (AO,) Pow(Float64, (n,), 2.0)

    # API functions
    @test_opt target_modules = (AO,) mul!(y, Eye(n), x)
    @test_opt target_modules = (AO,) Jacobian(Sigmoid(Float64, (n,), 2), x)
end
