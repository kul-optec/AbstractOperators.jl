if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)

@testset "Ax_mul_Bxt" begin
    verb && println(" --- Testing Ax_mul_Bxt --- ")
    n = 10
    A, B = Eye(n), Sin(n)
    P = Ax_mul_Bxt(A, B)
    x = randn(n)
    r = randn(n, n)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((A * x) * (B * x)' - y) < 1e-9

    n, m = 3, 4
    A, B = MatrixOp(randn(n, m)), MatrixOp(randn(n, m))
    P = Ax_mul_Bxt(A, B)
    x = randn(m)
    r = randn(n, n)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((A * x) * (B * x)' - y) < 1e-8

    n, m, l = 3, 7, 5
    A, B = MatrixOp(randn(n, m), l), MatrixOp(randn(n, m), l)
    P = Ax_mul_Bxt(A, B)
    x = randn(m, l)
    r = randn(n, n)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((A * x) * (B * x)' - y) < 1e-8

    n, m = 3, 7
    A, B = Sin(n, m), Cos(n, m)
    P = Ax_mul_Bxt(A, B)
    x = randn(n, m)
    r = randn(n, n)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((A * x) * (B * x)' - y) < 1e-8

    # testing with HCAT
    m, n = 3, 5
    x = ArrayPartition(randn(m), randn(n))
    r = randn(m, m)
    b = randn(m)
    A = AffineAdd(Sin(Float64, (m,)), b)
    B = MatrixOp(randn(m, n))
    op1 = HCAT(A, B)
    C = Cos(Float64, (m,))
    D = MatrixOp(randn(m, n))
    op2 = HCAT(C, D)
    P = Ax_mul_Bxt(op1, op2)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((op1 * x) * (op2 * x)' - y) < 1e-8

    #test remove_displacement
    y2, grad = test_NLop(remove_displacement(P), x, r, verb)
    @test norm((op1 * x - b) * (op2 * x)' - y2) < 1e-8

    # test permute
    p = [2, 1]
    Pp = AbstractOperators.permute(P, p)
    xp = ArrayPartition(x.x[p])
    y2, grad = test_NLop(Pp, xp, r, verb)
    @test norm(y2 - y) < 1e-8

    @test_throws Exception Ax_mul_Bxt(Eye(2, 2), Eye(2, 1))
    @test_throws Exception Ax_mul_Bxt(Eye(2, 2, 2), Eye(2, 2, 2))

    # test equality
    n, m = 3, 4
    A, B = MatrixOp(randn(n, m)), MatrixOp(randn(n, m))
    @test Ax_mul_Bxt(A, B) == Ax_mul_Bxt(A, B)
    @test Jacobian(Ax_mul_Bxt(A, B), x) == Jacobian(Ax_mul_Bxt(A, B), x)
end
