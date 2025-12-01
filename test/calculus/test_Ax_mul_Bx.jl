if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)

@testset "Ax_mul_Bx" begin
    verb && println(" --- Testing Ax_mul_Bx --- ")
    n = 3
    A, B = Eye(n, n), Eye(n, n)
    P = Ax_mul_Bx(A, B)
    x = randn(n, n)
    r = randn(n, n)
    y, grad = test_NLop(P, x, r, verb)
    @test norm(x * x - y) < 1e-9

    n = 3
    A, B = Sin(n, n), Cos(n, n)
    P = Ax_mul_Bx(A, B)
    x = randn(n, n)
    r = randn(n, n)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((A * x) * (B * x) - y) < 1e-9

    n = 3
    A, B, C = Sin(n, n), Cos(n, n), Atan(n, n)
    P = Ax_mul_Bx(A, B)
    P2 = Ax_mul_Bx(C, P)
    x = randn(n, n)
    r = randn(n, n)
    y, grad = test_NLop(P2, x, r, verb)
    @test norm((C * x) * (A * x) * (B * x) - y) < 1e-9

    n, l = 2, 3
    A, B = MatrixOp(randn(l, n), l), MatrixOp(randn(l, n), l)
    P = Ax_mul_Bx(A, B)
    x = randn(n, l)
    r = randn(l, l)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((A * x) * (B * x) - y) < 1e-8

    @test_throws Exception Ax_mul_Bx(Eye(2), Eye(2))
    @test_throws Exception Ax_mul_Bx(Eye(2, 2), Eye(2, 1))
    @test_throws Exception Ax_mul_Bx(Eye(2, 2, 2), Eye(2, 2, 2))

    # testing with HCAT
    m, n = 3, 5
    x = ArrayPartition(randn(n, n), randn(m, n))
    r = randn(n, n)
    b = randn(n, n)
    A = AffineAdd(Sin(Float64, (n, n)), b)
    B = MatrixOp(randn(n, m), n)
    op1 = HCAT(A, B)
    C = Sin(Float64, (n, n))
    D = MatrixOp(randn(n, m), n)
    op2 = HCAT(C, D)
    P = Ax_mul_Bx(op1, op2)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((op1 * x) * (op2 * x) - y) < 1e-8

    #test remove_displacement
    y2, grad = test_NLop(remove_displacement(P), x, r, verb)
    @test norm((op1 * x - b) * (op2 * x) - y2) < 1e-8

    # test permute
    p = [2, 1]
    Pp = AbstractOperators.permute(P, p)
    xp = ArrayPartition(x.x[p])
    y2, grad = test_NLop(Pp, xp, r, verb)
    @test norm(y2 - y) < 1e-8

    #### some combos of Ax_mul_Bx etc...
    n, m, l = 3, 7, 5
    A, B = MatrixOp(randn(n, m), l), MatrixOp(randn(n, m), l)
    P = Ax_mul_Bxt(A, B)
    P2 = Axt_mul_Bx(A, P)
    x = randn(m, l)
    r = randn(l, n)
    y, grad = test_NLop(P2, x, r, verb)
    @test norm((A * x)' * ((A * x) * (B * x)') - y) < 1e-8

    n, m, l, k = 3, 7, 5, 9
    A, B = MatrixOp(randn(n, m), l), MatrixOp(randn(n, m), l)
    C = MatrixOp(randn(k, m), l)
    P = Axt_mul_Bx(A, B)
    P2 = Ax_mul_Bx(C, P)
    x = randn(m, l)
    r = randn(k, l)
    y, grad = test_NLop(P2, x, r, verb)
    @test norm((C * x) * ((A * x)' * (B * x)) - y) < 1e-8

    n, m, l, k = 3, 7, 5, 9
    A, B = MatrixOp(randn(n, m), l), MatrixOp(randn(n, m), l)
    C = MatrixOp(randn(k, m), l)
    P = Axt_mul_Bx(A, B)
    P2 = Ax_mul_Bxt(C, P)
    x = randn(m, l)
    r = randn(k, l)
    y, grad = test_NLop(P2, x, r, verb)
    @test norm((C * x) * ((A * x)' * (B * x))' - y) < 1e-8
end
