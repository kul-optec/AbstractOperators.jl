# Split from test_linear_operators_calculus.jl: Mixed combination tests
if !isdefined(Main, :verb); verb = false; end
if !isdefined(Main, :test_op); include("../utils.jl"); end

@testset "Combinations" begin
    verb && println(" --- Testing Combinations --- ")

    ## test Compose of HCAT
    m1, m2, m3, m4 = 4, 7, 3, 2
    A1 = randn(m3, m1)
    A2 = randn(m3, m2)
    A3 = randn(m4, m3)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opH = HCAT(opA1, opA2)
    opC = Compose(opA3, opH)
    x1, x2 = randn(m1), randn(m2)
    y1 = test_op(opC, ArrayPartition(x1, x2), randn(m4), verb)

    y2 = A3 * (A1 * x1 + A2 * x2)

    @test norm(y1 - y2) < 1e-9

    opCp = AbstractOperators.permute(opC, [2, 1])
    y1 = test_op(opCp, ArrayPartition(x2, x1), randn(m4), verb)

    @test norm(y1 - y2) < 1e-9

    ## test HCAT of Compose of HCAT
    m5 = 10
    A4 = randn(m4, m5)
    x3 = randn(m5)
    opHC = HCAT(opC, MatrixOp(A4))
    x = ArrayPartition(x1, x2, x3)
    y1 = test_op(opHC, x, randn(m4), verb)

    @test norm(y1 - (y2 + A4 * x3)) < 1e-9

    p = randperm(ndoms(opHC, 2))
    opHP = AbstractOperators.permute(opHC, p)

    xp = ArrayPartition(x.x[p]...)

    y1 = test_op(opHP, xp, randn(m4), verb)

    pp = randperm(ndoms(opHC, 2))
    opHPP = AbstractOperators.permute(opHC, pp)
    xpp = ArrayPartition(x.x[pp]...)
    y1 = test_op(opHPP, xpp, randn(m4), verb)

    # test VCAT of HCAT's
    m1, m2, n1 = 4, 7, 3
    A1 = randn(n1, m1)
    A2 = randn(n1, m2)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opH1 = HCAT(opA1, opA2)

    m1, m2, n2 = 4, 7, 5
    A3 = randn(n2, m1)
    A4 = randn(n2, m2)
    opA3 = MatrixOp(A3)
    opA4 = MatrixOp(A4)
    opH2 = HCAT(opA3, opA4)

    opV = VCAT(opH1, opH2)
    x1, x2 = randn(m1), randn(m2)
    y1 = test_op(opV, ArrayPartition(x1, x2), ArrayPartition(randn(n1), randn(n2)), verb)
    y2 = ArrayPartition(A1 * x1 + A2 * x2, A3 * x1 + A4 * x2)
    @test norm(y1 - y2) <= 1e-12

    # test VCAT of HCAT's with complex num
    m1, m2, n1 = 4, 7, 5
    A1 = randn(n1, m1) + im * randn(n1, m1)
    opA1 = MatrixOp(A1)
    d1 = rand(ComplexF64, n1)
    opA2 = DiagOp(Float64, (n1,), d1)
    opH1 = HCAT(opA1, opA2)

    m1, m2, n2 = 4, 7, 5
    A3 = randn(n2, m1) + im * randn(n2, m1)
    opA3 = MatrixOp(A3)
    d2 = rand(ComplexF64, n2)
    opA4 = DiagOp(Float64, (n2,), d2)
    opH2 = HCAT(opA3, opA4)

    opV = VCAT(opH1, opH2)
    x1, x2 = randn(m1) + im * randn(m1), randn(n2)
    y1 = test_op(
        opV,
        ArrayPartition(x1, x2),
        ArrayPartition(randn(n1) + im * randn(n1), randn(n2) + im * randn(n2)),
        verb,
    )
    y2 = ArrayPartition(A1 * x1 + x2 .* d1, A3 * x1 + x2 .* d2)
    @test norm(y1 - y2) <= 1e-12

    # test HCAT of VCAT's

    n1, n2, m1, m2 = 3, 5, 4, 7
    A = randn(m1, n1);
    opA = MatrixOp(A);
    B = randn(m1, n2);
    opB = MatrixOp(B);
    C = randn(m2, n1);
    opC = MatrixOp(C);
    D = randn(m2, n2);
    opD = MatrixOp(D);
    opV = HCAT(VCAT(opA, opC), VCAT(opB, opD))
    x1 = randn(n1)
    x2 = randn(n2)
    y1 = test_op(opV, ArrayPartition(x1, x2), ArrayPartition(randn(m1), randn(m2)), verb)
    y2 = ArrayPartition(A * x1 + B * x2, C * x1 + D * x2)

    @test norm(y1 - y2) <= 1e-12

    # test Sum of HCAT's

    m, n1, n2, n3 = 4, 7, 5, 3
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    A3 = randn(m, n3)
    B1 = randn(m, n1)
    B2 = randn(m, n2)
    B3 = randn(m, n3)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opB1 = MatrixOp(B1)
    opB2 = MatrixOp(B2)
    opB3 = MatrixOp(B3)
    opHA = HCAT(opA1, opA2, opA3)
    opHB = HCAT(opB1, opB2, opB3)
    opS = Sum(opHA, opHB)
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = test_op(opS, ArrayPartition(x1, x2, x3), randn(m), verb)
    y2 = A1 * x1 + B1 * x1 + A2 * x2 + B2 * x2 + A3 * x3 + B3 * x3

    @test norm(y1 - y2) <= 1e-12

    p = [3; 2; 1]
    opSp = AbstractOperators.permute(opS, p)
    y1 = test_op(opSp, ArrayPartition(((x1, x2, x3)[p])...), randn(m), verb)

    # test Sum of VCAT's

    m1, m2, n = 4, 7, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    B1 = randn(m1, n)
    B2 = randn(m2, n)
    C1 = randn(m1, n)
    C2 = randn(m2, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opB1 = MatrixOp(B1)
    opB2 = MatrixOp(B2)
    opC1 = MatrixOp(C1)
    opC2 = MatrixOp(C2)
    opVA = VCAT(opA1, opA2)
    opVB = VCAT(opB1, opB2)
    opVC = VCAT(opC1, opC2)
    opS = Sum(opVA, opVB, opVC)
    x = randn(n)
    y1 = test_op(opS, x, ArrayPartition(randn(m1), randn(m2)), verb)
    y2 = ArrayPartition(A1 * x + B1 * x + C1 * x, A2 * x + B2 * x + C2 * x)

    @test norm(y1 - y2) .<= 1e-12

    # test Scale of DCAT

    m1, n1 = 4, 7
    m2, n2 = 3, 5
    A1 = randn(m1, n1)
    A2 = randn(m2, n2)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opD = DCAT(opA1, opA2)
    coeff = randn()
    opS = Scale(coeff, opD)
    x1 = randn(n1)
    x2 = randn(n2)
    y = test_op(opS, ArrayPartition(x1, x2), ArrayPartition(randn(m1), randn(m2)), verb)
    z = ArrayPartition(coeff * A1 * x1, coeff * A2 * x2)

    @test norm(y - z) <= 1e-12

    # test Scale of VCAT

    m1, m2, n = 4, 3, 7
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opV = VCAT(opA1, opA2)
    coeff = randn()
    opS = Scale(coeff, opV)
    x = randn(n)
    y = test_op(opS, x, ArrayPartition(randn(m1), randn(m2)), verb)
    z = ArrayPartition(coeff * A1 * x, coeff * A2 * x)

    @test norm(y - z) <= 1e-12

    # test Scale of HCAT

    m, n1, n2 = 4, 3, 7
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opH = HCAT(opA1, opA2)
    coeff = randn()
    opS = Scale(coeff, opH)
    x1 = randn(n1)
    x2 = randn(n2)
    y = test_op(opS, ArrayPartition(x1, x2), randn(m), verb)
    z = coeff * (A1 * x1 + A2 * x2)

    @test norm(y - z) <= 1e-12

    # test DCAT of HCATs

    m1, m2, n1, n2, l1, l2, l3 = 2, 3, 4, 5, 6, 7, 8
    A1 = randn(m1, n1)
    A2 = randn(m1, n2)
    B1 = randn(m2, n1)
    B2 = randn(m2, n2)
    B3 = randn(m2, n2)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opH1 = HCAT(opA1, opA2)
    opB1 = MatrixOp(B1)
    opB2 = MatrixOp(B2)
    opB3 = MatrixOp(B3)
    opH2 = HCAT(opB1, opB2, opB3)

    op = DCAT(opA1, opH2)
    x = ArrayPartition(randn.(size(op, 2))...)
    y0 = ArrayPartition(randn.(size(op, 1))...)
    y = test_op(op, x, y0, verb)

    op = DCAT(opH1, opH2)
    x = ArrayPartition(randn.(size(op, 2))...)
    y0 = ArrayPartition(randn.(size(op, 1))...)
    y = test_op(op, x, y0, verb)

    p = randperm(ndoms(op, 2))
    y2 = op[p] * ArrayPartition(x.x[p]...)

    @test norm(y - y2) <= 1e-8

    # test Scale of Sum

    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opS = Sum(opA1, opA2)
    coeff = pi
    opSS = Scale(coeff, opS)
    x1 = randn(n)
    y1 = test_op(opSS, x1, randn(m), verb)
    y2 = coeff * (A1 * x1 + A2 * x1)
    @test norm(y1 - y2) <= 1e-12

    # test Scale of Compose

    m1, m2, m3 = 4, 7, 3
    A1 = randn(m2, m1)
    A2 = randn(m3, m2)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)

    coeff = pi
    opC = Compose(opA2, opA1)
    opS = Scale(coeff, opC)
    x = randn(m1)
    y1 = test_op(opS, x, randn(m3), verb)
    y2 = coeff * (A2 * A1 * x)
    @test all(norm.(y1 .- y2) .<= 1e-12)


    # Testing nonlinear HCAT of VCAT
    n, m1, m2, m3 = 4, 3, 2, 7
    x1 = randn(m1)
    x2 = randn(m2)
    x3 = randn(m3)
    x = ArrayPartition(x1, x2, x3)
    r = ArrayPartition(randn(n), randn(m1))
    A1 = randn(n, m1)
    A2 = randn(n, m2)
    A3 = randn(n, m3)
    B1 = Sigmoid(Float64, (m1,), 2)
    B2 = randn(m1, m2)
    B3 = randn(m1, m3)
    op1 = VCAT(MatrixOp(A1), B1)
    op2 = VCAT(MatrixOp(A2), MatrixOp(B2))
    op3 = VCAT(MatrixOp(A3), MatrixOp(B3))
    op = HCAT(op1, op2, op3)

    y, grad = test_NLop(op, x, r, verb)

    Y = ArrayPartition(A1 * x1 + A2 * x2 + A3 * x3, B1 * x1 + B2 * x2 + B3 * x3)
    @test norm(Y - y) < 1e-8

    # Testing nonlinear VCAT of HCAT
    m1, m2, m3, n1, n2 = 3, 4, 5, 6, 7
    x1 = randn(m1)
    x2 = randn(n1)
    x3 = randn(m3)
    x = ArrayPartition(x1, x2, x3)
    r = ArrayPartition(randn(n1), randn(n2))
    A1 = randn(n1, m1)
    B1 = Sigmoid(Float64, (n1,), 2)
    C1 = randn(n1, m3)
    A2 = randn(n2, m1)
    B2 = randn(n2, n1)
    C2 = randn(n2, m3)
    x = ArrayPartition(x1, x2, x3)
    op1 = HCAT(MatrixOp(A1), B1, MatrixOp(C1))
    op2 = HCAT(MatrixOp(A2), MatrixOp(B2), MatrixOp(C2))
    op = VCAT(op1, op2)

    y, grad = test_NLop(op, x, r, verb)

    Y = ArrayPartition(A1 * x1 + B1 * x2 + C1 * x3, A2 * x1 + B2 * x2 + C2 * x3)
    @test norm(Y - y) < 1e-8

    # Testing nonlinear AffineAdd and Compose
    n = 10
    d1 = randn(n)
    d2 = randn(n)
    T = Compose(AffineAdd(Sin(n), d2), AffineAdd(Eye(n), d1))

    r = randn(n)
    x = randn(size(T, 2))
    y, grad = test_NLop(T, x, r, verb)
    @test norm(y - (sin.(x + d1) + d2)) < 1e-8

    n = 10
    d1 = randn(n)
    d2 = randn(n)
    d3 = pi
    T = Compose(
        AffineAdd(Sin(n), d3), Compose(AffineAdd(Exp(n), d2, false), AffineAdd(Eye(n), d1))
    )

    r = randn(n)
    x = randn(size(T, 2))
    y, grad = test_NLop(T, x, r, verb)
    @test norm(y - (sin.(exp.(x + d1) - d2) .+ d3)) < 1e-8
end
