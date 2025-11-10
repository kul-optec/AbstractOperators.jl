if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end

@testset "Scale" begin
    verb && println(" --- Testing Scale --- ")

    m, n = 8, 4
    coeff = pi
    A1 = randn(m, n)
    opA1 = MatrixOp(A1)
    opS = Scale(coeff, opA1)
    x1 = randn(n)
    y1 = test_op(opS, x1, randn(m), verb)
    y2 = coeff * A1 * x1
    @test norm(y1 - y2) <= 1e-12

    coeff2 = 3
    opS2 = Scale(coeff2, opS)
    y1 = test_op(opS2, x1, randn(m), verb)
    y2 = coeff2 * coeff * A1 * x1
    @test norm(y1 - y2) <= 1e-12

    opF = FiniteDiff((m,))
    opS = Scale(coeff, opF)
    x1 = randn(m)
    y1 = test_op(opS, x1, diff(randn(m)), verb)
    y2 = coeff * (diff(x1))
    @test norm(y1 - y2) <= 1e-12

    opS = Scale(coeff, opA1)
    @test is_null(opS) == is_null(opA1)
    @test is_eye(opS) == is_eye(opA1)
    @test is_diagonal(opS) == is_diagonal(opA1)
    @test is_AcA_diagonal(opS) == is_AcA_diagonal(opA1)
    @test is_AAc_diagonal(opS) == is_AAc_diagonal(opA1)
    @test is_orthogonal(opS) == is_orthogonal(opA1)
    @test is_invertible(opS) == is_invertible(opA1)
    @test is_full_row_rank(opS) == is_full_row_rank(opA1)
    @test is_full_column_rank(opS) == is_full_column_rank(opA1)

    op = Scale(-4.0, GetIndex((10,), 1:5))
    @test is_AAc_diagonal(op) == true
    @test diag_AAc(op) == 16

    op = Scale(-4.0, ZeroPad((10,), 20))
    @test is_AcA_diagonal(op) == true
    @test diag_AcA(op) == 16

    d = randn(10)
    op = Scale(3, DiagOp(d))
    @test typeof(op) <: DiagOp
    @test norm(diag(op) - 3 .* d) < 1e-12

    m, n = 8, 4
    coeff = im
    A1 = randn(m, n)
    opA1 = MatrixOp(A1)
    @test_throws ErrorException Scale(coeff, opA1)

    m, n = 8, 4
    coeff = pi
    A1 = randn(m, n)
    d1 = randn(m)
    opA1 = AffineAdd(MatrixOp(A1), d1)
    opS = Scale(coeff, opA1)
    x1 = randn(n)
    y1 = opS * x1
    y2 = coeff * (A1 * x1 + d1)
    @test norm(y1 - y2) <= 1e-12
    y1 = remove_displacement(opS) * x1
    y2 = coeff * (A1 * x1)
    @test norm(y1 - y2) <= 1e-12

    # Edge cases
    # 1. coeff == 1 path should return the original operator (early return)
    op = MatrixOp(randn(5,5))
    s = Scale(1, op)
    @test s === op  # identity return

    # 2. Real codomain, complex coefficient should throw (error branch in Scale(coeff, coeff_conj, L))
    real_op = MatrixOp(randn(4,4))  # domain/codomain both Real
    @test_throws ErrorException Scale(1 + 2im, real_op)

    # 3. Scale of a Scale promotes / multiplies coefficients
    base = FiniteDiff((5,))
    s1 = Scale(2.0, base)
    s2 = Scale(3.0, s1)  # invokes Scale(coeff::Number, L::Scale)
    @test s2.coeff ≈ 6.0
    @test s2.A === base

    # 4. == comparison
    @test (Scale(4.0, base) == Scale(4.0, base))
    @test (Scale(4.0, base) != Scale(5.0, base))

    # 5. is_null delegates
    z = Zeros(Float64, (5,), Float64, (5,))
    sz = Scale(10.0, z)
    @test is_null(sz)

    # 6. diag / diag_AcA / diag_AAc for diagonal underlying operator
    d = randn(6)
    dop = DiagOp(d)
    sdiag = Scale(2.0, dop)
    @test diag(sdiag) == 2.0 .* diag(dop)
    @test diag_AcA(sdiag) == (2.0)^2 * diag_AcA(dop)
    @test diag_AAc(sdiag) == (2.0)^2 * diag_AAc(dop)

    # Equality / inequality
    Aeq = FiniteDiff((6,))
    S1 = Scale(2.0, Aeq)
    S2 = Scale(2.0, Aeq)
    S3 = Scale(3.0, Aeq)
    @test S1 == S2
    @test S1 != S3

    # fun_name via show should start with α
    io = IOBuffer(); show(io, S1); sS = String(take!(io))
    @test occursin("α", sS)

    # has_optimized_normalop + get_normal_op passthrough (using GetIndex which has optimized normal)
    nGI = 8; kGI = 5
    GI = GetIndex(Float64, (nGI,), (1:kGI,))
    SG = Scale(2.0, GI)
    @test AbstractOperators.has_optimized_normalop(SG) == true
    normal_SG = AbstractOperators.get_normal_op(SG)
    @test normal_SG !== nothing

    # Slicing pass-through (is_sliced, expr, mask); avoid remove_slicing due to type
    @test is_sliced(SG) == is_sliced(GI)
    exprSG = AbstractOperators.get_slicing_expr(SG)
    @test exprSG == (1:kGI,)
    maskSG = AbstractOperators.get_slicing_mask(GI)
    @test sum(maskSG) == kGI

    # permute domain ordering (wrap HCAT to get multi-domain) and ensure same behavior when inputs permuted
    mH = 6; n1 = 2; n2 = 1
    A1p = MatrixOp(randn(mH, n1))
    A2p = MatrixOp(randn(mH, n2))
    H = HCAT(A1p, A2p)
    SH = Scale(2.0, H)
    x1p = randn(n1); x2p = randn(n2)
    y_orig = SH * ArrayPartition(x1p, x2p)
    p = [2, 1]
    SHp = AbstractOperators.permute(SH, p)
    y_perm = SHp * ArrayPartition(x2p, x1p)
    @test y_orig ≈ y_perm

    # opnorm and estimate_opnorm passthrough
    opnorm_S = opnorm(S1)
    @test opnorm_S ≈ abs(S1.coeff) * opnorm(Aeq)
    @test abs(opnorm_S - estimate_opnorm(S1)) / opnorm_S < 0.02

    # remove_displacement idempotence with displacement underlying
    dA = randn(m)
    SA = Scale(2.0, AffineAdd(MatrixOp(randn(m, n)), dA))
    SA_rd = remove_displacement(SA)
    @test remove_displacement(SA_rd) == SA_rd

    # Non-linear operators
    m = 3
    x = randn(m)
    r = randn(m)
    A = Sigmoid(Float64, (m,), 2)
    op = 30 * A

    y, grad = test_NLop(op, x, r, verb)

    Y = 30 * (A * x)
    @test norm(Y - y) < 1e-8

    m = 3
    x = randn(m)
    r = randn(m)
    A = Pow(Float64, (m,), 2)
    op = -A

    y, grad = test_NLop(op, x, r, verb)

    Y = -A * x
    @test norm(Y - y) < 1e-8
end
