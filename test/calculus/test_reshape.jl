if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end

@testset "Reshape" begin
    verb && println(" --- Testing Reshape --- ")

    m, n = 8, 4
    dim_out = (2, 2, 2)
    A1 = randn(m, n)
    opA1 = MatrixOp(A1)
    opR = Reshape(opA1, dim_out)
    opR = Reshape(opA1, dim_out...)
    x1 = randn(n)
    y1 = test_op(opR, x1, randn(dim_out), verb)
    y2 = reshape(A1 * x1, dim_out)
    @test norm(y1 - y2) <= 1e-12

    @test_throws Exception Reshape(opA1, (2, 2, 1))

    @test is_null(opR) == is_null(opA1)
    @test is_eye(opR) == is_eye(opA1)
    @test is_diagonal(opR) == is_diagonal(opA1)
    @test is_AcA_diagonal(opR) == is_AcA_diagonal(opA1)
    @test is_AAc_diagonal(opR) == is_AAc_diagonal(opA1)
    @test is_orthogonal(opR) == is_orthogonal(opA1)
    @test is_invertible(opR) == is_invertible(opA1)
    @test is_full_row_rank(opR) == is_full_row_rank(opA1)
    @test is_full_column_rank(opR) == is_full_column_rank(opA1)

    # testing displacement
    m, n = 8, 4
    dim_out = (2, 2, 2)
    A1 = randn(m, n)
    d1 = randn(m)
    opA1 = AffineAdd(MatrixOp(A1), d1)
    opR = Reshape(opA1, dim_out)
    x1 = randn(n)
    y1 = opR * x1
    y2 = reshape(A1 * x1 + d1, dim_out)
    @test norm(y1 - y2) <= 1e-12
    y1 = remove_displacement(opR) * x1
    y2 = reshape(A1 * x1, dim_out)
    @test norm(y1 - y2) <= 1e-12

    # fun_name / storage / thread safety / idempotent remove
    io = IOBuffer(); show(io, opR); s = String(take!(io))
    @test length(s) > 0
    _dst = domain_storage_type(opR)
    _cst = codomain_storage_type(opR)
    @test _dst !== nothing && _cst !== nothing
    @test is_thread_safe(opR) == is_thread_safe(opA1)
    rd1 = remove_displacement(opR)
    rd2 = remove_displacement(rd1)
    @test rd1 * x1 == rd2 * x1

    #######################
    ## test Scale   #######
    #######################

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

    # special case, Scale of DiagOp gets a DiagOp
    d = randn(10)
    op = Scale(3, DiagOp(d))
    @test typeof(op) <: DiagOp
    @test norm(diag(op) - 3 .* d) < 1e-12

    # Scale with imaginary coeff gives error
    m, n = 8, 4
    coeff = im
    A1 = randn(m, n)
    opA1 = MatrixOp(A1)
    @test_throws ErrorException Scale(coeff, opA1)

    ## testing displacement
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

    # Equality / inequality
    m, n = 8, 4
    dim_out = (2, 2, 2)
    Aeq = MatrixOp(randn(m, n))
    R1 = Reshape(Aeq, dim_out)
    R2 = Reshape(Aeq, dim_out)
    R3 = Reshape(Aeq, (4, 2))  # different shape with same product
    @test R1 == R2
    @test R1 != R3

    # Adjoint mapping explicit check
    x = randn(n)
    y = randn(dim_out)
    Rf = Reshape(Aeq, dim_out)
    lhs = Rf' * y
    rhs = Aeq' * vec(y)
    @test lhs ≈ rhs

    # fun_name via show should start with paragraph symbol
    io = IOBuffer(); show(io, Rf); sR = String(take!(io))
    @test occursin("¶", sR)

    # has_optimized_normalop + get_normal_op passthrough (using GetIndex which has optimized normal)
    nGI = 10; kGI = 7
    GI = GetIndex(Float64, (nGI,), (1:kGI,))  # sliced operator returning size kGI
    RG = Reshape(GI, (kGI, 1))
    normal_RG = AbstractOperators.get_normal_op(RG)
    @test normal_RG !== nothing

    # Slicing pass-through (is_sliced, expr, mask); avoid remove_slicing due to size mismatch
    @test is_sliced(RG) == true
    exprRG = AbstractOperators.get_slicing_expr(RG)
    @test exprRG == (1:kGI,)
    maskRG = AbstractOperators.get_slicing_mask(GI)
    @test sum(maskRG) == kGI

    # permute domain ordering (wrap HCAT to get multi-domain) and ensure same behavior when inputs permuted
    mH = 6; n1 = 3; n2 = 5
    A1p = MatrixOp(randn(mH, n1))
    A2p = MatrixOp(randn(mH, n2))
    H = HCAT(A1p, A2p)
    RH = Reshape(H, (2, 3))  # 6 = 2*3
    x1p = randn(n1); x2p = randn(n2)
    y_orig = RH * ArrayPartition(x1p, x2p)
    p = [2, 1]
    RHp = AbstractOperators.permute(RH, p)
    y_perm = RHp * ArrayPartition(x2p, x1p)
    @test y_orig ≈ y_perm

    # opnorm passthrough
    @test opnorm(R1) == opnorm(Aeq)

    # remove_displacement idempotence with displacement underlying
    dA = randn(m)
    RA = Reshape(AffineAdd(MatrixOp(randn(m, n)), dA), dim_out)
    RA_rd = remove_displacement(RA)
    @test remove_displacement(RA_rd) == RA_rd

    # Testing nonlinear Reshape
    n = 4
    x = randn(n)
    r = randn(n)
    opS = Sigmoid(Float64, (n,), 2)
    op = Reshape(opS, 2, 2)

    y, grad = test_NLop(op, x, r, verb)

    Y = reshape(opS * x, 2, 2)
    @test norm(Y - y) < 1e-8
end
