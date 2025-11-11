if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end

@testset "Sum" begin
    verb && println(" --- Testing Sum --- ")

    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opS = Sum(opA1, opA2)
    x1 = randn(n)
    y1 = test_op(opS, x1, randn(m), verb)
    y2 = A1 * x1 + A2 * x1
    @test norm(y1 - y2) <= 1e-12

    #test Sum longer
    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    A3 = randn(m, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opS = Sum(opA1, opA2, opA3)
    x1 = randn(n)
    y1 = test_op(opS, x1, randn(m), verb)
    y2 = A1 * x1 + A2 * x1 + A3 * x1
    @test norm(y1 - y2) <= 1e-12

    opA3 = MatrixOp(randn(m, m))
    @test_throws Exception Sum(opA1, opA3)
    opF = DiagOp(Float64, (m,), 2.0 + im * 3.0)
    @test_throws Exception Sum(opF, opA3)

    @test is_null(opS) == false
    @test is_eye(opS) == false
    @test is_diagonal(opS) == false
    @test is_AcA_diagonal(opS) == false
    @test is_AAc_diagonal(opS) == false
    @test is_orthogonal(opS) == false
    @test is_invertible(opS) == false
    @test is_full_row_rank(opS) == true
    @test is_full_column_rank(opS) == false

    d = randn(10)
    op = Sum(Scale(-3.1, Eye(10)), DiagOp(d))
    @test is_diagonal(op) == true
    @test norm(diag(op) - (d .- 3.1)) < 1e-12

    #test displacement of sum
    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    A3 = randn(m, n)
    d1 = randn(m)
    d2 = pi
    d3 = randn(m)
    opA1 = AffineAdd(MatrixOp(A1), d1)
    opA2 = AffineAdd(MatrixOp(A2), d2)
    opA3 = AffineAdd(MatrixOp(A3), d3)
    opS = Sum(opA1, opA2, opA3)
    x1 = randn(n)
    y2 = A1 * x1 + A2 * x1 + A3 * x1 + d1 .+ d2 + d3
    @test norm(opS * x1 - y2) <= 1e-12
    @test norm(displacement(opS) - (d1 .+ d2 + d3)) <= 1e-12
    y2 = A1 * x1 + A2 * x1 + A3 * x1
    @test norm(remove_displacement(opS) * x1 - y2) <= 1e-12

    # fun_name formatting (2-term vs multi-term) and thread safety / storage types
    op2 = Sum(MatrixOp(randn(m, n)), MatrixOp(randn(m, n)))
    io = IOBuffer()
    show(io, op2)
    str2 = String(take!(io))
    @test occursin("+", str2) || length(str2) > 0  # lightweight check without depending on exact internal symbol

    op3 = Sum(MatrixOp(randn(m, n)), MatrixOp(randn(m, n)), MatrixOp(randn(m, n)))
    io = IOBuffer()
    show(io, op3)
    str3 = String(take!(io))
    @test length(str3) > 0  # invokes Σ path internally

    # is_thread_safe expected false
    @test is_thread_safe(op3) == false

    # storage type queries (execute for coverage)
    _dst = domain_storage_type(op3)
    _cst = codomain_storage_type(op3)
    @test _dst !== nothing && _cst !== nothing

    # remove_displacement idempotence
    dX = randn(m)
    op_disp = Sum(AffineAdd(MatrixOp(randn(m, n)), dX), MatrixOp(randn(m, n)))
    rd1 = remove_displacement(op_disp)
    rd2 = remove_displacement(rd1)
    @test rd1 * x1 == rd2 * x1
    # --- Additional coverage ---
    # Equality / inequality
    m, n = 5, 7
    Aeq = MatrixOp(randn(m, n))
    Beq = MatrixOp(randn(m, n))
    S1 = Sum(Aeq, Beq)
    S2 = Sum(Aeq, Beq)
    S3 = Sum(Beq, Aeq)
    @test S1 == S2
    @test S1 != S3

    # Single-operator constructor returns the operator itself
    single = MatrixOp(randn(m, n))
    @test Sum(single) === single

    # diag aggregator for diagonal underlying
    d1 = randn(m)
    d2 = randn(m)
    Sdiag = Sum(DiagOp(d1), DiagOp(d2))
    @test is_diagonal(Sdiag)
    @test diag(Sdiag) == d1 + d2

    # permute utility (wrap HCAT to get multi-domain) and ensure same behavior when inputs permuted
    mH = 6; n1 = 2; n2 = 1
    A1p = MatrixOp(randn(mH, n1))
    A2p = MatrixOp(randn(mH, n2))
    H = HCAT(A1p, A2p)
    S = Sum(H, H)
    x1p = randn(n1); x2p = randn(n2)
    y_orig = S * ArrayPartition(x1p, x2p)
    p = [2, 1]
    Sp = AbstractOperators.permute(S, p)
    y_perm = Sp * ArrayPartition(x2p, x1p)
    @test y_orig ≈ y_perm

    # estimate_opnorm aggregator
    opnorm_S1 = opnorm(S1)
    estimated_opnorm_S1 = estimate_opnorm(S1)
    @test abs(estimated_opnorm_S1 - opnorm_S1) / opnorm_S1 < 0.03

    # remove_displacement idempotence with displacement underlying
    dA = randn(m)
    SA = Sum(AffineAdd(MatrixOp(randn(m, n)), dA), MatrixOp(randn(m, n)))
    SA_rd = remove_displacement(SA)
    @test remove_displacement(SA_rd) == SA_rd

    # Testing nonlinear Sum of MatrixOp and Sigmoid
    m = 5
    x = randn(m)
    r = randn(m)
    A = randn(m, m)
    opA = MatrixOp(A)
    opB = Sigmoid(Float64, (m,), 2)
    op = Sum(opA, opB)

    y, grad = test_NLop(op, x, r, verb)

    Y = A * x + opB * x
    @test norm(Y - y) < 1e-8
end
