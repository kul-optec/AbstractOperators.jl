if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end

@testset "VCAT" begin
    verb && println(" --- Testing VCAT --- ")

    m1, m2, n = 4, 7, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opV = VCAT(opA1, opA2)
    x1 = randn(n)
    y1 = test_op(opV, x1, ArrayPartition(randn(m1), randn(m2)), verb)
    y2 = ArrayPartition(A1 * x1, A2 * x1)
    @test norm(y1 - y2) .<= 1e-12

    m1, n = 4, 5
    A1 = randn(m1, n) + im * randn(m1, n)
    opA1 = MatrixOp(A1)
    opA2 = DiagOp(Float64, (n,), 2.0 + im * 3.0)'
    opV = VCAT(opA1, opA2)
    x1 = randn(n) + im * randn(n)
    y1 = test_op(opV, x1, ArrayPartition(randn(m1) + im * randn(m1), randn(n)), verb)
    y2 = ArrayPartition(A1 * x1, opA2 * x1)
    @test norm(y1 - y2) .<= 1e-12

    #test VCAT longer
    m1, m2, m3, n = 4, 7, 3, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    A3 = randn(m3, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opV = VCAT(opA1, opA2, opA3)
    x1 = randn(n)
    y1 = test_op(opV, x1, ArrayPartition(randn(m1), randn(m2), randn(m3)), verb)
    y2 = ArrayPartition(A1 * x1, A2 * x1, A3 * x1)
    @test norm(y1 - y2) .<= 1e-12

    #test VCAT of VCAT
    opVV = VCAT(opV, opA3)
    y1 = test_op(opVV, x1, ArrayPartition(randn(m1), randn(m2), randn(m3), randn(m3)), verb)
    y2 = ArrayPartition(A1 * x1, A2 * x1, A3 * x1, A3 * x1)
    @test norm(y1 .- y2) <= 1e-12

    opVV = VCAT(opA1, opV, opA3)
    y1 = test_op(
        opVV, x1, ArrayPartition(randn(m1), randn(m1), randn(m2), randn(m3), randn(m3)), verb
    )
    y2 = ArrayPartition(A1 * x1, A1 * x1, A2 * x1, A3 * x1, A3 * x1)
    @test norm(y1 - y2) <= 1e-12

    opA3 = MatrixOp(randn(m1, m1))
    @test_throws Exception VCAT(opA1, opA2, opA3)
    opD = DiagOp(Float64, (m1,), 2.0 + im * 3.0)
    @test_throws Exception VCAT(opA1, opD, opA2)

    ###properties
    m1, m2, m3, n = 4, 7, 3, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    A3 = randn(m3, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    op = VCAT(opA1, opA2, opA3)
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == false
    @test is_full_column_rank(op) == true

    d = randn(n) .+ im .* randn(n)
    op = VCAT(DiagOp(d), Eye(ComplexF64, n))
    @test is_AcA_diagonal(op) == true
    @test diag_AcA(op) == d .* conj(d) .+ 1

    ##test displacement
    m1, m2, n = 4, 7, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    d1 = randn(m1)
    d2 = randn(m2)
    opV = VCAT(AffineAdd(opA1, d1), AffineAdd(opA2, d2))
    x1 = randn(n)
    y1 = opV * x1
    y2 = ArrayPartition(A1 * x1 + d1, A2 * x1 + d2)
    @test norm(y1 - y2) <= 1e-12
    y1 = remove_displacement(opV) * x1
    y2 = ArrayPartition(A1 * x1, A2 * x1)
    @test norm(y1 - y2) <= 1e-12

    m1, m2, n = 4, 7, 5
    A1 = MatrixOp(randn(m1, n))
    A2 = MatrixOp(randn(m2, n))
    op = VCAT(A1, A2)
    _ds = domain_storage_type(op)
    _cs = codomain_storage_type(op)
    @test _ds !== nothing
    @test _cs !== nothing
    @test is_thread_safe(op) == false

    @test remove_displacement(op) == op

    d1 = randn(m1)
    d2 = randn(m2)
    opd = VCAT(AffineAdd(A1, d1), AffineAdd(A2, d2))
    opd_removed = remove_displacement(opd)
    @test remove_displacement(opd_removed) == opd_removed

    # equality / inequality
    Aeq1 = MatrixOp(randn(3, 4))
    Aeq2 = MatrixOp(randn(5, 4))
    Veq1 = VCAT(Aeq1, Aeq2)
    Veq2 = VCAT(Aeq1, Aeq2)
    Veq3 = VCAT(Aeq2, Aeq1)
    @test Veq1 == Veq2
    @test Veq1 != Veq3

    # single operator constructor returns operator itself
    single = MatrixOp(randn(6, 6))
    @test VCAT(single) === single

    # show output (fun_name indirect) for 2-op VCAT
    io = IOBuffer(); show(io, VCAT(Eye(4), Eye(4))); s = String(take!(io))
    @test occursin("VCAT", s) || occursin("[", s)

    # sliced helpers using GetIndex
    n = 10
    g1 = GetIndex(Float64, (n,), (1:5,))
    g2 = GetIndex(Float64, (n,), (6:10,))
    Vs = VCAT(g1, g2)
    @test is_sliced(Vs) == true
    exprs = AbstractOperators.get_slicing_expr(Vs)
    @test length(exprs) == 2
    @test exprs[1] == (1:5,)
    @test exprs[2] == (6:10,)
    Vs_removed = AbstractOperators.remove_slicing(Vs)
    @test is_sliced(Vs_removed) == false

    # diag_AAc aggregator and is_AAc_diagonal true case
    Veye = VCAT(Eye(5), Eye(5))
    @test is_AAc_diagonal(Veye) == true
    @test diag_AAc(Veye) == (1, 1)

    # VCAT of nonlinear operator
    n, m = 4, 3
    x = randn(m)
    r = ArrayPartition(randn(n), randn(m))
    A = randn(n, m)
    B = Sigmoid(Float64, (m,), 2)
    op = VCAT(MatrixOp(A), B)

    y, grad = test_NLop(op, x, r, verb)

    Y = ArrayPartition(A * x, B * x)
    @test norm(Y - y) < 1e-8
end
