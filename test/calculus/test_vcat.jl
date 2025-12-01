if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)

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

    # Extended coverage tests

    # Test: domain_type mismatch error
    A_real = MatrixOp(randn(3, 4))
    A_complex = MatrixOp(randn(ComplexF64, 3, 4))
    @test_throws Exception VCAT(A_real, A_complex)

    # Test: VCAT with stacked operators (operators with multiple codomains)
    verb && println("Testing stacked operators with multiple codomains")
    n = 5
    m1, m2 = 3, 4
    A1 = MatrixOp(randn(m1, n))
    A2 = MatrixOp(randn(m2, n))
    V1 = VCAT(A1, A2)  # V1 has multiple codomains
    B1 = MatrixOp(randn(m1, n))
    B2 = MatrixOp(randn(m2, n))
    V2 = VCAT(B1, B2)  # V2 has multiple codomains
    VV = VCAT(V1, V2)  # VCAT of VCATs - creates stacked structure
    x = randn(n)
    y = VV * x
    @test y isa ArrayPartition
    @test length(y.x) == 4  # Should have 4 outputs
    y_test = ArrayPartition(randn(m1), randn(m2), randn(m1), randn(m2))
    x_adj = VV' * y_test
    @test length(x_adj) == n

    # Test: VCAT of HCAT with Zeros (remove_slicing edge case)
    verb && println("Testing VCAT of HCAT with Zeros")
    n, m = 5, 6
    A1 = MatrixOp(randn(n, m))
    Z1 = Zeros(Float64, (m,), Float64, (n,))
    H1 = HCAT(A1, Z1)
    A2 = MatrixOp(randn(n, m))
    A3 = MatrixOp(randn(n, m))
    H2 = HCAT(A2, A3)
    V = VCAT(H1, H2)
    G1 = GetIndex(Float64, (2 * m,), (1:m,))
    G2 = GetIndex(Float64, (2 * m,), ((m + 1):(2 * m),))
    V_sliced = V * VCAT(G1, G2)
    @test is_sliced(V_sliced)
    V_removed = AbstractOperators.remove_slicing(V_sliced)
    @test V == V_removed  # Should be equal after removing slicing

    H1 = HCAT(A1 * G1, Z1 * G2)
    H2 = HCAT(A2 * G1, A3 * G2)
    V2 = VCAT(H1, H2)
    @test is_sliced(V2)
    V2_removed = AbstractOperators.remove_slicing(V2)
    @test V == V2_removed  # Should be equal after removing slicing

    # Test: permute function
    verb && println("Testing permute function")
    m1, m2, n = 3, 4, 5
    A1 = MatrixOp(randn(m1, n))
    A2 = MatrixOp(randn(m2, n))
    V = VCAT(A1, A2)
    p = [2, 1]
    V_perm = AbstractOperators.permute(V, p)
    @test size(V, 1) == size(V_perm, 1)[p]

    # Test: Deeply nested VCAT structures
    verb && println("Testing deeply nested VCAT")
    n = 4
    A1 = Eye(n)
    A2 = DiagOp(randn(n))
    A3 = MatrixOp(randn(n, n))
    V1 = VCAT(A1, A2)
    V2 = VCAT(V1, A3)  # VCAT of VCAT
    V3 = VCAT(A1, V2, A2)  # Mix of regular and VCAT operators
    x = randn(n)
    y = V3 * x
    @test y isa ArrayPartition
    @test length(y.x) == 5  # A1, A1, A2, A3, A2
    y_adj = ArrayPartition([randn(n) for _ in 1:5]...)
    x_back = V3' * y_adj
    @test length(x_back) == n

    # Test: fun_name for VCAT with > 2 operators
    A1 = Eye(3)
    A2 = Eye(3)
    A3 = Eye(3)
    V2 = VCAT(A1, A2)
    V3 = VCAT(A1, A2, A3)
    name2 = AbstractOperators.fun_name(V2)
    name3 = AbstractOperators.fun_name(V3)
    @test occursin("[", name2) || occursin("]", name2)
    @test name3 == "VCAT"

    # Test: VCAT equality with different internal structures (flattened vs nested)
    A1 = Eye(4)
    A2 = DiagOp(randn(4))
    A3 = MatrixOp(randn(4, 4))
    V1 = VCAT(A1, A2, A3)
    V_temp = VCAT(A1, A2)
    V2 = VCAT(V_temp, A3)  # Should flatten
    @test V1 == V2

    # Test: is_full_column_rank with mixed operators
    n, m = 5, 3
    A1 = MatrixOp(randn(n, m))
    A2 = Zeros(Float64, (m,), Float64, (n,))
    V = VCAT(A1, A2)
    @test is_full_column_rank(V) isa Bool

    # Test: Adjoint with complex nested structure
    verb && println("Testing adjoint with complex nested structure")
    n = 4
    m1, m2 = 3, 3
    A1 = MatrixOp(randn(m1, n))
    A2 = MatrixOp(randn(m2, n))
    V1 = 2 * VCAT(A1, A2)
    B1 = MatrixOp(randn(m1, n))
    V2 = VCAT(V1, B1)
    y = ArrayPartition(randn(m1), randn(m2), randn(m1))
    x = V2' * y
    @test length(x) == n
    @test x isa AbstractArray
    y2 = V2 * x
    @test size.(y.x) == size.(y2.x)

    V2 = VCAT(B1, V1)
    y = ArrayPartition(randn(m1), randn(m1), randn(m2))
    x = V2' * y
    @test length(x) == n
    @test x isa AbstractArray
    y2 = V2 * x
    @test size.(y.x) == size.(y2.x)
end
