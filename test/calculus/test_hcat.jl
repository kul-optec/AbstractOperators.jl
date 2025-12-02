if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)

@testset "HCAT" begin
    verb && println(" --- Testing HCAT --- ")

    m, n1, n2 = 4, 7, 5
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opH = HCAT(opA1, opA2)
    x1 = randn(n1)
    x2 = randn(n2)
    y1 = test_op(opH, ArrayPartition(x1, x2), randn(m), verb)
    y2 = A1 * x1 + A2 * x2
    @test norm(y1 - y2) <= 1e-12

    # permutation 
    p = [2; 1]
    opHp = opH[p]
    y1 = test_op(opHp, ArrayPartition(x2, x1), randn(m), verb)
    @test norm(y1 - y2) <= 1e-12

    # test HCAT longer

    m, n1, n2, n3 = 4, 7, 5, 6
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    A3 = randn(m, n3)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opH = HCAT(opA1, opA2, opA3)
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = test_op(opH, ArrayPartition(x1, x2, x3), randn(m), verb)
    y2 = A1 * x1 + A2 * x2 + A3 * x3
    @test norm(y1 - y2) <= 1e-12

    # test HCAT of HCAT
    opHH = HCAT(opH, opA2, opA3)
    y1 = test_op(opHH, ArrayPartition(x1, x2, x3, x2, x3), randn(m), verb)
    y2 = A1 * x1 + A2 * x2 + A3 * x3 + A2 * x2 + A3 * x3
    @test norm(y1 - y2) <= 1e-12

    opHH = HCAT(opH, opH, opA3)
    x = ArrayPartition(x1, x2, x3, x1, x2, x3, x3)
    y1 = test_op(opHH, x, randn(m), verb)
    y2 = A1 * x1 + A2 * x2 + A3 * x3 + A1 * x1 + A2 * x2 + A3 * x3 + A3 * x3
    @test norm(y1 - y2) <= 1e-12

    opA3 = MatrixOp(randn(n1, n1))
    @test_throws Exception HCAT(opA1, opA2, opA3)
    opF = MatrixOp(randn(ComplexF64, m, m))
    @test_throws Exception HCAT(opA1, opF, opA2)

    # test utilities

    # permutation
    p = randperm(ndoms(opHH, 2))
    opHP = AbstractOperators.permute(opHH, p)

    xp = ArrayPartition(x.x[p]...)

    y1 = test_op(opHP, xp, randn(m), verb)

    pp = randperm(ndoms(opHH, 2))
    opHPP = AbstractOperators.permute(opHH, pp)
    xpp = ArrayPartition(x.x[pp]...)
    y1 = test_op(opHPP, xpp, randn(m), verb)

    #properties
    m, n1, n2, n3 = 4, 7, 5, 6
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    A3 = randn(m, n3)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    op = HCAT(opA1, opA2, opA3)
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == false

    d1 = randn(n1) .+ im .* randn(n1)
    d2 = randn(n1) .+ im .* randn(n1)
    op = HCAT(DiagOp(d1), DiagOp(d2))
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == false

    @test diag_AAc(op) == d1 .* conj(d1) .+ d2 .* conj(d2)

    y1 = randn(n1) .+ im .* randn(n1)
    @test norm(op * (op' * y1) .- diag_AAc(op) .* y1) < 1e-12

    #test displacement

    m, n1, n2 = 4, 7, 5
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    d1 = randn(m)
    d2 = randn(m)
    opA1 = AffineAdd(MatrixOp(A1), d1)
    opA2 = AffineAdd(MatrixOp(A2), d2)
    opH = HCAT(opA1, opA2)
    x1 = randn(n1)
    x2 = randn(n2)
    y1 = opH * ArrayPartition(x1, x2)
    y2 = A1 * x1 + d1 + A2 * x2 + d2
    @test norm(y1 - y2) <= 1e-12
    y1 = remove_displacement(opH) * ArrayPartition(x1, x2)
    y2 = A1 * x1 + A2 * x2
    @test norm(y1 - y2) <= 1e-12

    m, n1, n2 = 4, 7, 5
    A1 = MatrixOp(randn(m, n1))
    A2 = MatrixOp(randn(m, n2))
    op = HCAT(A1, A2)
    # storage type accessors (results not asserted for value, just exercise paths)
    _ds = domain_storage_type(op)
    _cs = codomain_storage_type(op)
    @test _ds !== nothing
    @test _cs !== nothing
    # thread safety (expected false because of shared output accumulation)
    @test is_thread_safe(op) == false

    # remove_displacement idempotence (no displacement present)
    @test remove_displacement(op) == op

    # with displacement once
    d1 = randn(m)
    d2 = randn(m)
    opd = HCAT(AffineAdd(A1, d1), AffineAdd(A2, d2))
    opd_removed = remove_displacement(opd)
    # applying again should be a no-op
    @test remove_displacement(opd_removed) == opd_removed

    # Show output / fun_name indirect check (two-operator bracket form)
    E = Eye(5)
    D = DiagOp(2 .* ones(5))
    H2 = HCAT(E, D)
    io = IOBuffer(); show(io, H2); shown = String(take!(io))
    @test occursin("HCAT", shown) || occursin("[", shown)

    # Equality and inequality
    Aeq = MatrixOp(randn(4, 3))
    Beq = MatrixOp(randn(4, 2))
    H1a = HCAT(Aeq, Beq)
    H1b = HCAT(Aeq, Beq)
    H1c = HCAT(Beq, Aeq)
    @test H1a == H1b
    @test H1a != H1c

    # Slicing related helpers using GetIndex
    n = 8
    op1 = GetIndex(Float64, (n,), (1:4,))
    op2 = GetIndex(Float64, (n,), (5:8,))
    Hs = HCAT(op1, op2)
    @test is_sliced(Hs) == true
    exprs = AbstractOperators.get_slicing_expr(Hs)
    @test length(exprs) == 2
    @test exprs[1] == (1:4,)
    @test exprs[2] == (5:8,)
    masks = AbstractOperators.get_slicing_mask(Hs)
    @test length(masks) == 2
    @test sum(masks[1]) == 4 && sum(masks[2]) == 4
    Hs_removed = AbstractOperators.remove_slicing(Hs)
    @test is_sliced(Hs_removed) == false

    # Permute domain ordering and ensure type stability
    p2 = collect(Iterators.reverse(1:ndoms(H1a, 2)))
    Hp = AbstractOperators.permute(H1a, p2)
    @test typeof(Hp) <: HCAT
    xA = randn(size(Aeq, 2)); xB = randn(size(Beq, 2))
    y_orig = H1a * ArrayPartition(xA, xB)
    xin = p2 == [2,1] ? ArrayPartition(xB, xA) : ArrayPartition(xA, xB)
    y_perm = Hp * xin
    @test y_orig ≈ y_perm

    # diag_AAc accumulation explicit check on known simple ops (DiagOp and Eye)
    dvals = randn(5) .+ im .* randn(5)
    Hdiag = HCAT(DiagOp(dvals), Eye(ComplexF64, 5))
    @test diag_AAc(Hdiag) == dvals .* conj(dvals) .+ 1

    # nonlinear operator test (Sigmoid + MatrixOp)
    n, m = 4, 3
    x = ArrayPartition(randn(n), randn(m))
    r = randn(m)
    A = randn(m, n)
    B = Sigmoid(Float64, (m,), 2)
    op = HCAT(MatrixOp(A), B)

    y, grad = test_NLop(op, x, r, verb)

    Y = A * x.x[1] + B * x.x[2]
    @test norm(Y - y) < 1e-8

    m, n = 3, 5
    x = ArrayPartition(randn(m), randn(n))
    r = randn(m)
    A = Sin(Float64, (m,))
    M = randn(m, n)
    B = MatrixOp(M)
    op = HCAT(A, B)

    y, grad = test_NLop(op, x, r, verb)

    Y = A * x.x[1] + M * x.x[2]
    @test norm(Y - y) < 1e-8

    p = [2, 1]
    opP = AbstractOperators.permute(op, p)
    xp = ArrayPartition(x.x[p]...)
    J = Jacobian(opP, xp)'
    verb && println(size(J, 1))
    y, grad = test_NLop(opP, xp, r, verb)

    # Test HCAT constructor error paths
    @testset "HCAT constructor errors" begin
        # DimensionMismatch: operators with different codomain dimensions
        A1 = MatrixOp(randn(4, 3))
        A2 = MatrixOp(randn(5, 2))  # Different codomain dimension (5 vs 4)
        @test_throws DimensionMismatch HCAT(A1, A2)
        
        # codomain_type mismatch: real vs complex
        A1 = MatrixOp(randn(4, 3))
        A2 = MatrixOp(randn(ComplexF64, 4, 2))
        @test_throws Exception HCAT(A1, A2)
    end

    # Test HCAT with nested HCAT (flattening behavior)
    @testset "HCAT flattening" begin
        m, n1, n2, n3 = 4, 3, 2, 5
        A1 = MatrixOp(randn(m, n1))
        A2 = MatrixOp(randn(m, n2))
        A3 = MatrixOp(randn(m, n3))
        
        # Create nested HCAT
        H1 = HCAT(A1, A2)
        H2 = HCAT(H1, A3)  # This should flatten
        
        # Test that it works correctly
        x1, x2, x3 = randn(n1), randn(n2), randn(n3)
        y = H2 * ArrayPartition(x1, x2, x3)
        y_expected = A1 * x1 + A2 * x2 + A3 * x3
        @test norm(y - y_expected) < 1e-12
        
        # Test adjoint also works
        y_test = randn(m)
        x_adj = H2' * y_test
        @test length(x_adj.x) == 3  # ArrayPartition has .x field
        
        # More complex nesting: HCAT(HCAT(...), HCAT(...))
        H3 = HCAT(A1, A2)
        H4 = HCAT(A2, A3)
        H5 = HCAT(H3, H4)  # Should flatten all
        
        x_full = ArrayPartition(x1, x2, x2, x3)
        y2 = H5 * x_full
        y2_expected = A1 * x1 + A2 * x2 + A2 * x2 + A3 * x3
        @test norm(y2 - y2_expected) < 1e-12
    end

    # Test single operator HCAT (should return the operator itself)
    @testset "HCAT single operator" begin
        A = MatrixOp(randn(4, 3))
        H_single = HCAT(A)
        @test H_single === A  # Should return the same operator
    end
end
