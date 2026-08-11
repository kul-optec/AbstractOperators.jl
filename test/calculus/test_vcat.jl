@testitem "VCAT: basic mul" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, m2, n = 4, 7, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opV = VCAT(opA1, opA2)
    x1 = randn(n)
    y1 = test_op(opV, x1, ArrayPartition(randn(m1), randn(m2)), verb)
    @test norm(y1 - ArrayPartition(A1 * x1, A2 * x1)) .<= 1.0e-12

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
    @test norm(y1 - ArrayPartition(A1 * x1, A2 * x1, A3 * x1)) .<= 1.0e-12

    # VCAT of VCAT (flattening)
    opVV = VCAT(opV, opA3)
    y1 = test_op(opVV, x1, ArrayPartition(randn(m1), randn(m2), randn(m3), randn(m3)), verb)
    @test norm(y1 .- ArrayPartition(A1 * x1, A2 * x1, A3 * x1, A3 * x1)) <= 1.0e-12
end

@testitem "VCAT: properties" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, m2, m3, n = 4, 7, 3, 5
    op = VCAT(MatrixOp(randn(m1, n)), MatrixOp(randn(m2, n)), MatrixOp(randn(m3, n)))
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

    d = randn(5) .+ im .* randn(5)
    op2 = VCAT(DiagOp(d), Eye(ComplexF64, 5))
    @test is_AcA_diagonal(op2) == true
    @test diag_AcA(op2) == d .* conj(d) .+ 1

    m1, m2, n = 4, 7, 5
    A1 = MatrixOp(randn(m1, n))
    A2 = MatrixOp(randn(m2, n))
    op3 = VCAT(A1, A2)
    @test domain_array_type(op3) !== nothing
    @test codomain_array_type(op3) !== nothing
    @test is_thread_safe(op3) == false
end

@testitem "VCAT: displacement" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, m2, n = 4, 7, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    d1 = randn(m1)
    d2 = randn(m2)
    opV = VCAT(AffineAdd(MatrixOp(A1), d1), AffineAdd(MatrixOp(A2), d2))
    x1 = randn(n)
    @test norm(opV * x1 - ArrayPartition(A1 * x1 + d1, A2 * x1 + d2)) <= 1.0e-12
    @test norm(remove_displacement(opV) * x1 - ArrayPartition(A1 * x1, A2 * x1)) <= 1.0e-12

    A1b = MatrixOp(randn(m1, n))
    A2b = MatrixOp(randn(m2, n))
    op = VCAT(A1b, A2b)
    @test remove_displacement(op) == op
    opd = VCAT(AffineAdd(A1b, d1), AffineAdd(A2b, d2))
    opd_removed = remove_displacement(opd)
    @test remove_displacement(opd_removed) == opd_removed
end

@testitem "VCAT: nonlinear operators" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m = 4, 3
    x = randn(m)
    r = ArrayPartition(randn(n), randn(m))
    A = randn(n, m)
    B = Sigmoid(Float64, (m,), 2)
    op = VCAT(MatrixOp(A), B)
    y, grad = test_NLop(op, x, r, verb)
    @test norm(ArrayPartition(A * x, B * x) - y) < 1.0e-8
end

@testitem "VCAT: slicing utilities" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 10
    g1 = GetIndex(Float64, (n,), (1:5,))
    g2 = GetIndex(Float64, (n,), (6:10,))
    Vs = VCAT(g1, g2)
    @test is_sliced(Vs) == true
    exprs = AbstractOperators.get_slicing_expr(Vs)
    @test exprs[1] == (1:5,) && exprs[2] == (6:10,)
    @test !is_sliced(AbstractOperators.remove_slicing(Vs))

    # remove_slicing second branch: VCAT of HCATs with GetIndex of different output sizes
    # Forces the elseif branch where new_ops have different domain sizes after remove_slicing
    g3 = GetIndex(Float64, (5,), (1:3,))  # 5-dim → 3-dim
    g4 = GetIndex(Float64, (5,), (1:4,))  # 5-dim → 4-dim
    z3 = Zeros(Float64, (5,), Float64, (3,))  # domain (5,), codomain (3,)
    z4 = Zeros(Float64, (5,), Float64, (4,))  # domain (5,), codomain (4,)
    H1 = HCAT(g3, z3)  # 3×(5+5), both ops have codomain (3,)
    H2 = HCAT(z4, g4)  # 4×(5+5), both ops have codomain (4,)
    Vs2 = VCAT(H1, H2)  # 7×10
    @test is_sliced(Vs2)
    rs2 = AbstractOperators.remove_slicing(Vs2)
    @test rs2 isa VCAT
    @test !is_sliced(rs2)
    # Result should act like DCAT(Eye(3), Eye(4)): maps (x1, x2) → (x1, x2) with no-op zeros
    b3 = randn(3)
    b4 = randn(4)
    b_domain = ArrayPartition(b3, b4)
    y_codomain = ArrayPartition(randn(3), randn(4))
    mul!(y_codomain, rs2, b_domain)
    @test y_codomain.x[1] ≈ b3
    @test y_codomain.x[2] ≈ b4

    # fun_name and equality
    A1 = Eye(3)
    A2 = Eye(3)
    A3 = Eye(3)
    V2 = VCAT(A1, A2)
    V3 = VCAT(A1, A2, A3)
    name2 = AbstractOperators.fun_name(V2)
    @test occursin("[", name2) || occursin("]", name2)
    @test AbstractOperators.fun_name(V3) == "VCAT"
    # Use different operators for equality/inequality test
    Aeq1 = MatrixOp(randn(3, 4))
    Aeq2 = MatrixOp(randn(5, 4))
    @test VCAT(Aeq1, Aeq2) == VCAT(Aeq1, Aeq2)
    @test VCAT(Aeq1, Aeq2) != VCAT(Aeq2, Aeq1)
end

@testitem "VCAT (GPU)" tags = [:gpu, :calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n = 4
        opV = VCAT(DiagOp(gpu_ones(backend, Float64, n)), DiagOp(to_gpu(backend, 2 .* ones(n))))
        test_op(opV, gpu_randn(backend, n), ArrayPartition(gpu_randn(backend, n), gpu_randn(backend, n)), false)

        m1, m2, n = 4, 7, 5
        A1 = gpu_randn(backend, m1, n)
        A2 = gpu_randn(backend, m2, n)
        opV2 = VCAT(MatrixOp(A1), MatrixOp(A2))
        test_op(opV2, gpu_randn(backend, n), ArrayPartition(gpu_randn(backend, m1), gpu_randn(backend, m2)), false)
    end
end

@testitem "VCAT: constructor errors" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    # Line 49: domain dimension mismatch → DimensionMismatch
    @test_throws DimensionMismatch VCAT(MatrixOp(randn(3, 4)), MatrixOp(randn(3, 5)))

    # Line 52: domain type mismatch → generic error (throw(error(...)))
    @test_throws Exception VCAT(MatrixOp(randn(3, 4)), MatrixOp(ones(ComplexF64, 2, 4)))
end

@testitem "VCAT remove_slicing: unsupported VCAT error (line 200)" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using AbstractOperators
    n = 8
    # VCAT of HCATs with no null operators: both removal branches fail → error at line 200
    g1 = GetIndex(Float64, (n,), (1:4,))
    g2 = GetIndex(Float64, (n,), (3:6,))
    g3 = GetIndex(Float64, (n,), (5:8,))
    H1 = HCAT(g1, g2)
    H2 = HCAT(g2, g3)
    Vs = VCAT(H1, H2)
    @test AbstractOperators.is_sliced(Vs)
    @test_throws ErrorException AbstractOperators.remove_slicing(Vs)
end

@testitem "VCAT: copy_operator" tags = [:calculus, :VCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(4)

    n, m1, m2 = 8, 5, 6
    opV = VCAT(MatrixOp(randn(m1, n)), MatrixOp(randn(m2, n)))
    opV2 = copy_operator(opV)
    @test opV2 isa VCAT
    x = randn(n)
    y1 = opV * x
    y2 = opV2 * x
    @test collect(y1) ≈ collect(y2)
    # Verify independence: forward into opV2 alone
    x2 = randn(n)
    @test collect(opV2 * x2) ≈ collect(opV * x2)
end
