@testitem "HCAT: basic mul" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

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
    @test norm(y1 - y2) <= 1.0e-12

    # permutation
    p = [2; 1]
    opHp = opH[p]
    y1 = test_op(opHp, ArrayPartition(x2, x1), randn(m), verb)
    @test norm(y1 - y2) <= 1.0e-12

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
    @test norm(y1 - (A1 * x1 + A2 * x2 + A3 * x3)) <= 1.0e-12

    # HCAT of HCAT (flattening)
    opHH = HCAT(opH, opA2, opA3)
    y1 = test_op(opHH, ArrayPartition(x1, x2, x3, x2, x3), randn(m), verb)
    @test norm(y1 - (A1 * x1 + A2 * x2 + A3 * x3 + A2 * x2 + A3 * x3)) <= 1.0e-12
end

@testitem "HCAT: properties" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    m, n1, n2, n3 = 4, 7, 5, 6
    opA1 = MatrixOp(randn(m, n1))
    opA2 = MatrixOp(randn(m, n2))
    opA3 = MatrixOp(randn(m, n3))
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
    op2 = HCAT(DiagOp(d1), DiagOp(d2))
    @test is_AAc_diagonal(op2) == true
    @test diag_AAc(op2) == d1 .* conj(d1) .+ d2 .* conj(d2)
    y1 = randn(n1) .+ im .* randn(n1)
    @test norm(op2 * (op2' * y1) .- diag_AAc(op2) .* y1) < 1.0e-12

    # storage type and thread safety
    A1 = MatrixOp(randn(m, n1))
    A2 = MatrixOp(randn(m, n2))
    op3 = HCAT(A1, A2)
    @test domain_array_type(op3) !== nothing
    @test codomain_array_type(op3) !== nothing
    @test is_thread_safe(op3) == false
end

@testitem "HCAT: displacement" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

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
    @test norm(y1 - (A1 * x1 + d1 + A2 * x2 + d2)) <= 1.0e-12
    y2 = remove_displacement(opH) * ArrayPartition(x1, x2)
    @test norm(y2 - (A1 * x1 + A2 * x2)) <= 1.0e-12

    # remove_displacement idempotence
    A1b = MatrixOp(randn(m, n1))
    A2b = MatrixOp(randn(m, n2))
    op = HCAT(A1b, A2b)
    @test remove_displacement(op) == op
    opd = HCAT(AffineAdd(A1b, d1), AffineAdd(A2b, d2))
    opd_removed = remove_displacement(opd)
    @test remove_displacement(opd_removed) == opd_removed
end

@testitem "HCAT: slicing and permute utilities" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = 8
    op1 = GetIndex(Float64, (n,), (1:4,))
    op2 = GetIndex(Float64, (n,), (5:8,))
    Hs = HCAT(op1, op2)
    @test is_sliced(Hs) == true
    exprs = AbstractOperators.get_slicing_expr(Hs)
    @test length(exprs) == 2
    @test exprs[1] == (1:4,) && exprs[2] == (5:8,)
    masks = AbstractOperators.get_slicing_mask(Hs)
    @test length(masks) == 2
    @test sum(masks[1]) == 4 && sum(masks[2]) == 4
    @test !is_sliced(AbstractOperators.remove_slicing(Hs))

    d1, d2 = randn(5), randn(5)
    D1 = DiagOp(d1) * GetIndex((10,), 1:5)
    D2 = DiagOp(d2) * GetIndex((10,), 6:10)
    Hs_comp = HCAT(D1, D2)
    @test is_sliced(Hs_comp)
    @test AbstractOperators.get_slicing_expr(Hs_comp) == ((1:5,), (6:10,))
    @test !is_sliced(AbstractOperators.remove_slicing(Hs_comp))

    m, n1, n2 = 4, 3, 2
    Aeq = MatrixOp(randn(m, n1))
    Beq = MatrixOp(randn(m, n2))
    H1a = HCAT(Aeq, Beq)
    p2 = collect(Iterators.reverse(1:ndoms(H1a, 2)))
    Hp = AbstractOperators.permute(H1a, p2)
    @test typeof(Hp) <: HCAT
    xA = randn(size(Aeq, 2))
    xB = randn(size(Beq, 2))
    y_orig = H1a * ArrayPartition(xA, xB)
    xin = p2 == [2, 1] ? ArrayPartition(xB, xA) : ArrayPartition(xA, xB)
    @test y_orig ≈ Hp * xin
end

@testitem "HCAT: nonlinear operators" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m = 4, 3
    x = ArrayPartition(randn(n), randn(m))
    r = randn(m)
    A = randn(m, n)
    B = Sigmoid(Float64, (m,), 2)
    op = HCAT(MatrixOp(A), B)
    y, grad = test_NLop(op, x, r, verb)
    @test norm(A * x.x[1] + B * x.x[2] - y) < 1.0e-8

    n, m = 5, 3
    x = ArrayPartition(randn(m), randn(n))
    r = randn(m)
    A_sin = Sin(Float64, (m,))
    M = randn(m, n)
    op2 = HCAT(A_sin, MatrixOp(M))
    y2, grad2 = test_NLop(op2, x, r, verb)
    @test norm(A_sin * x.x[1] + M * x.x[2] - y2) < 1.0e-8
end

@testitem "HCAT constructor errors" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    A1 = MatrixOp(randn(4, 3))
    A2 = MatrixOp(randn(5, 2))
    @test_throws DimensionMismatch HCAT(A1, A2)

    A1 = MatrixOp(randn(4, 3))
    A2 = MatrixOp(randn(ComplexF64, 4, 2))
    @test_throws Exception HCAT(A1, A2)
end

@testitem "HCAT flattening" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n1, n2, n3 = 4, 3, 2, 5
    A1 = MatrixOp(randn(m, n1))
    A2 = MatrixOp(randn(m, n2))
    A3 = MatrixOp(randn(m, n3))
    H1 = HCAT(A1, A2)
    H2 = HCAT(H1, A3)

    x1, x2, x3 = randn(n1), randn(n2), randn(n3)
    y = H2 * ArrayPartition(x1, x2, x3)
    y_expected = A1 * x1 + A2 * x2 + A3 * x3
    @test norm(y - y_expected) < 1.0e-12

    y_test = randn(m)
    x_adj = H2' * y_test
    @test length(x_adj.x) == 3

    H3 = HCAT(A1, A2)
    H4 = HCAT(A2, A3)
    H5 = HCAT(H3, H4)
    x_full = ArrayPartition(x1, x2, x2, x3)
    y2 = H5 * x_full
    y2_expected = A1 * x1 + A2 * x2 + A2 * x2 + A3 * x3
    @test norm(y2 - y2_expected) < 1.0e-12
end

@testitem "HCAT single operator" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    A = MatrixOp(randn(4, 3))
    H_single = HCAT(A)
    @test H_single === A
end

@testitem "HCAT (GPU)" tags = [:gpu, :calculus, :HCAT] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n = 4
        opH = HCAT(DiagOp(gpu_ones(backend, Float64, n)), DiagOp(to_gpu(backend, 2 .* ones(n))))
        test_op(opH, ArrayPartition(gpu_randn(backend, n), gpu_randn(backend, n)), gpu_randn(backend, n), false)

        m, n1, n2 = 4, 7, 5
        A1 = gpu_randn(backend, m, n1)
        A2 = gpu_randn(backend, m, n2)
        opH2 = HCAT(MatrixOp(A1), MatrixOp(A2))
        test_op(opH2, ArrayPartition(gpu_randn(backend, n1), gpu_randn(backend, n2)), gpu_randn(backend, m), false)
    end
end

@testitem "HCAT fun_name reversed idxs (line 294)" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n = 8
    A1 = MatrixOp(randn(4, n))
    A2 = MatrixOp(randn(4, n))
    H = HCAT(A1, A2)
    # permute swaps domain slot ordering → idxs[1] == 2 triggers reversed branch (line 294)
    Hp = AbstractOperators.permute(H, [2, 1])
    @test Hp isa HCAT
    name = AbstractOperators.fun_name(Hp)
    @test occursin(",", name)
end

@testitem "HCAT get_slicing_expr: single-expr return (line 337)" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n = 10
    # Single-element HCAT → length(exprs) == 1 → return exprs[1]
    op_gi = GetIndex(Float64, (n,), (1:4,))
    H_single = HCAT((op_gi,), (1,), zeros(4))
    @test AbstractOperators.is_sliced(H_single)
    expr_single = AbstractOperators.get_slicing_expr(H_single)
    @test expr_single == (1:4,)
end

@testitem "HCAT get_slicing_expr: multi-element loop (line 330)" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using AbstractOperators
    n = 12
    op1 = GetIndex(Float64, (n,), (1:4,))
    op2 = GetIndex(Float64, (n,), (5:8,))
    op3 = GetIndex(Float64, (n,), (9:12,))
    H = HCAT(op1, op2, op3)
    @test AbstractOperators.is_sliced(H)
    exprs = AbstractOperators.get_slicing_expr(H)
    @test exprs == ((1:4,), (5:8,), (9:12,))
end

@testitem "HCAT getindex: tuple-idxs error (line 87)" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n = 4
    # HCAT with a Compose(MatrixOp, HCAT) sub-operator gives tuple idxs
    H_inner = HCAT(MatrixOp(randn(n, n)), MatrixOp(randn(n, n)))
    C_sub = Compose(MatrixOp(randn(n, n)), H_inner)
    H_outer = HCAT(MatrixOp(randn(n, n)), C_sub)
    @test H_outer.idxs == (1, (2, 3))
    # Selecting partial index into the tuple-idxs sub-op should error
    @test_throws ErrorException H_outer[2]
end

@testitem "HCAT: copy_operator" tags = [:calculus, :HCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(4)

    m, n1, n2 = 4, 7, 5
    opH = HCAT(MatrixOp(randn(m, n1)), MatrixOp(randn(m, n2)))
    opH2 = copy_operator(opH; threaded = true)
    @test opH2 isa HCAT
    x = ArrayPartition(randn(n1), randn(n2))
    @test collect(opH * x) ≈ collect(opH2 * x)
    # Verify independence: forward into opH2 alone
    x2 = ArrayPartition(randn(n1), randn(n2))
    @test collect(opH2 * x2) ≈ collect(opH * x2)
end

@testitem "HCAT: multi-element tuple-idxs mul!/adjoint/permute, serial and threaded" tags = [
    :calculus, :HCAT,
] setup = [TestUtils] begin
    using Random, AbstractOperators, RecursiveArrayTools
    Random.seed!(6)

    # Build an HCAT whose sub-operators are a mix of single-domain and a genuinely
    # multi-domain (non-HCAT) operator, so one entry of `H.idxs` is a tuple rather than an
    # Integer -- the `Pi <: Integer` false branch in every generated function below.
    # `Compose(cheap_op, HCAT(...))` keeps the inner HCAT from being auto-flattened by the
    # outer constructor while staying cheap (no dense matrices) so the size can be pushed
    # above HCAT's block-threading threshold without blowing up memory. The wrapper must be
    # non-diagonal: `Compose(DiagOp, HCAT(...))` gets simplified straight back into a flat
    # HCAT by the package's combination rules ("diagonal * HCAT always simplifies to HCAT").
    # `GetIndex` with a same-length index range is not a usable alternative here either --
    # its constructor detects `dim_out == dim_in` and silently returns `Eye` regardless of
    # index order, which is diagonal too. FiniteDiff is cheap and genuinely non-diagonal, at
    # the cost of shrinking the domain by one element, hence the mixed slot sizes below.
    function build_hcat(n)
        inner = HCAT(DiagOp(randn(n)), DiagOp(randn(n)))
        multi = Compose(FiniteDiff(Float64, (n,)), inner)   # domain (n,n), codomain (n-1,)
        return HCAT(DiagOp(randn(n - 1)), DiagOp(randn(n - 1)), DiagOp(randn(n - 1)), multi)
    end

    # Small, serial case. `test_op` checks in-place-vs-allocating consistency and the
    # forward/adjoint dot-product identity, so it is a real correctness check, not just a
    # smoke test.
    n = 5
    H = build_hcat(n)
    @test H.idxs == (1, 2, 3, (4, 5))   # confirms the tuple-idxs (natural) branch is in play
    io = IOBuffer(); show(io, H); s = String(take!(io))
    @test occursin("HCAT", s)   # fun_name's N != 2 branch

    x = ArrayPartition(randn(n - 1), randn(n - 1), randn(n - 1), randn(n), randn(n))
    r = randn(n - 1)
    test_op(H, x, r, false)

    # `permute` reindexes over the *flattened* domain slots (length 5 here: 3 singles + the
    # 2-slot tuple entry), not over the 4 top-level sub-operators -- a length-4 permutation
    # vector silently corrupts `invpermute!`. Swapping only the two slots *within* the
    # tuple entry keeps every slot's size matched to its new position (both are size `n`)
    # while still making the resulting `idxs` non-natural, exercising the indexed branch.
    p = [1, 2, 3, 5, 4]
    H_perm = AbstractOperators.permute(H, p)
    @test !AbstractOperators._hcat_has_natural_idxs(H_perm)
    test_op(H_perm, x, r, false)

    if Threads.nthreads() > 1
        # Above HCAT's per-block threshold (2^17) with >= MIN_BLOCKS_FOR_PARALLEL (4) blocks,
        # so the threaded adjoint (`_hcat_block_adj!`) actually runs, in both its natural and
        # indexed multi-element forms.
        big_n = 1 << 17
        H_big = build_hcat(big_n)
        H_big = AbstractOperators.adapt_operator(H_big; threaded = true)
        @test AbstractOperators.is_block_threaded(H_big) == true
        xb = ArrayPartition(randn(big_n - 1), randn(big_n - 1), randn(big_n - 1), randn(big_n), randn(big_n))
        rb = randn(big_n - 1)
        test_op(H_big, xb, rb, false)

        H_big_perm = AbstractOperators.permute(H_big, p)
        @test AbstractOperators.is_block_threaded(H_big_perm) == true
        @test !AbstractOperators._hcat_has_natural_idxs(H_big_perm)
        test_op(H_big_perm, xb, rb, false)
    end
end
