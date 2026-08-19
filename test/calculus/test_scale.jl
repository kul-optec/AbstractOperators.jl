@testitem "Scale" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n = 8, 4
    coeff = pi
    A1 = randn(m, n)
    opA1 = MatrixOp(A1)
    opS = Scale(coeff, opA1)
    x1 = randn(n)
    y1 = test_op(opS, x1, randn(m), verb)
    y2 = coeff * A1 * x1
    @test norm(y1 - y2) <= 1.0e-12

    coeff2 = 3
    opS2 = Scale(coeff2, opS)
    y1 = test_op(opS2, x1, randn(m), verb)
    y2 = coeff2 * coeff * A1 * x1
    @test norm(y1 - y2) <= 1.0e-12

    opF = FiniteDiff((m,))
    opS = Scale(coeff, opF)
    x1 = randn(m)
    y1 = test_op(opS, x1, diff(randn(m)), verb)
    y2 = coeff * (diff(x1))
    @test norm(y1 - y2) <= 1.0e-12

    op = Scale(-4.0, GetIndex((10,), 1:5))
    @test is_AAc_diagonal(op) == true
    @test diag_AAc(op) == 16

    op = Scale(-4.0, ZeroPad((10,), 20))
    @test is_AcA_diagonal(op) == true
    @test diag_AcA(op) == 16

    d = randn(10)
    op = Scale(3, DiagOp(d))
    @test typeof(op) <: DiagOp
    @test norm(diag(op) - 3 .* d) < 1.0e-12

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
    @test norm(y1 - y2) <= 1.0e-12
    y1 = remove_displacement(opS) * x1
    y2 = coeff * (A1 * x1)
    @test norm(y1 - y2) <= 1.0e-12

    # Edge cases
    # 1. coeff == 1 path should return the original operator (early return)
    op = MatrixOp(randn(5, 5))
    s = Scale(1, op)
    @test s === op  # identity return

    # 2. Real codomain, complex coefficient should throw (error branch in Scale(coeff, coeff_conj, L))
    real_op = MatrixOp(randn(4, 4))  # domain/codomain both Real
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
    io = IOBuffer()
    show(io, S1)
    sS = String(take!(io))
    @test occursin("α", sS)

    # has_optimized_normalop + get_normal_op passthrough (using GetIndex which has optimized normal)
    nGI = 8
    kGI = 5
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
    mH = 6
    n1 = 2
    n2 = 1
    A1p = MatrixOp(randn(mH, n1))
    A2p = MatrixOp(randn(mH, n2))
    H = HCAT(A1p, A2p)
    SH = Scale(2.0, H)
    x1p = randn(n1)
    x2p = randn(n2)
    y_orig = SH * ArrayPartition(x1p, x2p)
    p = [2, 1]
    SHp = AbstractOperators.permute(SH, p)
    y_perm = SHp * ArrayPartition(x2p, x1p)
    @test y_orig ≈ y_perm

    # opnorm and estimate_opnorm passthrough
    opnorm_S = opnorm(S1)
    @test opnorm_S ≈ abs(S1.coeff) * opnorm(Aeq) rtol = 5.0e-6
    @test opnorm_S ≈ estimate_opnorm(S1) rtol = 0.05

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
    @test norm(Y - y) < 1.0e-8

    m = 3
    x = randn(m)
    r = randn(m)
    A = Pow(Float64, (m,), 2)
    op = -A

    y, grad = test_NLop(op, x, r, verb)

    Y = -A * x
    @test norm(Y - y) < 1.0e-8

end

@testitem "Scale constructors and basic mapping" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = MatrixOp(randn(6, 4))
    α = 2.5
    S = Scale(α, A)
    @test size(S) == size(A)
    x = randn(4)
    @test S * x ≈ α * (A * x)
    z = randn(6)
    @test S' * z ≈ α * (A' * z)
end

@testitem "Scale special constructor scale-of-scale" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = FiniteDiff((10, 2))
    α = 3.0
    β = -2.0
    S1 = Scale(α, A)
    S2 = Scale(β, S1)
    x = randn(10, 2)
    @test S1 * x ≈ α * (A * x)
    @test S2 * x ≈ (β * α) * (A * x)
end

@testitem "Scale coeff==1 returns original" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = MatrixOp(randn(5, 5))
    S = Scale(1.0, A)
    @test S === A || S == A
    x = randn(5)
    @test S * x ≈ A * x
end

@testitem "Scale properties delegation" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = Eye(7)
    α = 4.2
    S = Scale(α, A)
    @test domain_type(S) == domain_type(A)
    @test codomain_type(S) == codomain_type(A)
    @test domain_array_type(S) == domain_array_type(A)
    @test codomain_array_type(S) == codomain_array_type(A)
    @test is_thread_safe(S) == is_thread_safe(A)
    @test is_linear(S) && !is_null(S)
    @test is_diagonal(S) == is_diagonal(A)
    @test is_invertible(S) == is_invertible(A)
    @test AbstractOperators.fun_name(S) == "α" * AbstractOperators.fun_name(A)
    @test AbstractOperators.diag(S) == α * AbstractOperators.diag(A)
    @test AbstractOperators.diag_AcA(S) == α^2 * AbstractOperators.diag_AcA(A)
    @test AbstractOperators.diag_AAc(S) == α^2 * AbstractOperators.diag_AAc(A)
    @test is_full_row_rank(S) == is_full_row_rank(A)
    @test is_full_column_rank(S) == is_full_column_rank(A)
end

@testitem "Scale real vs complex coefficient error path" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    @test_throws ErrorException Scale(1.0 + 2.0im, Eye(5))
end

@testitem "Scale get_normal_op paths" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = Eye(8)
    α = 2.0
    S = Scale(α, A)
    N = AbstractOperators.get_normal_op(S)
    @test N isa Scale
    @test N.coeff ≈ α * α
    @test N.A == AbstractOperators.get_normal_op(A)
    NL = Sigmoid((5,))
    S2 = Scale(α, NL)
    @test !AbstractOperators.is_linear(NL)
    @test_throws ErrorException AbstractOperators.get_normal_op(S2)
end

@testitem "Scale equality and remove_displacement" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = AffineAdd(Eye(6), randn(6))
    α = 1.5
    S1 = Scale(α, A)
    S2 = Scale(α, A)
    @test S1 == S2
    Srd = AbstractOperators.remove_displacement(S1)
    @test Srd.A == AbstractOperators.remove_displacement(A)
    @test Srd.coeff == α
end

@testitem "Scale slicing and remove_slicing" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = Eye(11)
    α = 2.0
    As = A[1:9]
    S = Scale(α, As)
    @test AbstractOperators.is_sliced(S)
    @test AbstractOperators.get_slicing_expr(S) !== nothing
    Sr = AbstractOperators.remove_slicing(S)
    @test Sr == Scale(α, AbstractOperators.remove_slicing(As))
end

@testitem "Scale opnorm and estimate_opnorm" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = MatrixOp(randn(7, 4))
    α = -1.2
    S = Scale(α, A)
    @test AbstractOperators.has_fast_opnorm(S) == AbstractOperators.has_fast_opnorm(A)
    @test opnorm(S) ≈ abs(α) * opnorm(A)
    @test estimate_opnorm(S) ≈ opnorm(S) rtol = 0.05
end

@testitem "Scale permute utility" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = HCAT(Eye(5), Eye(5))
    α = 2.2
    S = Scale(α, A)
    p = [2, 1]
    Spr = AbstractOperators.permute(S, p)
    @test Spr.A == AbstractOperators.permute(S.A, p)
    x1 = randn(5)
    x2 = randn(5)
    y_original = S * ArrayPartition(x1, x2)
    y_permuted = Spr * ArrayPartition(x2, x1)
    @test y_original ≈ y_permuted
end

@testitem "Scale threaded behavior" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = MatrixOp(randn(20000, 5))
    α = 0.75
    S = Scale(α, A; threaded = true)
    x = randn(5)
    @test S * x ≈ α * (A * x)
end

@testitem "Scale (GPU)" tags = [:gpu, :calculus, :Scale] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        m = 8
        op = Scale(pi, FiniteDiff(gpu_zeros(backend, Float64, m)))
        test_op(op, gpu_randn(backend, m), gpu_randn(backend, m - 1), false)

        n = 4
        op = Scale(2.0, DiagOp(gpu_randn(backend, n)))
        test_op(op, gpu_randn(backend, n), gpu_randn(backend, n), false)
    end
end

@testitem "Scale(coeff, AdjointMatrixOp) constructor paths" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators, LinearAlgebra
    Random.seed!(0)
    n = 5
    A = MatrixOp(randn(n, n))
    adj_A = A'
    # coeff == 1 → returns the AdjointOperator unchanged (line 79)
    result1 = Scale(1.0, adj_A)
    @test result1 === adj_A
    x = randn(n)
    @test result1 * x ≈ adj_A * x
    # coeff != 1 → returns AdjointOperator(Scale(conj(coeff), A)) (line 84)
    result2 = Scale(2.0, adj_A)
    @test result2 * x ≈ 2.0 * (adj_A * x)
    # complex coefficient on complex adjoint MatrixOp
    B = MatrixOp(randn(ComplexF64, n, n))
    adj_B = B'
    result3 = Scale(1.5 + 0.5im, adj_B)
    xc = randn(ComplexF64, n)
    @test result3 * xc ≈ (1.5 + 0.5im) * (adj_B * xc)
end

@testitem "Scale: threading traits and copy_operator" tags = [:calculus, :Scale] setup = [TestUtils] begin
    using Random, AbstractOperators
    using RecursiveArrayTools: ArrayPartition
    Random.seed!(1)

    n = 4096
    threaded_leaf = Sin(Float64, (n,); threaded = true)
    serial_leaf = Cos(Float64, (n,); threaded = false)
    @test is_threaded(Scale(2.0, threaded_leaf; threaded = false)) == true
    @test is_threaded(Scale(2.0, serial_leaf; threaded = false)) == false
    @test supports_threading(Scale(2.0, serial_leaf)) == true

    S = Scale(2.0, serial_leaf)
    S2 = copy_operator(S; threaded = true)
    @test S2 isa AbstractOperators.Scale
    x = randn(n)
    @test S * x ≈ S2 * x

    # Multi-codomain operator: `codomain_type`/`codomain_array_type` come back as a Tuple /
    # ArrayPartition, exercised through `codomain_type_for_policy`/`codomain_array_type_for_policy`.
    m1, m2, k = 3, 5, 2
    D = DCAT(MatrixOp(randn(m1, k)), MatrixOp(randn(m2, k)))
    S_multi = Scale(2.0, D)
    @test supports_threading(S_multi) == true
    y = ArrayPartition(randn(k), randn(k))
    @test collect(S_multi * y) ≈ 2.0 .* collect(D * y)
end
