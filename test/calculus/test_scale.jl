using Test
using LinearAlgebra
using AbstractOperators
using RecursiveArrayTools

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
    @test opnorm_S ≈ estimate_opnorm(S1) rtol=0.05

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

    @testset "Scale constructors and basic mapping" begin
        # Base operator
        A = MatrixOp(randn(6,4))
        α = 2.5
        S = Scale(α, A)
        @test size(S) == size(A)
        x = randn(4)
        y_ref = α * (A * x)
        y = S * x
        @test y ≈ y_ref
        # Adjoint mapping
        y2_ref = α * (A * x)
        @test S * x ≈ y2_ref
        z = randn(6)
        # Adjoint operator action (uses coeff_conj though real here)
        @test S' * z ≈ α * (A' * z)
    end

    @testset "Scale special constructor scale-of-scale" begin
        A = FiniteDiff((10,2))
        α = 3.0
        β = -2.0
        S1 = Scale(α, A)
        S2 = Scale(β, S1)  # should multiply coefficients and unwrap
        x = randn(10,2)
        # FiniteDiff maps (10,2)->(9,2), ensure shape works
        v1 = S1 * x
        v2 = S2 * x
        @test v1 ≈ α * (A * x)
        @test v2 ≈ (β*α) * (A * x)
    end

    @testset "Scale coeff==1 returns original" begin
        A = MatrixOp(randn(5,5))
        S = Scale(1.0, A)
        @test S === A || S == A
        # Ensure behavior matches
        x = randn(5)
        @test (S * x) ≈ (A * x)
    end

    @testset "Scale properties delegation" begin
        A = Eye(7)
        α = 4.2
        S = Scale(α, A)
        @test domain_type(S) == domain_type(A)
        @test codomain_type(S) == codomain_type(A)
        @test domain_storage_type(S) == domain_storage_type(A)
        @test codomain_storage_type(S) == codomain_storage_type(A)
        @test is_thread_safe(S) == is_thread_safe(A)
        @test is_linear(S)
        @test !is_null(S)
        @test is_diagonal(S) == is_diagonal(A)
        @test is_invertible(S) == is_invertible(A)
        @test AbstractOperators.fun_name(S) == "α" * AbstractOperators.fun_name(A)
        # diag and diag_AcA / diag_AAc for Eye should be scalar scaling (Eye diag returns ones vector)
        @test AbstractOperators.diag(S) == α * AbstractOperators.diag(A)
        @test AbstractOperators.diag_AcA(S) == α^2 * AbstractOperators.diag_AcA(A)
        @test AbstractOperators.diag_AAc(S) == α^2 * AbstractOperators.diag_AAc(A)
    end

    @testset "Scale real vs complex coefficient error path" begin
        A = Eye(5)  # real codomain
        αc = 1.0 + 2.0im
        @test_throws ErrorException Scale(αc, A)  # triggers codomain real + complex coeff error
    end

    @testset "Scale get_normal_op paths" begin
        # Linear operator case: Eye has optimized normal op
        A = Eye(8)
        α = 2.0
        S = Scale(α, A)
        N = AbstractOperators.get_normal_op(S)
        # For linear A: should be Scale(|α|^2, |α|^2, get_normal_op(A))
        @test N isa Scale
        @test N.coeff ≈ α*α
        @test N.A == AbstractOperators.get_normal_op(A)

        NL = Sigmoid((5,))  # 1D nonlinear operator
        @test !AbstractOperators.is_linear(NL)
        S2 = Scale(α, NL)
        # Nonlinear path attempts L' * L which is not allowed; ensure it throws
        @test_throws ErrorException AbstractOperators.get_normal_op(S2)
    end

    @testset "Scale equality and remove_displacement" begin
        A = AffineAdd(Eye(6), randn(6))  # has displacement
        α = 1.5
        S1 = Scale(α, A)
        S2 = Scale(α, A)
        @test S1 == S2
        Srd = AbstractOperators.remove_displacement(S1)
        # remove_displacement(AffineAdd) removes the displacement leaving the inner operator
        @test Srd.A == AbstractOperators.remove_displacement(A)
        @test Srd.coeff == α
    end

    @testset "Scale slicing and remove_slicing" begin
        # Use Eye so that slicing puts GetIndex first in the Compose chain (required by remove_slicing)
        A = Eye(11)
        α = 2.0
        As = A[1:9]              # sliced codomain, Compose(GetIndex, Eye)
        S = Scale(α, As)
        @test AbstractOperators.is_sliced(S)
        expr = AbstractOperators.get_slicing_expr(S)
        @test expr !== nothing
        # remove_slicing should succeed and strip the GetIndex
        Sr = AbstractOperators.remove_slicing(S)
        @test Sr == Scale(α, AbstractOperators.remove_slicing(As))
    end

    @testset "Scale opnorm and estimate_opnorm" begin
        A = MatrixOp(randn(7,4))
        α = -1.2
        S = Scale(α, A)
        @test opnorm(S) ≈ abs(α) * opnorm(A)
        @test estimate_opnorm(S) ≈ opnorm(S) rtol=0.05
    end

    @testset "Scale permute utility" begin
        # Use HCAT of two Eyes so underlying operator supports permute
        A = HCAT(Eye(5), Eye(5))
        α = 2.2
        S = Scale(α, A)
        # Permutation over domain blocks (two blocks) swaps them
        p = [2,1]
        Spr = AbstractOperators.permute(S, p)
        @test Spr.A == AbstractOperators.permute(S.A, p)
        # Build domain input as ArrayPartition matching HCAT domain ordering
        x1 = randn(5); x2 = randn(5)
        xp = ArrayPartition(x1, x2)
        y_original = S * xp
        # After permuting, operator expects swapped domain order
        xp_swapped = ArrayPartition(x2, x1)
        y_permuted = Spr * xp_swapped
        @test y_original ≈ y_permuted
    end

    @testset "Scale threaded behavior" begin
        # Force threading decision by large output length
        A = MatrixOp(randn(20000,5))
        α = 0.75
        S = Scale(α, A; threaded=true)
        x = randn(5)
        y = S * x
        @test y ≈ α * (A * x)
    end
end
