@testitem "Combinations: HCAT and Compose" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(42)
    verb && println(" --- Testing Combinations: HCAT and Compose --- ")

    m1, m2, m3, m4 = 4, 7, 3, 2
    A1 = randn(m3, m1)
    A2 = randn(m3, m2)
    A3 = randn(m4, m3)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opH = HCAT(opA1, opA2)
    opC = Compose(opA3, opH)
    x1, x2 = randn(m1), randn(m2)
    y1 = test_op(opC, ArrayPartition(x1, x2), randn(m4), verb)

    y2 = A3 * (A1 * x1 + A2 * x2)

    @test norm(y1 - y2) < 1.0e-9

    opCp = AbstractOperators.permute(opC, [2, 1])
    y1 = test_op(opCp, ArrayPartition(x2, x1), randn(m4), verb)
    @test norm(y1 - y2) < 1.0e-9

    m5 = 10
    A4 = randn(m4, m5)
    x3 = randn(m5)
    opHC = HCAT(opC, MatrixOp(A4))
    x = ArrayPartition(x1, x2, x3)
    y1 = test_op(opHC, x, randn(m4), verb)
    @test norm(y1 - (y2 + A4 * x3)) < 1.0e-9

    p = randperm(ndoms(opHC, 2))
    opHP = AbstractOperators.permute(opHC, p)
    xp = ArrayPartition(x.x[p]...)
    y1 = test_op(opHP, xp, randn(m4), verb)

    pp = randperm(ndoms(opHC, 2))
    opHPP = AbstractOperators.permute(opHC, pp)
    xpp = ArrayPartition(x.x[pp]...)
    y1 = test_op(opHPP, xpp, randn(m4), verb)
end

@testitem "Combinations: VCAT and HCAT mixtures" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(43)
    verb && println(" --- Testing Combinations: VCAT/HCAT --- ")

    # VCAT of HCATs
    m1, m2, n1 = 4, 7, 3
    A1 = randn(n1, m1)
    A2 = randn(n1, m2)
    opH1 = HCAT(MatrixOp(A1), MatrixOp(A2))
    m1, m2, n2 = 4, 7, 5
    A3 = randn(n2, m1)
    A4 = randn(n2, m2)
    opH2 = HCAT(MatrixOp(A3), MatrixOp(A4))
    opV = VCAT(opH1, opH2)
    x1, x2 = randn(m1), randn(m2)
    y1 = test_op(opV, ArrayPartition(x1, x2), ArrayPartition(randn(n1), randn(n2)), verb)
    y2 = ArrayPartition(A1 * x1 + A2 * x2, A3 * x1 + A4 * x2)
    @test norm(y1 - y2) <= 1.0e-12

    # VCAT of HCATs with complex
    m1, m2, n1 = 4, 7, 5
    A1c = randn(n1, m1) + im * randn(n1, m1)
    d1 = rand(ComplexF64, n1)
    opH1c = HCAT(MatrixOp(A1c), DiagOp(Float64, (n1,), d1))
    m1, m2, n2 = 4, 7, 5
    A3c = randn(n2, m1) + im * randn(n2, m1)
    d2 = rand(ComplexF64, n2)
    opH2c = HCAT(MatrixOp(A3c), DiagOp(Float64, (n2,), d2))
    opVc = VCAT(opH1c, opH2c)
    x1c = randn(m1) + im * randn(m1)
    x2c = randn(n2)
    y1c = test_op(
        opVc,
        ArrayPartition(x1c, x2c),
        ArrayPartition(randn(n1) + im * randn(n1), randn(n2) + im * randn(n2)),
        verb,
    )
    y2c = ArrayPartition(A1c * x1c + x2c .* d1, A3c * x1c + x2c .* d2)
    @test norm(y1c - y2c) <= 1.0e-12

    # HCAT of VCATs
    n1, n2, m1, m2 = 3, 5, 4, 7
    A = randn(m1, n1)
    B = randn(m1, n2)
    C = randn(m2, n1)
    D = randn(m2, n2)
    opV2 = HCAT(VCAT(MatrixOp(A), MatrixOp(C)), VCAT(MatrixOp(B), MatrixOp(D)))
    x1 = randn(n1)
    x2 = randn(n2)
    y1 = test_op(opV2, ArrayPartition(x1, x2), ArrayPartition(randn(m1), randn(m2)), verb)
    y2 = ArrayPartition(A * x1 + B * x2, C * x1 + D * x2)
    @test norm(y1 - y2) <= 1.0e-12
end

@testitem "Combinations: Sum structures" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(44)
    verb && println(" --- Testing Combinations: Sum --- ")

    # Sum of HCATs
    m, n1, n2, n3 = 4, 7, 5, 3
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    A3 = randn(m, n3)
    B1 = randn(m, n1)
    B2 = randn(m, n2)
    B3 = randn(m, n3)
    opHA = HCAT(MatrixOp(A1), MatrixOp(A2), MatrixOp(A3))
    opHB = HCAT(MatrixOp(B1), MatrixOp(B2), MatrixOp(B3))
    opS = Sum(opHA, opHB)
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = test_op(opS, ArrayPartition(x1, x2, x3), randn(m), verb)
    y2 = A1 * x1 + B1 * x1 + A2 * x2 + B2 * x2 + A3 * x3 + B3 * x3
    @test norm(y1 - y2) <= 1.0e-12

    p = [3; 2; 1]
    opSp = AbstractOperators.permute(opS, p)
    y1 = test_op(opSp, ArrayPartition(((x1, x2, x3)[p])...), randn(m), verb)

    # Sum of VCATs
    m1, m2, n = 4, 7, 5
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    B1 = randn(m1, n)
    B2 = randn(m2, n)
    C1 = randn(m1, n)
    C2 = randn(m2, n)
    opVA = VCAT(MatrixOp(A1), MatrixOp(A2))
    opVB = VCAT(MatrixOp(B1), MatrixOp(B2))
    opVC = VCAT(MatrixOp(C1), MatrixOp(C2))
    opS = Sum(opVA, opVB, opVC)
    x = randn(n)
    y1 = test_op(opS, x, ArrayPartition(randn(m1), randn(m2)), verb)
    y2 = ArrayPartition(A1 * x + B1 * x + C1 * x, A2 * x + B2 * x + C2 * x)
    @test norm(y1 - y2) .<= 1.0e-12
end

@testitem "Combinations: Scale structures" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(45)
    verb && println(" --- Testing Combinations: Scale --- ")

    # Scale of DCAT
    m1, n1 = 4, 7
    m2, n2 = 3, 5
    A1 = randn(m1, n1)
    A2 = randn(m2, n2)
    opD = DCAT(MatrixOp(A1), MatrixOp(A2))
    coeff = randn()
    opS = Scale(coeff, opD)
    x1 = randn(n1)
    x2 = randn(n2)
    y = test_op(opS, ArrayPartition(x1, x2), ArrayPartition(randn(m1), randn(m2)), verb)
    z = ArrayPartition(coeff * A1 * x1, coeff * A2 * x2)
    @test norm(y - z) <= 1.0e-12

    # Scale of VCAT
    m1, m2, n = 4, 3, 7
    A1 = randn(m1, n)
    A2 = randn(m2, n)
    opV = VCAT(MatrixOp(A1), MatrixOp(A2))
    coeff = randn()
    opS = Scale(coeff, opV)
    x = randn(n)
    y = test_op(opS, x, ArrayPartition(randn(m1), randn(m2)), verb)
    z = ArrayPartition(coeff * A1 * x, coeff * A2 * x)
    @test norm(y - z) <= 1.0e-12

    # Scale of HCAT
    m, n1, n2 = 4, 3, 7
    A1 = randn(m, n1)
    A2 = randn(m, n2)
    opH = HCAT(MatrixOp(A1), MatrixOp(A2))
    coeff = randn()
    opS = Scale(coeff, opH)
    x1 = randn(n1)
    x2 = randn(n2)
    y = test_op(opS, ArrayPartition(x1, x2), randn(m), verb)
    z = coeff * (A1 * x1 + A2 * x2)
    @test norm(y - z) <= 1.0e-12

    # DCAT of HCATs
    m1, m2, n1, n2 = 2, 3, 4, 5
    A1 = randn(m1, n1)
    A2 = randn(m1, n2)
    B1 = randn(m2, n1)
    B2 = randn(m2, n2)
    B3 = randn(m2, n2)
    opH1 = HCAT(MatrixOp(A1), MatrixOp(A2))
    opH2 = HCAT(MatrixOp(B1), MatrixOp(B2), MatrixOp(B3))
    op = DCAT(MatrixOp(A1), opH2)
    x = ArrayPartition(randn.(size(op, 2))...)
    y0 = ArrayPartition(randn.(size(op, 1))...)
    y = test_op(op, x, y0, verb)
    op2 = DCAT(opH1, opH2)
    x = ArrayPartition(randn.(size(op2, 2))...)
    y0 = ArrayPartition(randn.(size(op2, 1))...)
    y = test_op(op2, x, y0, verb)
    p = randperm(ndoms(op2, 2))
    y2 = op2[p] * ArrayPartition(x.x[p]...)
    @test norm(y - y2) <= 1.0e-8

    # Scale of Sum and Compose
    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    opSum = Sum(MatrixOp(A1), MatrixOp(A2))
    coeff = pi
    opSS = Scale(coeff, opSum)
    x1 = randn(n)
    y1 = test_op(opSS, x1, randn(m), verb)
    y2 = coeff * (A1 * x1 + A2 * x1)
    @test norm(y1 - y2) <= 1.0e-12

    m1, m2, m3 = 4, 7, 3
    Ac1 = randn(m2, m1)
    Ac2 = randn(m3, m2)
    opC = Compose(MatrixOp(Ac2), MatrixOp(Ac1))
    opSC = Scale(coeff, opC)
    x = randn(m1)
    y1 = test_op(opSC, x, randn(m3), verb)
    y2 = coeff * (Ac2 * Ac1 * x)
    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "Combinations: Nonlinear" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(46)
    verb && println(" --- Testing Combinations: Nonlinear --- ")

    # Nonlinear HCAT of VCAT
    n, m1, m2, m3 = 4, 3, 2, 7
    x1 = randn(m1)
    x2 = randn(m2)
    x3 = randn(m3)
    x = ArrayPartition(x1, x2, x3)
    r = ArrayPartition(randn(n), randn(m1))
    A1 = randn(n, m1)
    A2 = randn(n, m2)
    A3 = randn(n, m3)
    B1 = Sigmoid(Float64, (m1,), 2)
    B2 = randn(m1, m2)
    B3 = randn(m1, m3)
    op1 = VCAT(MatrixOp(A1), B1)
    op2 = VCAT(MatrixOp(A2), MatrixOp(B2))
    op3 = VCAT(MatrixOp(A3), MatrixOp(B3))
    op = HCAT(op1, op2, op3)
    y, grad = test_NLop(op, x, r, verb)
    Y = ArrayPartition(A1 * x1 + A2 * x2 + A3 * x3, B1 * x1 + B2 * x2 + B3 * x3)
    @test norm(Y - y) < 1.0e-8

    # Nonlinear VCAT of HCAT
    m1, m2, m3, n1, n2 = 3, 4, 5, 6, 7
    x1 = randn(m1)
    x2 = randn(n1)
    x3 = randn(m3)
    x = ArrayPartition(x1, x2, x3)
    r = ArrayPartition(randn(n1), randn(n2))
    A1 = randn(n1, m1)
    B1 = Sigmoid(Float64, (n1,), 2)
    C1 = randn(n1, m3)
    A2 = randn(n2, m1)
    B2 = randn(n2, n1)
    C2 = randn(n2, m3)
    op = VCAT(HCAT(MatrixOp(A1), B1, MatrixOp(C1)), HCAT(MatrixOp(A2), MatrixOp(B2), MatrixOp(C2)))
    y, grad = test_NLop(op, x, r, verb)
    Y = ArrayPartition(A1 * x1 + B1 * x2 + C1 * x3, A2 * x1 + B2 * x2 + C2 * x3)
    @test norm(Y - y) < 1.0e-8

    # Nonlinear AffineAdd and Compose
    n = 10
    d1 = randn(n)
    d2 = randn(n)
    T = Compose(AffineAdd(Sin(n), d2), AffineAdd(Eye(n), d1))
    r = randn(n)
    x = randn(size(T, 2))
    y, grad = test_NLop(T, x, r, verb)
    @test norm(y - (sin.(x + d1) + d2)) < 1.0e-8

    d3 = pi
    T2 = Compose(
        AffineAdd(Sin(n), d3), Compose(AffineAdd(Exp(n), d2, false), AffineAdd(Eye(n), d1))
    )
    r = randn(n)
    x = randn(size(T2, 2))
    y, grad = test_NLop(T2, x, r, verb)
    @test norm(y - (sin.(exp.(x + d1) - d2) .+ d3)) < 1.0e-8
end

@testitem "Combinations: AffineAdd merging and Zeros" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(47)
    verb && println(" --- Testing Combinations: AffineAdd merging and Zeros --- ")

    n = 8

    # AffineAdd(linear) * AffineAdd(linear) — S1==S2==true
    d1 = randn(n)
    d2 = randn(n)
    T1 = AffineAdd(Eye(n), d1)        # x + d1, S=true
    T2 = AffineAdd(Eye(n), d2)        # x + d2, S=true
    Tc = T1 * T2                       # should combine, not stay as Compose
    @test !(Tc isa AbstractOperators.Compose)
    x = randn(n)
    @test norm(Tc * x - (x + d2 + d1)) < 1.0e-12

    # AffineAdd(linear) * AffineAdd(linear) — S1=true, S2=false
    T3 = AffineAdd(Eye(n), d1, true)  # x + d1
    T4 = AffineAdd(Eye(n), d2, false) # x - d2
    Tc2 = T3 * T4
    @test !(Tc2 isa AbstractOperators.Compose)
    @test norm(Tc2 * x - (x - d2 + d1)) < 1.0e-12

    # AffineAdd(linear) * AffineAdd(linear) — S1=false, S2=true
    T5 = AffineAdd(Eye(n), d1, false) # x - d1
    T6 = AffineAdd(Eye(n), d2, true)  # x + d2
    Tc3 = T5 * T6
    @test !(Tc3 isa AbstractOperators.Compose)
    @test norm(Tc3 * x - (x + d2 - d1)) < 1.0e-12

    # combine(linear_op, AffineAdd) with scalar displacement
    m = 6
    A = randn(m, n)
    scalar_d = 2.5
    opA = MatrixOp(A)
    opAA_scalar = AffineAdd(Eye(n), scalar_d)  # scalar displacement: x + 2.5
    Tc4 = opA * opAA_scalar                     # A * (x + 2.5) = A*x + A*fill(2.5,n)
    @test !(Tc4 isa AbstractOperators.Compose)
    @test norm(Tc4 * x - A * (x .+ scalar_d)) < 1.0e-12

    # combine(L, Sum) with non-square L
    n2, m2 = 5, 7
    A_outer = MatrixOp(randn(n2, m2))  # n2 × m2 (non-square)
    A1 = MatrixOp(randn(m2, n2))       # m2 × n2
    A2 = MatrixOp(randn(m2, n2))       # m2 × n2
    Sop = Sum(A1, A2)
    combined_sum = A_outer * Sop       # should combine; non-square → Sum(ops...) branch
    @test !(combined_sum isa AbstractOperators.Compose)
    x2 = randn(n2)
    y_ref = A_outer.A * (A1.A * x2 + A2.A * x2)
    @test norm(combined_sum * x2 - y_ref) < 1.0e-12

    # combine(Zeros, R) — is_null(L): square R with same types (returns L)
    Z_sq = Zeros(Float64, (n,), Float64, (n,))
    E_sq = Eye(n)
    ZE = Z_sq * E_sq
    @test is_null(ZE)
    @test size(ZE) == size(Z_sq)

    # combine(Zeros, R) — is_null(L): non-square R (creates new Zeros)
    # Z_rect: domain=(p,), codomain=(q,); A_rect must have codomain=(p,) to compose with Z_rect
    p, q = 4, 6
    Z_rect = Zeros(Float64, (p,), Float64, (q,))  # domain=(p,), codomain=(q,)
    A_rect = MatrixOp(randn(p, n))                 # domain=(n,), codomain=(p,)
    ZA = Z_rect * A_rect
    @test is_null(ZA)
    @test size(ZA, 1) == (q,)   # codomain of Z_rect
    @test size(ZA, 2) == (n,)   # domain of A_rect

    # combine(L, Zeros) — is_null(R): square L (returns R)
    EZ = E_sq * Z_sq
    @test is_null(EZ)
    @test size(EZ) == size(Z_sq)

    # combine(L, Zeros) — is_null(R): non-square L (creates new Zeros)
    A_ns = MatrixOp(randn(m2, n2))  # m2×n2 (non-square)
    Z_ns = Zeros(Float64, (n2,), Float64, (n2,))  # n2×n2
    AZ = A_ns * Z_ns
    @test is_null(AZ)
    @test size(AZ, 1) == (m2,)
    @test size(AZ, 2) == (n2,)
end

@testitem "Combinations: Scale+Compose forwarding branches" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(48)

    n, m = 4, 5
    d = randn(n)
    A = randn(n, m)
    d2 = randn(n)

    # combine(Scale, Compose): can_be_combined(L.A, R.A[end]) path (line 82)
    # Compose(DiagOp(d2), MatrixOp(A)) stores as A=(MatrixOp,DiagOp), so A[end] = DiagOp(d2)
    # can_be_combined(DiagOp(d), DiagOp(d2)) = true → forwarding branch
    inner_comp = DiagOp(d2) * MatrixOp(A)
    s_diag = Scale(2.0, DiagOp(d))
    combined_sc = s_diag * inner_comp
    x = randn(m)
    @test combined_sc * x ≈ s_diag * (inner_comp * x)

    # combine(Scale, MatrixOp): can_be_combined(T1.A, T2) = true path (line 199)
    # Scale(DiagOp) * MatrixOp — can_be_combined(DiagOp, MatrixOp) = true
    sm = Scale(3.0, DiagOp(d)) * MatrixOp(A)
    @test sm * x ≈ 3.0 * (d .* (A * x))

    # combine(Scale, AdjointMatrixOp): else branch (line 208)
    # Scale(FiniteDiff) * MatrixOp(n×n)' — can_be_combined(FiniteDiff, AdjointMatrixOp) = false
    A2 = randn(n, n)
    sf = Scale(2.0, FiniteDiff((n,))) * MatrixOp(A2)'
    xf = randn(n)
    @test sf * xf ≈ 2.0 * (FiniteDiff((n,)) * (MatrixOp(A2)' * xf))

    # combine(AdjointScale, DiagOp): can_be_combined forwarding branch (line 234)
    sd = Scale(2.0, DiagOp(d))
    adj_sd_diag = sd' * DiagOp(d2)
    xd = randn(n)
    @test adj_sd_diag * xd ≈ sd' * (DiagOp(d2) * xd)

    # combine(DiagOp, Scale) forwarding branch (line 250)
    ds = DiagOp(d) * Scale(2.0, DiagOp(d2))
    @test ds * xd ≈ DiagOp(d) * (2.0 * (DiagOp(d2) * xd))

    # combine(AdjointDiagOp, Scale) forwarding branch (line 258)
    ads = DiagOp(d)' * Scale(2.0, DiagOp(d2))
    @test ads * xd ≈ DiagOp(d)' * (2.0 * (DiagOp(d2) * xd))
end

@testitem "Combinations: AdjointScale and Compose*Scale forwarding branches" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(49)

    n = 4
    d, d2 = randn(n), randn(n)
    A = randn(n, n)

    # combine(AdjointScale, Compose): can_be_combined(Scale.A', Compose.A[end]) = true path (line 90)
    # Scale(DiagOp(d))' * Compose(DiagOp(d2), MatrixOp(A))
    # Compose stores as (MatrixOp, DiagOp), A[end] = DiagOp(d2)
    # can_be_combined(DiagOp(d)', DiagOp(d2)) = true → forwarding
    sc = Scale(2.0, DiagOp(d))
    inner_comp = DiagOp(d2) * MatrixOp(A)
    combined_adj = sc' * inner_comp
    x = randn(n)
    @test combined_adj * x ≈ sc' * (inner_comp * x)

    # combine(Compose, Scale): can_be_combined(Compose.A[1], Scale.A) = true path (lines 98-99)
    # Compose(MatrixOp(A), DiagOp(d)) stores as (DiagOp, MatrixOp), A[1] = DiagOp(d)
    # can_be_combined(DiagOp(d), DiagOp(d2)) = true → forwarding
    comp_sc = MatrixOp(A) * DiagOp(d)  # Compose(MatrixOp, DiagOp), stored (DiagOp, MatrixOp)
    combined_cs = comp_sc * Scale(2.0, DiagOp(d2))
    @test combined_cs * x ≈ comp_sc * (2.0 * (DiagOp(d2) * x))

    # combine(Compose, AdjointScale): can_be_combined(Compose.A[1], Scale.A') = true path (lines 109-110)
    # Same Compose(MatrixOp, DiagOp), Scale(DiagOp)' → can_be_combined(DiagOp, AdjointDiagOp) = true
    combined_cas = comp_sc * Scale(2.0, DiagOp(d2))'
    @test combined_cas * x ≈ comp_sc * (Scale(2.0, DiagOp(d2))' * x)
end


@testitem "Combinations: Scale+Compose else branches (lines 82, 90, 98-99, 109-110)" tags = [:calculus, :Combinations] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    using AbstractOperators: combine, can_be_combined
    Random.seed!(0)
    n = 5
    # Build a 2-op Compose that doesn't get simplified: FD(n) * MatrixOp(n-1,n-1)
    # A tuple order is (inner, outer) so comp.A = (FD(n), MatrixOp(n-1,n-1))
    comp = MatrixOp(randn(n-1, n-1)) * FiniteDiff((n,))   # domain (n,) → codomain (n-1,)

    # Line 82 else: combine(Scale, Compose) when can_be_combined(L.A, comp.A[end]) = false
    # but can_be_combined(Scale, comp) = true via the all-linear+MatrixOp condition
    L_82 = Scale(2.0, FiniteDiff((n-1,)))  # domain (n-1,) → codomain (n-2,)
    @test can_be_combined(L_82, comp)
    result_82 = combine(L_82, comp)
    x1 = randn(n)
    @test result_82 * x1 ≈ L_82 * (comp * x1)

    # Line 90 else: combine(AdjointScale, Compose) when can_be_combined(FD(n)', comp.A[end]) = false
    # Use MatrixOp(n-1,n-1) * FD(n,) so inner operator (FD') doesn't combine with outer (MatrixOp)
    comp2 = MatrixOp(randn(n-1, n-1)) * FiniteDiff((n,))  # domain (n,) → codomain (n-1,)
    adj_L = Scale(2.0, FiniteDiff((n,)))'   # domain (n-1,) → codomain (n,)
    @test can_be_combined(adj_L, comp2)
    result_90 = combine(adj_L, comp2)
    x2 = randn(n)
    @test result_90 * x2 ≈ adj_L * (comp2 * x2)

    # Lines 98-99 else: combine(Compose, Scale) when can_be_combined(comp.A[1]=FD(n), FD(n+1)) = false
    scale_inner = Scale(2.0, FiniteDiff((n+1,)))  # domain (n+1,) → codomain (n,)
    @test can_be_combined(comp, scale_inner)
    result_98 = combine(comp, scale_inner)
    x3 = randn(n+1)
    @test result_98 * x3 ≈ comp * (scale_inner * x3)

    # Lines 109-110 else: combine(Compose, AdjointScale) when can_be_combined(comp.A[1]=FD(n), FD(n)') = false
    adj_scale_inner = Scale(2.0, FiniteDiff((n,)))'  # domain (n-1,) → codomain (n,)
    @test can_be_combined(comp, adj_scale_inner)
    result_109 = combine(comp, adj_scale_inner)
    x4 = randn(n-1)
    @test result_109 * x4 ≈ comp * (adj_scale_inner * x4)
end

@testitem "Combinations (GPU)" tags = [:gpu, :calculus, :Combinations] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        m1, m2, m3, m4 = 4, 7, 3, 2
        A1 = gpu_randn(backend, m3, m1)
        A2 = gpu_randn(backend, m3, m2)
        A3 = gpu_randn(backend, m4, m3)
        opH = HCAT(MatrixOp(A1), MatrixOp(A2))
        opC = Compose(MatrixOp(A3), opH)
        x1, x2 = gpu_randn(backend, m1), gpu_randn(backend, m2)
        test_op(opC, ArrayPartition(x1, x2), gpu_randn(backend, m4), false)
    end
end
