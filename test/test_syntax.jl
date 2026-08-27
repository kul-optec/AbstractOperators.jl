@testitem "Syntax: Adjoint" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n, m = 5, 3
    A = randn(n, m)
    x2 = randn(n)
    opA = MatrixOp(A)

    opt = opA'
    y1 = opt * x2
    y2 = A' * x2
    @test norm(y1 - y2) < 1.0e-9
end

@testitem "Syntax: Addition and Subtraction" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n, m = 5, 3
    A = randn(n, m)
    B = randn(n, m)
    C = randn(n, m)
    x1 = randn(m)
    opA = MatrixOp(A)
    opB = MatrixOp(B)
    opC = MatrixOp(C)

    opp = +opA
    y1 = opp * x1
    y2 = A * x1
    @test norm(y1 - y2) < 1.0e-9

    opm = -opA
    y1 = opm * x1
    y2 = -A * x1
    @test norm(y1 - y2) < 1.0e-9

    ops = opA + opB
    y1 = ops * x1
    y2 = A * x1 + B * x1
    @test norm(y1 - y2) < 1.0e-9

    ops = opA - opB
    y1 = ops * x1
    y2 = A * x1 - B * x1
    @test norm(y1 - y2) < 1.0e-9

    ops = opA + opB
    opss = ops + opC
    y1 = opss * x1
    y2 = A * x1 + B * x1 + C * x1
    @test norm(y1 - y2) < 1.0e-9

    ops = opB + opC
    opss = opA - ops
    y1 = opss * x1
    y2 = A * x1 - B * x1 - C * x1
    @test norm(y1 - y2) < 1.0e-9

    ops = opB + opC
    opss = ops - opA
    y1 = opss * x1
    y2 = -A * x1 + B * x1 + C * x1
    @test norm(y1 - y2) < 1.0e-9
end

@testitem "Syntax: Multiplication" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n, m, l = 5, 3, 7
    A = randn(n, m)
    B = randn(l, n)
    x1 = randn(m)
    opA = MatrixOp(A)
    opB = MatrixOp(B)
    Id = Eye(m)
    alpha = pi
    beta = 4

    ops = alpha * opA
    y1 = ops * x1
    y2 = alpha * A * x1
    @test norm(y1 - y2) < 1.0e-9

    ops = alpha * opA
    opss = beta * ops
    y1 = opss * x1
    y2 = beta * alpha * A * x1
    @test norm(y1 - y2) < 1.0e-9

    ops1 = alpha * opA
    ops2 = beta * opB
    opss = ops2 * ops1
    y1 = opss * x1
    y2 = beta * B * alpha * A * x1
    @test norm(y1 - y2) < 1.0e-9

    opc = opB * opA
    y1 = opc * x1
    y2 = B * A * x1
    @test norm(y1 - y2) < 1.0e-9

    opc = Id * opA
    y1 = opc * x1
    y2 = A * x1
    @test norm(y1 - y2) < 1.0e-9

    opc = opA * Id
    y1 = opc * x1
    y2 = A * x1
    @test norm(y1 - y2) < 1.0e-9

    opc = Id * Id
    y1 = opc * x1
    y2 = x1
    @test norm(y1 - y2) < 1.0e-9
end

@testitem "Syntax: Getindex basic" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n, m, l = 5, 3, 7
    A = randn(n, m)
    B = randn(l, n)
    x1 = randn(m)
    opA = MatrixOp(A)
    opB = MatrixOp(B)

    ops = opA[1:(n - 1)]
    y1 = ops * x1
    y2 = (A * x1)[1:(n - 1)]
    @test norm(y1 - y2) < 1.0e-9

    d = randn(n, m, l)
    opF = DiagOp(d)
    x3 = randn(n, m, l)
    ops = opF[1:(n - 1), :, 2:l]
    y1 = ops * x3
    y2 = (d .* x3)[1:(n - 1), :, 2:l]
    @test norm(y1 - y2) < 1.0e-9

    opF = DiagOp(d)
    ops = opF[1:(n - 1), 2:m, 1]
    y1 = ops * x3
    y2 = (d .* x3)[1:(n - 1), 2:m, 1]
    @test norm(y1 - y2) < 1.0e-9

    opV = Variation(n, m, l)
    ops = opV[1:4]
    y1 = ops * x3
    y2 = (opV * x3)[1:4]
    @test norm(y1 - y2) < 1.0e-9

    ops = (opB * opA)[1:(l - 1)]
    y1 = ops * x1
    y2 = (B * A * x1)[1:(l - 1)]
    @test norm(y1 - y2) < 1.0e-9

    ops = (10.0 * opA)[1:(n - 1)]
    y1 = ops * x1
    y2 = (10 * A * x1)[1:(n - 1)]
    @test norm(y1 - y2) < 1.0e-9

    # Compose getindex with diagonal intermediate operators
    dtmp = randn(n)
    comp_diag = MatrixOp(randn(n, n)) * DiagOp(dtmp)
    sliced_diag = comp_diag[1:3]
    xdiag = randn(n)
    @test sliced_diag * xdiag ≈ (comp_diag * xdiag)[1:3]
end

@testitem "Syntax: Getindex HCAT" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n, m1, m2, m3 = 5, 6, 7, 8
    A = randn(n, m1)
    B = randn(n, m2)
    C = randn(n, m3)
    x1 = randn(m1)
    x2 = randn(m2)
    x3 = randn(m3)
    opA = MatrixOp(A)
    opB = MatrixOp(B)
    opC = MatrixOp(C)
    opH = HCAT(opA, opB, opC)

    opH2 = opH[1:2]
    y1 = opH2 * ArrayPartition(x1, x2)
    y2 = A * x1 + B * x2
    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    opH3 = opH[3]
    y1 = opH3 * x3
    y2 = C * x3
    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    opHperm = opH[[3, 1, 2]]
    @test norm(opH * ArrayPartition(x1, x2, x3) - opHperm * ArrayPartition(x3, x1, x2)) < 1.0e-12

    @test opHperm[1] == opC
    @test opHperm[2] == opA
    @test opHperm[3] == opB

    opHperm = opH[[3, 1]]
    @test norm(opC * x3 + opA * x1 - opHperm * ArrayPartition(x3, x1)) < 1.0e-12

    # slicing Affine add of HCAT
    d = randn(n)
    opHA = AffineAdd(opH, d)
    @test norm(opHA[1] * x1 - (A * x1 + d)) < 1.0e-9
    @test norm(opHA[2] * x2 - (B * x2 + d)) < 1.0e-9
    @test norm(opHA[3] * x3 - (C * x3 + d)) < 1.0e-9

    m4 = 9
    D = randn(n, n)
    E = randn(n, m4)
    opD = MatrixOp(D)
    opE = MatrixOp(E)
    opCH = opD * opH
    opHCH = HCAT(opCH, opE)
    opH4 = opHCH[4]
    @test opH4 == opE
    @test_throws ErrorException opHCH[1]
    @test_throws ErrorException opHCH[1:2]

    # HCAT getindex split-error path with stacked indices
    h_joint = HCAT(MatrixOp(randn(4, 2)), MatrixOp(randn(4, 3)))
    h_stacked = HCAT((h_joint, MatrixOp(randn(4, 5))), zeros(4))
    @test_throws ErrorException h_stacked[1]

    # AffineAdd{HCAT} ndoms==1 branch via low-level single-operator HCAT
    Hsingle = HCAT((MatrixOp(randn(4, 4)),), (1,), zeros(4))
    AHsingle = AffineAdd(Hsingle, randn(4))
    @test AHsingle[1:2] isa AbstractOperator

    # Compose getindex branch for diagonal tail (ndoms(A,2)>1)
    Hbase = HCAT((MatrixOp(randn(4, 2)), MatrixOp(randn(4, 3))), zeros(4))
    Cdiag = DiagOp(randn(4)) * Hbase
    xin = ArrayPartition(randn(2), randn(3))
    @test Cdiag[[2, 1]] * ArrayPartition(xin.x[2], xin.x[1]) ≈ Cdiag * xin

    # Compose getindex error path for non-diagonal tail
    Cnondiag = MatrixOp(randn(4, 4)) * Hbase
    @test_throws ErrorException Cnondiag[[2, 1]]
end

@testitem "Syntax: Getindex VCAT" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n1, n2, n3, m = 5, 6, 7, 8
    A = randn(n1, m)
    B = randn(n2, m)
    C = randn(n3, m)
    x1 = randn(m)
    x3 = randn(m)
    opA = MatrixOp(A)
    opB = MatrixOp(B)
    opC = MatrixOp(C)
    opV = VCAT(opA, opB, opC)

    opV2 = opV[1:2]
    y1 = opV2 * x1
    y2 = ArrayPartition(A * x1, B * x1)
    @test norm(y1 - y2) <= 1.0e-12

    opV3 = opV[3]
    y1 = opV3 * x3
    y2 = C * x3
    @test norm(y1 - y2) <= 1.0e-12

    # VCAT full-index permutation branch
    opVperm = opV[[3, 1, 2]]
    @test opVperm * x1 == ArrayPartition((opV * x1).x[[3, 1, 2]]...)

    # Generic getindex error path for multi-domain operators without split support
    opD_err = DCAT(MatrixOp(randn(2, 2)), MatrixOp(randn(3, 2)))
    @test_throws ErrorException opD_err[[1]]

    # Compose getindex branch for ndoms(A,2)==1
    comp_single = FiniteDiff((m,)) * MatrixOp(randn(m, m))
    x_comp = randn(m)
    @test comp_single[1:3] * x_comp ≈ (comp_single * x_comp)[1:3]
end

@testitem "Syntax: Check utility" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    op_diag = DiagOp(rand(3))
    ygood = zeros(3)
    xgood = rand(3)
    @test_throws ArgumentError AbstractOperators.check(ygood, op_diag, "bad")
    @test_throws ArgumentError AbstractOperators.check("bad", op_diag, xgood)
    @test_throws ArgumentError AbstractOperators.check(ygood, op_diag, ArrayPartition(rand(3), rand(3)))
    @test_throws ArgumentError AbstractOperators.check(ArrayPartition(zeros(3), zeros(3)), op_diag, xgood)
end

@testitem "Syntax: HCAT and VCAT construction" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n, m1, m2 = 5, 6, 7
    A = randn(n, m1)
    B = randn(n, m2)
    opA = MatrixOp(A)
    opB = MatrixOp(B)
    opH = [opA opB]
    x1 = randn(m1)
    x2 = randn(m2)
    y1 = opH * ArrayPartition(x1, x2)
    y2 = [A B] * [x1; x2]
    @test norm(y1 - y2) <= 1.0e-12

    opHH = [opH opB]
    y1 = opHH * ArrayPartition(x1, x2, x2)
    y2 = [A B B] * [x1; x2; x2]
    @test norm(y1 - y2) <= 1.0e-12

    n, m1, m2 = 5, 6, 7
    A = randn(m1, n)
    B = randn(m2, n)
    opA = MatrixOp(A)
    opB = MatrixOp(B)
    opH = [opA; opB]
    x1 = randn(n)
    y1 = opH * x1
    y2 = ArrayPartition(A * x1, B * x1)
    @test norm(y1 - y2) <= 1.0e-12

    opVV = [opA; opH]
    y1 = opVV * x1
    y2 = ArrayPartition(A * x1, A * x1, B * x1)
    @test norm(y1 - y2) <= 1.0e-12
end

@testitem "Syntax: Reshape ndims ndoms" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n, m = 10, 5
    A = randn(n, m)
    x1 = randn(m)
    opA = MatrixOp(A)
    opR = reshape(opA, 2, 5)
    opR = reshape(opA, (2, 5))
    y1 = opR * x1
    y2 = reshape(A * x1, 2, 5)
    @test norm(y1 - y2) <= 1.0e-12

    L = Variation((3, 4, 5))
    @test ndims(L) == (2, 3)
    @test ndims(L, 1) == 2
    @test ndims(L, 2) == 3
    @test ndoms(L) == (1, 1)
    H = hcat(L, L)
    @test ndims(H) == (2, (3, 3))
    @test ndims(H, 1) == 2
    @test ndims(H, 2) == (3, 3)
    @test ndoms(H) == (1, 2)
    @test ndoms(H, 1) == 1
    @test ndoms(H, 2) == 2
    D = DCAT(L, L)
    @test ndims(D) == ((2, 2), (3, 3))
    @test ndoms(D) == (2, 2)
end

@testitem "Syntax: Jacobian convert displacement" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n, m = 10, 5
    A = MatrixOp(randn(n, m))
    B = Sigmoid(Float64, (n,), 100.0)
    op = B * A
    J = jacobian(op, randn(m))

    L = Eye(10)
    LL = convert(AbstractOperator, Float64, (10,), L)
    @test LL == L

    LL = convert(LinearOperator, Float64, (10,), L)
    @test LL == L

    @test_throws MethodError convert(NonLinearOperator, Float64, (10,), L)

    L = Eye(10)
    @test displacement(L) == 0.0
end

@testitem "Syntax: VCAT and HCAT with Tuple input" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators, RecursiveArrayTools
    n1, n2, m = 3, 4, 5
    x1 = randn(n1)
    x2 = randn(n2)

    # HCAT * Tuple  (returns codomain array, not ArrayPartition)
    opH = HCAT(MatrixOp(randn(m, n1)), MatrixOp(randn(m, n2)))
    yH = opH * (x1, x2)
    @test yH ≈ opH * ArrayPartition(x1, x2)

    # VCAT * Tuple (multi-domain VCAT — each sub-op is an HCAT, so domain is a tuple)
    # opV has domain (n1,)×(n2,) and codomain (m,)×(m,); input must be a Tuple
    opH1 = HCAT(MatrixOp(randn(m, n1)), MatrixOp(randn(m, n2)))
    opH2 = HCAT(MatrixOp(randn(m, n1)), MatrixOp(randn(m, n2)))
    opV = VCAT(opH1, opH2)
    yV = opV * (x1, x2)          # calls *(L::VCAT, b::Tuple) → returns y.x as Tuple
    @test yV isa Tuple
    ref = opV * ArrayPartition(x1, x2)
    @test all(collect(yV[i]) ≈ ref.x[i] for i in eachindex(yV))
end

@testitem "Syntax: L * coeff (operator-right scalar mul)" tags = [:misc, :Syntax] setup = [
    TestUtils,
] begin
    using AbstractOperators
    n = 4
    op = FiniteDiff(Float64, (n,), 1)
    alpha = 3.0
    s1 = op * alpha
    s2 = alpha * op
    x = randn(n)
    @test s1 * x ≈ s2 * x
end

@testitem "Syntax: mul! fallback throws MethodError" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    # Custom operator with no mul! defined so the fallback in syntax.jl fires
    struct _NoMulOp <: AbstractOperators.LinearOperator end
    AbstractOperators.size(::_NoMulOp) = ((3,), (3,))
    AbstractOperators.domain_type(::_NoMulOp) = Float64
    AbstractOperators.codomain_type(::_NoMulOp) = Float64
    op = _NoMulOp()
    @test_throws MethodError mul!(zeros(3), op, zeros(3))
end

@testitem "Syntax: domain/codomain_array_type generic fallback for Tuple domain_type" tags = [
    :misc, :Syntax,
] setup = [TestUtils] begin
    using AbstractOperators
    # Custom operator that relies on the generic AbstractOperator fallback for
    # domain_array_type/codomain_array_type (no override), with a Tuple-valued
    # domain_type/codomain_type — exercises _storage_type_for_elem(::Tuple).
    struct _TupleDomainOp <: AbstractOperators.LinearOperator end
    AbstractOperators.size(::_TupleDomainOp) = (((3,), (3,)), ((3,), (3,)))
    AbstractOperators.domain_type(::_TupleDomainOp) = (Float64, Float64)
    AbstractOperators.codomain_type(::_TupleDomainOp) = (Float64, Float64)
    op = _TupleDomainOp()
    ds = domain_array_type(op)
    cs = codomain_array_type(op)
    @test ds <: AbstractOperators.ArrayPartition{Float64, Tuple{Array{Float64}, Array{Float64}}}
    @test cs <: AbstractOperators.ArrayPartition{Float64, Tuple{Array{Float64}, Array{Float64}}}
end

@testitem "Syntax: Sum getindex (multi-domain)" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n = 5
    A = MatrixOp(randn(n, n))
    B = DiagOp(randn(n))
    S = Sum(A, B)
    sliced = S[2:(n - 1)]
    x = randn(n)
    @test sliced * x ≈ (S * x)[2:(n - 1)]

    # Sum with multi-domain input (ndoms > 1 → iterates over A.A)
    # getindex on multi-domain Sum selects domain sub-operators (not codomain rows)
    n2, m1, m2 = 4, 2, 3
    Hbase = HCAT(MatrixOp(randn(n2, m1)), MatrixOp(randn(n2, m2)))
    S_md = Sum(Hbase, Hbase)
    sliced_md = S_md[1]   # selects first sub-op (domain = m1) from each HCAT member
    x1 = randn(m1)
    @test sliced_md * x1 ≈ (Hbase[1] + Hbase[1]) * x1
end

@testitem "Syntax: Scale getindex (ndoms == 1 branch)" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    n = 5
    op = FiniteDiff(Float64, (n,), 1)   # ndoms(Scale(coeff, op), 2) == 1
    s = Scale(2.0, op)
    sliced = s[1:2]
    x = randn(n)
    @test sliced * x ≈ (s * x)[1:2]

    # Scale wrapping multi-domain operator
    # getindex on multi-domain Scale selects domain sub-operators (not codomain rows)
    n2, m1, m2 = 4, 2, 3
    Hbase = HCAT(MatrixOp(randn(n2, m1)), MatrixOp(randn(n2, m2)))
    sc_md = Scale(2.0, Hbase)
    sc_sliced = sc_md[1]   # selects first sub-op (domain = m1)
    x1 = randn(m1)
    @test sc_sliced * x1 ≈ 2.0 .* (Hbase[1] * x1)
end

@testitem "Syntax: check domain/codomain ArrayPartition errors" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators, RecursiveArrayTools
    n, m1, m2 = 4, 2, 3
    op_multi_in = HCAT(MatrixOp(randn(n, m1)), MatrixOp(randn(n, m2)))
    y = zeros(n)
    # multi-domain op with non-ArrayPartition input
    @test_throws ArgumentError AbstractOperators.check(y, op_multi_in, randn(m1))

    # multi-codomain op with non-ArrayPartition output
    op_multi_out = DCAT(MatrixOp(randn(m1, m1)), MatrixOp(randn(m2, m2)))
    @test_throws ArgumentError AbstractOperators.check(
        randn(m1), op_multi_out, ArrayPartition(randn(m1), randn(m2))
    )
end

@testitem "Syntax: Scale complex coeff on real AdjointMatrixOp errors" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators
    # real-codomain adjoint MatrixOp scaled by complex scalar
    n = 4
    op = MatrixOp(randn(n, n))'
    @test_throws ErrorException Scale(1.0im, op)
end

@testitem "Syntax: Compose getindex with multi-domain errors (line 62)" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using AbstractOperators, RecursiveArrayTools
    # Compose with ndoms>1 cannot be split (error branch)
    # diagonal * HCAT always simplifies to HCAT via combination rules,
    # so any Compose with ndoms>1 has a non-diagonal tail and hits the error.
    n = 4
    M = MatrixOp(randn(n, n))
    H = HCAT(DiagOp(randn(n)), DiagOp(randn(n)))
    C = M * H   # non-diagonal outer: Compose with ndoms>1 and non-diagonal tail
    @test ndoms(C, 2) == 2
    @test_throws ErrorException C[1:2]
end

@testitem "copy_operator: fast path, slow path, default fallback" tags = [:misc, :Syntax] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(8)

    n = 5
    A = randn(n, n)
    op = MatrixOp(A)
    @test is_thread_safe(op) == true

    # Fast path: thread-safe operator, no kwargs -> same object shared
    op_shared = copy_operator(op)
    @test op_shared === op

    # Slow path: explicit kwarg forces the default (deepcopy) _copy_operator_impl fallback
    op_copy = copy_operator(op; threaded = true)
    @test op_copy isa MatrixOp
    x = randn(n)
    @test op_copy * x ≈ op * x

end
