@testitem "Compose: basic mul" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    m1, m2 = 4, 7
    A1 = randn(m2, m1)
    opA1 = MatrixOp(A1)
    opF = FiniteDiff((m2,))
    opC = Compose(opF, opA1)
    x = randn(m1)
    y1 = test_op(opC, x, randn(m2 - 1), verb)
    @test y1 == diff(A1 * x)

    m1, m2, m3 = 4, 7, 3
    A1 = randn(m2, m1)
    A2 = randn(m3, m2 - 1)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opC1 = Compose(opA2, Compose(FiniteDiff((m2,)), opA1))
    opC2 = Compose(Compose(opA2, FiniteDiff((m2,))), opA1)
    x = randn(m1)
    y1 = test_op(opC1, x, randn(m3), verb)
    y2 = test_op(opC2, x, randn(m3), verb)
    @test all(norm.(y1 .- A2 * diff(A1 * x)) .<= 1.0e-12)
    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    @test typeof(opA1 * Eye(m1)) == typeof(opA1)
    @test typeof(Eye(m2) * opA1) == typeof(opA1)
    @test typeof(Eye(m2) * Eye(m2)) == typeof(Eye(m2))
    @test_throws Exception Compose(MatrixOp(randn(5, 4)), MatrixOp(randn(3, 2)))
end

@testitem "Compose: properties" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    m1, m2, m3 = 4, 7, 3
    A1 = randn(m2, m1)
    A2 = randn(m3, m2 - 1)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opF = FiniteDiff((m2,))
    opC = Compose(opF, opA1)
    opC1 = Compose(opA2, Compose(opF, opA1))
    @test is_sliced(opC) == false
    @test is_linear(opC1) == true
    @test is_null(opC1) == false
    @test is_eye(opC1) == false
    @test is_diagonal(opC1) == false
    @test is_orthogonal(opC1) == false
    @test is_invertible(opC1) == false

    d = randn(5)
    opC2 = DiagOp(d) * GetIndex((10,), 1:5)
    @test is_sliced(opC2) == true
    @test is_diagonal(opC2) == true
    @test diag(opC2) == d

    Z = Zeros(Float64, (m2,), Float64, (m2 - 1,))
    ZC = Compose(opA2, Z)
    @test is_null(ZC)
end

@testitem "Compose: displacement" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, m2, m3, m4, m5 = 4, 7, 3, 2, 11
    A1 = randn(m2, m1)
    A2 = randn(m3, m2)
    A3 = randn(m4, m3)
    A4 = randn(m5, m4)
    d1 = randn(m2)
    d2 = pi
    d3 = 0.0
    d4 = randn(m5)
    opA1 = AffineAdd(MatrixOp(A1), d1)
    opA2 = AffineAdd(MatrixOp(A2), d2)
    opA3 = MatrixOp(A3)
    opA4 = AffineAdd(MatrixOp(A4), d4, false)
    opC = Compose(Compose(Compose(opA4, opA3), opA2), opA1)
    x = randn(m1)
    @test norm(opC * x - (A4 * (A3 * (A2 * (A1 * x + d1) .+ d2) .+ d3) - d4)) < 1.0e-9
    @test norm(displacement(opC) - (A4 * (A3 * (A2 * d1 .+ d2) .+ d3) - d4)) < 1.0e-9
    @test norm(remove_displacement(opC) * x - (A4 * (A3 * (A2 * (A1 * x))))) < 1.0e-9
end

@testitem "Compose: nonlinear" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    l, n, m = 5, 4, 3
    x = randn(m)
    r = randn(l)
    A = randn(l, n)
    C = randn(n, m)
    opA = MatrixOp(A)
    opB = Sigmoid(Float64, (n,), 2)
    opC = MatrixOp(C)
    op = Compose(opA, Compose(opB, opC))
    y, grad = test_NLop(op, x, r, verb)
    @test norm(A * (opB * (opC * x)) - y) < 1.0e-8
end

@testitem "Compose internal buffer mismatch" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = MatrixOp(randn(3, 3))
    B = MatrixOp(randn(3, 3))
    @test_throws DimensionMismatch AbstractOperators.Compose((B, A), ())
end

@testitem "Compose: type mismatch error" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using AbstractOperators
    # Compose L1 ∘ L2 where domain_type(L1) ≠ codomain_type(L2)
    L1 = MatrixOp(randn(Float64, 3, 4))
    L2 = MatrixOp(randn(ComplexF64, 4, 5))
    @test_throws DomainError Compose(L1, L2)
end

@testitem "Compose: L'*L shortcut to get_normal_op" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    # DiagOp has has_optimized_normalop = true
    # L' * L should short-circuit through line 118-119 in Compose(L1, L2)
    d = randn(4)
    op = DiagOp(d)
    normal = op' * op
    x = randn(4)
    @test normal * x ≈ op' * (op * x)
end

@testitem "get_normal_op(Compose): optimized if branch" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    # MatrixOp as outer/left operator has has_optimized_normalop = true
    # L.A[end] = MatrixOp → hits the if branch in get_normal_op(L::Compose)
    A = randn(4, 5)
    L = Compose(MatrixOp(A), FiniteDiff((6,)))
    N = AbstractOperators.get_normal_op(L)
    x = randn(6)
    @test N isa AbstractOperator
    @test N * x ≈ L' * (L * x)
end

@testitem "Scale(1, Compose) returns Compose unchanged" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    A = randn(4, 5)
    d = randn(4)
    comp = Compose(DiagOp(d), MatrixOp(A))
    s1 = Scale(1.0, comp)
    @test s1 === comp
end

@testitem "Adjacent adjoint optimized normal (GetIndex*GetIndex')" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    G = GetIndex((5,), 2:4)
    L = AbstractOperators.Compose((G, G'), (randn(size(G, 1)),))
    x = randn(5)
    @test L * x == G' * (G * x)
    @test is_diagonal(L)
end

@testitem "Combine branch producing nested Compose is inlined" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    M = MatrixOp(randn(3, 3))
    S = Scale(2.0, Eye(3))
    L = AbstractOperators.Compose((S, M), (zeros(3),))
    @test L * randn(3) isa AbstractVector
    x = randn(3)
    @test L * x ≈ 2.0 * (M * x)
    @test L isa AbstractOperator
end

@testitem "remove_slicing first op GetIndex" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    G = GetIndex((5,), 2:4)
    A2 = FiniteDiff((3,))
    A3 = MatrixOp(randn(2, 2))
    L = Compose(A3, Compose(A2, G))
    L2 = AbstractOperators.remove_slicing(L)
    @test !is_sliced(L2)
    @test size(L2, 2) == (3,)
    v = randn(3)
    xfull = zeros(5)
    xfull[2:4] .= v
    @test L * xfull ≈ L2 * v
end

@testitem "remove_slicing first op Scale(GetIndex)" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    G = GetIndex((5,), 2:4)
    S = 3.0 * G
    A2 = FiniteDiff((3,))
    L = Compose(A2, S)
    L2 = AbstractOperators.remove_slicing(L)
    @test !is_sliced(L2)
    @test size(L2, 2) == (3,)
    v = randn(3)
    xfull = zeros(5)
    xfull[2:4] .= v
    @test L * xfull ≈ L2 * v
    A3 = MatrixOp(randn(2, 2))
    L3 = Compose(A3, L)
    L4 = AbstractOperators.remove_slicing(L3)
    @test !is_sliced(L4)
    @test size(L4, 2) == (3,)
    @test L3 * xfull ≈ L4 * v
end

@testitem "remove_slicing error path first not sliced" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    op1 = MyLinOp(Float64, (3,), Float64, (3,), (y, x) -> (y .= x), (y, x) -> (y .= x))
    op2 = MatrixOp(randn(2, 3))
    L = AbstractOperators.Compose((op1, op2), (zeros(3),))
    @test_throws ArgumentError AbstractOperators.remove_slicing(L)
end

@testitem "diag_AAc on Compose ok and error" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    d = randn(5)
    sel = 1:3
    L_ok = Compose(DiagOp(d[sel]), GetIndex((length(d),), sel))
    @test is_AAc_diagonal(L_ok)
    @test AbstractOperators.diag_AAc(L_ok) == diag_AAc(DiagOp(d[sel]))
    L_bad = Compose(MatrixOp(randn(3, 3)), MatrixOp(randn(3, 3)))
    @test !is_AAc_diagonal(L_bad)
    @test_throws ErrorException AbstractOperators.diag_AAc(L_bad)
end

@testitem "get_normal_op on Compose fallback path" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    L = Compose(FiniteDiff((3,)), MatrixOp(randn(3, 3)))
    N = AbstractOperators.get_normal_op(L)
    x = randn(3)
    @test N isa AbstractOperator
    @test N * x ≈ L' * (L * x)
end

@testitem "Scale(coeff, L::Compose) specialized paths" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    Llin = FiniteDiff((3,)) * MatrixOp(randn(3, 3)) * FiniteDiff((4,))
    Slin = Scale(1.7, Llin)
    x = randn(4)
    @test Slin isa Compose
    @test Slin * x ≈ 1.7 * (Llin * x)
    Lnl = Compose(FiniteDiff((3,)), Sigmoid(Float64, (3,), 2))
    Snl = Scale(2.0, Lnl)
    x2 = randn(3)
    @test Snl isa Scale
    @test Snl * x2 ≈ 2.0 * (Lnl * x2)
end

@testitem "get_normal_op(Compose) else branch" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    F1 = FiniteDiff((5,))
    F2 = FiniteDiff((6,))
    L = Compose(F1, F2)
    N = AbstractOperators.get_normal_op(L)
    x = randn(6)
    @test N isa AbstractOperator
    @test N * x ≈ L' * (L * x)
end

@testitem "Buffer reuse in 4-operator chain" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    F1 = FiniteDiff((10,))
    F2 = MatrixOp(rand(10, 9))
    F3 = FiniteDiff((10,))
    F4 = MatrixOp(rand(10, 10))
    L = F1 * F2 * F3 * F4
    @test L isa Compose
    @test length(L * randn(10)) == 9
    @test length(L' * randn(9)) == 10
    @test L.buf[1] === L.buf[3]
end

@testitem "DEBUG_COMPOSE logging branches" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    old_debug = AbstractOperators.DEBUG_COMPOSE[]
    try
        AbstractOperators.DEBUG_COMPOSE[] = true
        original_stdout = stdout
        (read_pipe, write_pipe) = redirect_stdout()
        G = GetIndex((5,), 2:4)
        _ = AbstractOperators.Compose((G, G'), (randn(3),))
        D1 = DiagOp(randn(3))
        D2 = DiagOp(randn(3))
        _ = AbstractOperators.Compose((D1, D2), (randn(3),))
        redirect_stdout(original_stdout)
        close(write_pipe)
        log_str = read(read_pipe, String)
        @test occursin("Replacing", log_str) || occursin("Combining", log_str)
    finally
        AbstractOperators.DEBUG_COMPOSE[] = old_debug
    end
end

@testitem "Triple combination path" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    struct TripleCombTestOp <: LinearOperator end
    LinearAlgebra.size(::TripleCombTestOp) = ((5,), (5,))
    AbstractOperators.domain_type(::TripleCombTestOp) = Float64
    AbstractOperators.codomain_type(::TripleCombTestOp) = Float64
    AbstractOperators.fun_name(::TripleCombTestOp) = "TCT"
    AbstractOperators.can_be_combined(::TripleCombTestOp, ::TripleCombTestOp, ::TripleCombTestOp) = true
    AbstractOperators.combine(::TripleCombTestOp, ::TripleCombTestOp, ::TripleCombTestOp) = DiagOp(3 .* ones(5))
    AbstractOperators.can_be_combined(::FiniteDiff, ::DiagOp, ::TripleCombTestOp) = true
    AbstractOperators.combine(L1::FiniteDiff, L2::DiagOp, ::TripleCombTestOp) = L1 * L2
    AbstractOperators.can_be_combined(::TripleCombTestOp, ::DiagOp, ::FiniteDiff) = true
    AbstractOperators.combine(::TripleCombTestOp, L1::DiagOp, L2::FiniteDiff) = L1 * L2

    T1, T2, T3 = TripleCombTestOp(), TripleCombTestOp(), TripleCombTestOp()
    D1, D2 = DiagOp(randn(5)), DiagOp(randn(4))
    F1, F2 = FiniteDiff((5,)), FiniteDiff((6,))
    old_debug = AbstractOperators.DEBUG_COMPOSE[]
    try
        AbstractOperators.DEBUG_COMPOSE[] = true
        original_stdout = stdout
        (read_pipe, write_pipe) = redirect_stdout()

        L = T1 * T2 * T3
        @test L isa DiagOp
        @test diag(L) == 3 .* ones(5)
        L = F1 * T1 * T2 * T3
        x = randn(5)
        @test L isa Compose
        @test L * x ≈ F1 * (3 .* x)
        L = T1 * T2 * T3 * F2
        x = randn(6)
        @test L isa Compose
        @test L * x ≈ 3 .* (F2 * x)
        L = F1 * D1 * T1
        x = randn(5)
        @test L isa Compose
        @test length(L.A) == 2
        @test L * x ≈ F1 * (D1 * x)
        L = F1 * D1 * T1 * F2
        x = randn(6)
        @test L isa Compose
        @test length(L.A) == 3
        @test L * x ≈ F1 * (D1 * (F2 * x))
        L = D2 * F1 * D1 * T1
        x = randn(5)
        @test L isa Compose
        @test length(L.A) == 3
        @test L * x ≈ D2 * (F1 * (D1 * x))
        L = D2 * F1 * D1 * T1 * F2
        x = randn(6)
        @test L isa Compose
        @test length(L.A) == 4
        @test L * x ≈ D2 * (F1 * (D1 * (F2 * x)))
        L = F1 * D1 * T1 * D1
        x = randn(5)
        @test L isa Compose
        @test length(L.A) == 2
        @test L * x ≈ F1 * ((diag(D1) .^ 2) .* x)
        L = D1 * T1 * D1 * F2
        x = randn(6)
        @test L isa Compose
        @test length(L.A) == 2
        @test L * x ≈ ((diag(D1) .^ 2) .* (F2 * x))

        redirect_stdout(original_stdout)
        close(write_pipe)
        log_str = read(read_pipe, String)
        @test occursin("Replacing", log_str) || occursin("Combining", log_str)
    finally
        AbstractOperators.DEBUG_COMPOSE[] = old_debug
    end
end

@testitem "get_normal_op(Compose) if branch" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    # get_normal_op lines 227-229: has_optimized_normalop(L.A[end]) == true
    # Compose(MatrixOp, FiniteDiff) stores A = (FiniteDiff, MatrixOp); A[end] = MatrixOp
    L = Compose(MatrixOp(randn(5, 4)), FiniteDiff((5,)))
    @test AbstractOperators.has_optimized_normalop(L) == true
    N = AbstractOperators.get_normal_op(L)
    @test N isa AbstractOperator
    x = randn(5)
    @test N * x ≈ L' * (L * x)
end

@testitem "copy_operator on Compose" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    # _copy_operator_impl lines 307-309: use FiniteDiff+MatrixOp (no combine rule → stays Compose)
    A = randn(4, 4)
    L = Compose(FiniteDiff((4,)), MatrixOp(A))
    @test L isa Compose
    L2 = copy_operator(L)
    @test L2 isa Compose
    @test length(L2.A) == length(L.A)
    x = randn(4)
    @test L2 * x ≈ L * x
    y = randn(3)
    @test L2' * y ≈ L' * y
end

@testitem "remove_slicing Compose: is_eye path" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    # is_sliced(L.A[1]) && is_eye(remove_slicing(L.A[1])) branch (lines 267-273)
    # AffineAdd(GetIndex): is_sliced=true, not isa GetIndex, remove_slicing returns Eye
    G = GetIndex((5,), 2:4)
    b = randn(3)
    af = AffineAdd(G, b)
    @test AbstractOperators.is_sliced(af)
    @test !(af isa GetIndex)
    @test AbstractOperators.is_eye(AbstractOperators.remove_slicing(af))
    # length == 2 case: Compose((af, A2), buf) → remove_slicing returns A2
    A2 = MatrixOp(randn(3, 3))
    L2 = AbstractOperators.Compose((af, A2), (zeros(3),))
    out = AbstractOperators.remove_slicing(L2)
    @test out === A2
    # length > 2 case: Compose((af, fd, M), bufs) → remove_slicing returns Compose(fd, M)
    # use FiniteDiff + MatrixOp (no combine rule → stays Compose after remove_slicing)
    fd = FiniteDiff((3,))
    A3 = MatrixOp(randn(4, 3))
    L3 = AbstractOperators.Compose((af, fd, A3), (zeros(3), zeros(3)))
    out3 = AbstractOperators.remove_slicing(L3)
    @test out3 isa Compose
    @test length(out3.A) == 2
end

@testitem "Compose (GPU)" tags = [:gpu, :calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        m1, m2 = 4, 7
        A1 = gpu_randn(backend, m2, m1)
        opC = Compose(FiniteDiff(gpu_zeros(backend, Float64, m2)), MatrixOp(A1))
        test_op(opC, gpu_randn(backend, m1), gpu_randn(backend, m2 - 1), false)

        n = 5
        opC2 = DiagOp(gpu_randn(backend, n)) * DiagOp(gpu_randn(backend, n))
        test_op(opC2, gpu_randn(backend, n), gpu_randn(backend, n), false)
    end
end

@testitem "Compose: combine-at-i2 inlines nested Compose (line 73)" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators, LinearAlgebra
    Random.seed!(0)
    n = 6
    fd = FiniteDiff((n,))                   # domain (n,) → codomain (n-1,)
    M = MatrixOp(randn(n - 1, n - 1))       # domain (n-1,) → codomain (n-1,)
    s = Scale(2.0, FiniteDiff((n - 1,)))    # domain (n-1,) → codomain (n-2,)
    # At i=2: can_be_combined(s, M) → combine returns a Compose → triggers i -= 1 (line 73)
    buf1 = zeros(n - 1)
    buf2 = zeros(n - 1)
    L = AbstractOperators.Compose((fd, M, s), (buf1, buf2))
    @test L isa AbstractOperators.Compose
    x = randn(n)
    @test L * x ≈ s * (M * (fd * x))
end

@testitem "Compose: 4-op chain mid-pair combination triggers i-decrement (line 88)" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators, LinearAlgebra
    Random.seed!(0)
    n = 6
    d1, d2 = randn(n - 1), randn(n - 1)
    fd_n = FiniteDiff((n,))        # domain (n,) → codomain (n-1,)
    diag1 = DiagOp(d1)             # domain/codomain (n-1,)
    diag2 = DiagOp(d2)             # domain/codomain (n-1,)
    fd_nm1 = FiniteDiff((n - 1,))  # domain (n-1,) → codomain (n-2,)
    # At i=2: can_be_combined(diag2, diag1)=true → non-Compose result → elseif branch,
    # i > 1 and buffers not equal → i -= 1 (line 88)
    buf1, buf2, buf3 = zeros(n - 1), zeros(n - 1), zeros(n - 1)
    L = AbstractOperators.Compose((fd_n, diag1, diag2, fd_nm1), (buf1, buf2, buf3))
    @test L isa AbstractOperators.Compose
    x = randn(n)
    @test length(L * x) == n - 2
    @test L * x ≈ fd_nm1 * (diag2 * (diag1 * (fd_n * x)))
end

@testitem "Compose: buffer adjacency check after combination (lines 81-82)" tags = [:calculus, :Compose] setup = [TestUtils] begin
    using Random, AbstractOperators, LinearAlgebra
    Random.seed!(0)
    n = 6
    d1, d2 = randn(n - 1), randn(n - 1)
    fd_n = FiniteDiff((n,))
    diag1 = DiagOp(d1)
    diag2 = DiagOp(d2)
    fd_nm1 = FiniteDiff((n - 1,))
    # buf[1] === buf[3]: after DiagOp pair combines at i=2 and buffers are rebuilt,
    # buf[i-1] === buf[i] is detected → reallocation triggered (lines 81-82)
    shared_buf = zeros(n - 1)
    mid_buf = zeros(n - 1)
    L = AbstractOperators.Compose((fd_n, diag1, diag2, fd_nm1), (shared_buf, mid_buf, shared_buf))
    @test L isa AbstractOperators.Compose
    x = randn(n)
    @test length(L * x) == n - 2
    @test L * x ≈ fd_nm1 * (diag2 * (diag1 * (fd_n * x)))
end
