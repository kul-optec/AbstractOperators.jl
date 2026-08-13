@testitem "HadamardProd: basic mul" tags = [:calculus, :HadamardProd] setup = [TestUtils] begin
    using AbstractOperators
    verb && println(" --- Testing HadamardProd: basic mul --- ")

    # Basic square identity factors (Eye.*Eye)
    n = 3
    A, B = Eye(n, n), Eye(n, n)
    P = HadamardProd(A, B)
    x = randn(n, n)
    r = randn(n, n)
    y, grad = test_NLop(P, x, r, verb)
    @test norm(x .* x - y) < 1.0e-9

    # Sin .* Cos multi-column
    n, l = 3, 2
    A, B = Sin(n, l), Cos(n, l)
    P = HadamardProd(A, B)
    x = randn(n, l)
    r = randn(n, l)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((A * x) .* (B * x) - y) < 1.0e-9

    # HCAT factors to exercise ArrayPartition domain/codomain handling
    m, n = 3, 5
    x = ArrayPartition(randn(m), randn(n))
    r = randn(m)
    b = randn(m)
    A1 = AffineAdd(Sin(Float64, (m,)), b)
    B1 = MatrixOp(randn(m, n))
    op1 = HCAT(A1, B1)
    C1 = Cos(Float64, (m,))
    D1 = MatrixOp(randn(m, n))
    op2 = HCAT(C1, D1)
    P = HadamardProd(op1, op2)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((op1 * x) .* (op2 * x) - y) < 1.0e-9
end

@testitem "HadamardProd: properties" tags = [:calculus, :HadamardProd] setup = [TestUtils] begin
    using AbstractOperators
    verb && println(" --- Testing HadamardProd: properties --- ")

    # Re-create the HCAT-based P for remove_displacement and permute tests
    m, n = 3, 5
    x = ArrayPartition(randn(m), randn(n))
    r = randn(m)
    b = randn(m)
    A1 = AffineAdd(Sin(Float64, (m,)), b)
    B1 = MatrixOp(randn(m, n))
    op1 = HCAT(A1, B1)
    C1 = Cos(Float64, (m,))
    D1 = MatrixOp(randn(m, n))
    op2 = HCAT(C1, D1)
    P = HadamardProd(op1, op2)
    y, grad = test_NLop(P, x, r, verb)

    # remove_displacement and its idempotence
    y2, grad2 = test_NLop(remove_displacement(P), x, r, verb)
    @test norm((op1 * x - b) .* (op2 * x) - y2) < 1.0e-8
    @test remove_displacement(remove_displacement(P)) == remove_displacement(P)

    # permute
    p = [2, 1]
    Pp = AbstractOperators.permute(P, p)
    xp = ArrayPartition(x.x[p])
    yperm, gradperm = test_NLop(Pp, xp, r, verb)
    @test norm(yperm - y) < 1.0e-8

    # Dimension mismatch error path
    @test_throws Exception HadamardProd(Eye(2, 2, 2), Eye(1, 2, 2))

    # Storage type / thread safety accessors
    _ds = domain_array_type(P)
    _cs = codomain_array_type(P)
    @test _ds !== nothing
    @test _cs !== nothing
    @test is_thread_safe(P) == false

    # show / fun_name pattern (indirect)
    io = IOBuffer()
    show(io, P)
    str = String(take!(io))
    @test occursin(".*", str)
end

@testitem "HadamardProd: equality and permute" tags = [:calculus, :HadamardProd] setup = [TestUtils] begin
    using AbstractOperators
    verb && println(" --- Testing HadamardProd: equality and permute --- ")

    # Equality / inequality
    n = 3
    A = Eye(n, n)
    B = Eye(n, n)
    C = DiagOp(randn(n, n))
    P1 = HadamardProd(A, B)
    P2 = HadamardProd(A, B)
    P3 = HadamardProd(B, C)
    @test P1 == P2
    @test P1 != P3

    # size, domain_type, codomain_type, storage types
    @test size(P1) == ((n, n), (n, n))
    @test domain_type(P1) == domain_type(A)
    @test codomain_type(P1) == codomain_type(A)
    @test domain_array_type(P1) !== nothing
    @test codomain_array_type(P1) !== nothing

    # fun_name direct
    io = IOBuffer()
    show(io, P1)
    sP1 = String(take!(io))
    @test occursin(".*", sP1)

    # permute with more than 2 domains (using HCAT)
    mH = 4
    n1 = 2
    n2 = 2
    A1p = MatrixOp(randn(mH, n1))
    A2p = MatrixOp(randn(mH, n2))
    H1 = HCAT(A1p, A2p)
    H2 = HCAT(A2p, A1p)
    P = HadamardProd(H1, H2)
    x1p = randn(n1)
    x2p = randn(n2)
    y_orig, _ = test_NLop(P, ArrayPartition(x1p, x2p), randn(mH), verb)
    p = [2, 1]
    Pp = AbstractOperators.permute(P, p)
    y_perm, _ = test_NLop(Pp, ArrayPartition(x2p, x1p), randn(mH), verb)
    @test y_orig ≈ y_perm

    # remove_displacement idempotence with displacement underlying
    b = randn(n, n)
    Pdisp = HadamardProd(AffineAdd(A, b), B)
    Prd = remove_displacement(Pdisp)
    @test remove_displacement(Prd) == Prd
end

@testitem "HadamardProd (GPU)" tags = [:gpu, :calculus, :HadamardProd] setup = [TestUtils, GPUNLTestUtils] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n = 3
        P = HadamardProd(
            Eye(Float64, (n, n); array_type = gpu_wrapper(backend, Float64, n, n)),
            Eye(Float64, (n, n); array_type = gpu_wrapper(backend, Float64, n, n)),
        )
        x = gpu_randn(backend, n, n)
        r = gpu_randn(backend, n, n)
        test_NLop_gpu(P, x, r, false)

        n2, l = 3, 2
        AT = gpu_wrapper(backend, Float64, n2, l)
        P2 = HadamardProd(Sin(Float64, (n2, l); array_type = AT), Cos(Float64, (n2, l); array_type = AT))
        x2 = gpu_randn(backend, n2, l)
        r2 = gpu_randn(backend, n2, l)
        test_NLop_gpu(P2, x2, r2, false)
    end
end

@testitem "HadamardProd: copy_operator" tags = [:calculus, :HadamardProd] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(3)

    n = 6
    P = HadamardProd(Sin((n,)), Cos((n,)))
    P2 = copy_operator(P)
    @test P2 isa HadamardProd
    x = randn(n)
    y1 = zeros(n)
    y2 = zeros(n)
    mul!(y1, P, x)
    mul!(y2, P2, x)
    @test y1 ≈ y2
    # Verify buffer independence — a second forward should not cross-contaminate
    x2 = randn(n)
    y3 = zeros(n)
    mul!(y3, P2, x2)
    @test y3 ≈ P * x2

    # Explicit threaded kwarg: exercise _copy_operator_impl unambiguously
    P3 = copy_operator(P; threaded = true, storage_type = nothing)
    @test P3 isa HadamardProd
    y4 = zeros(n)
    mul!(y4, P3, x)
    @test y4 ≈ y1
end

@testitem "HadamardProdJac: copy_operator" tags = [:calculus, :HadamardProd] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(3)

    n = 6
    P = HadamardProd(Sin((n,)), Cos((n,)))
    x = randn(n)
    P * x  # forward pass populates P's buffers, which Jacobian(P, x) reuses
    J = Jacobian(P, x)
    J2 = copy_operator(J; threaded = true, storage_type = nothing)
    @test J2 isa AbstractOperators.HadamardProdJac
    # Buffers must be independent copies, not aliases, before either gets mutated
    @test J2.bufA !== J.bufA
    @test J2.bufB !== J.bufB
    @test J2.bufD !== J.bufD

    y = randn(n)
    g1 = J' * y
    g2 = J2' * y
    @test g1 ≈ g2
end
