if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_NLop)
    include("../utils.jl")
end

@testset "HadamardProd" begin
    verb && println(" --- Testing HadamardProd --- ")

    # Basic square identity factors (Eye.*Eye)
    n = 3
    A, B = Eye(n, n), Eye(n, n)
    P = HadamardProd(A, B)
    x = randn(n, n)
    r = randn(n, n)
    y, grad = test_NLop(P, x, r, verb)
    @test norm(x .* x - y) < 1e-9

    # Sin .* Cos multi-column
    n, l = 3, 2
    A, B = Sin(n, l), Cos(n, l)
    P = HadamardProd(A, B)
    x = randn(n, l)
    r = randn(n, l)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((A * x) .* (B * x) - y) < 1e-9

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
    @test norm((op1 * x) .* (op2 * x) - y) < 1e-9

    # remove_displacement and its idempotence
    y2, grad2 = test_NLop(remove_displacement(P), x, r, verb)
    @test norm((op1 * x - b) .* (op2 * x) - y2) < 1e-8
    @test remove_displacement(remove_displacement(P)) == remove_displacement(P)

    # permute
    p = [2, 1]
    Pp = AbstractOperators.permute(P, p)
    xp = ArrayPartition(x.x[p])
    yperm, gradperm = test_NLop(Pp, xp, r, verb)
    @test norm(yperm - y) < 1e-8

    # Dimension mismatch error path
    @test_throws Exception HadamardProd(Eye(2, 2, 2), Eye(1, 2, 2))

    # Storage type / thread safety accessors
    _ds = domain_storage_type(P)
    _cs = codomain_storage_type(P)
    @test _ds !== nothing
    @test _cs !== nothing
    @test is_thread_safe(P) == false

    # show / fun_name pattern (indirect)
    io = IOBuffer()
    show(io, P)
    str = String(take!(io))
    @test occursin(".*", str)
    # --- Additional coverage ---
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
    @test domain_storage_type(P1) !== nothing
    @test codomain_storage_type(P1) !== nothing

    # fun_name direct
    io = IOBuffer(); show(io, P1); sP1 = String(take!(io))
    @test occursin(".*", sP1)

    # permute with more than 2 domains (using HCAT)
    mH = 4; n1 = 2; n2 = 2
    A1p = MatrixOp(randn(mH, n1))
    A2p = MatrixOp(randn(mH, n2))
    H1 = HCAT(A1p, A2p)
    H2 = HCAT(A2p, A1p)
    P = HadamardProd(H1, H2)
    x1p = randn(n1); x2p = randn(n2)
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
