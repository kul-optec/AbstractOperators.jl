if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end

@testset "Compose" begin
    verb && println(" --- Testing Compose --- ")

    m1, m2 = 4, 7
    A1 = randn(m2, m1)
    opA1 = MatrixOp(A1)
    opF = FiniteDiff((m2,))

    opC = Compose(opF, opA1)
    x = randn(m1)
    y1 = test_op(opC, x, randn(m2-1), verb)
    y2 = diff(A1 * x)
    @test y1 == y2

    # test Compose longer
    m1, m2, m3 = 4, 7, 3
    A1 = randn(m2, m1)
    A2 = randn(m3, m2-1)
    opA1 = MatrixOp(A1)
    opF = FiniteDiff((m2,))
    opA2 = MatrixOp(A2)

    opC1 = Compose(opA2, Compose(opF, opA1))
    opC2 = Compose(Compose(opA2, opF), opA1)
    x = randn(m1)
    y1 = test_op(opC1, x, randn(m3), verb)
    y2 = test_op(opC2, x, randn(m3), verb)
    y3 = A2 * diff(A1 * x)
    @test all(norm.(y1 .- y2) .<= 1e-12)
    @test all(norm.(y3 .- y2) .<= 1e-12)

    #test Compose special cases
    @test typeof(opA1 * Eye(m1)) == typeof(opA1)
    @test typeof(Eye(m2) * opA1) == typeof(opA1)
    @test typeof(Eye(m2) * Eye(m2)) == typeof(Eye(m2))

    opS1 = Compose(opF, opA1)
    opS1c = Scale(pi, opS1)
    @test opS1c isa Compose # Scaling is fused with opA1

    # In-place multiplication coverage
    opS = MyLinOp(Float64, (m2,), Float64, (m2,), (y, x) -> y .= x .* 2, (y, x) -> y .= x ./ 2)
    C = Compose(opS, opA1)
    x = randn(m1)
    y = zeros(m2)
    mul!(y, C, x)
    @test y ≈ 2 .* (A1 * x)

    # Adjoint reversal property ( (A*B* C)' == C'*B'*A')
    A2 = randn(m3, m2)
    opA2 = MatrixOp(A2)
    chain = Compose(opA2, Compose(opS, opA1))
    chain_adj = chain'
    x_in = randn(m3)
    y_chain = chain_adj * x_in
    y_ref = opA1' * (1//2 .* (opA2' * x_in))
    @test y_chain ≈ y_ref

    # Dimension mismatch error
    @test_throws Exception Compose(MatrixOp(randn(5,4)), MatrixOp(randn(3,2)))

    # Identity elimination & fusion checks
    E = Eye(m2)
    comp1 = opA2 * E * opA1
    @test comp1 * x ≈ A2 * A1 * x

    # Composition with Zeros yields Zeros (front)
    Z = Zeros(Float64, (m2,), Float64, (m2,))
    ZC = Compose(opA2, Z)
    @test is_null(ZC)
    @test ZC * x == zeros(m3)

    # Composition with diagonal preserves diagonality when both diagonal
    struct MyDiagOp <: LinearOperator end
    LinearAlgebra.size(::MyDiagOp) = ((m1,), (m1,))
    AbstractOperators.domain_type(::MyDiagOp) = Float64
    AbstractOperators.codomain_type(::MyDiagOp) = Float64
    AbstractOperators.is_diagonal(::MyDiagOp) = true
    AbstractOperators.diag(::MyDiagOp) = 3.0
    d = randn(m1)
    D1 = DiagOp(d)
    D2 = MyDiagOp()
    DD = Compose(D2, D1)
    @test is_diagonal(DD)
    @test diag(DD) == diag(D2) .* diag(D1)

    # Scale inside composition
    S = Scale(2.5, opF)
    SC = Compose(S, opA1)
    @test SC * x ≈ 2.5 * diff(A1 * x)

    # Show output coverage
    io = IOBuffer(); show(io, chain); str = String(take!(io)); @test occursin("Π", str)

    # Displacement: nested remove_displacement idempotence
    dvec = randn(m2)
    Aff = AffineAdd(opA1, dvec)
    comp_disp = Compose(opA2, Aff)
    x = randn(m1)
    y_full = comp_disp * x
    y_split = A2 * (A1 * x + dvec)
    @test y_full ≈ y_split
    rd = remove_displacement(comp_disp)
    rd2 = remove_displacement(rd)
    @test rd * x ≈ A2 * (A1 * x)
    @test rd2 * x ≈ rd * x

    # Sliced + diagonal detection after composition GetIndex * DiagOp
    sel = 1:minimum((length(d), 3))
    Sliced1 = Compose(GetIndex((length(d),), sel), DiagOp(d))
    @test !is_sliced(Sliced1)
    @test !is_diagonal(Sliced1)
    @test is_AAc_diagonal(Sliced1)
    Sliced2 = Compose(DiagOp(d[sel]), GetIndex((length(d),), sel))
    @test is_sliced(Sliced2)
    @test is_diagonal(Sliced2)

    #properties
    @test is_sliced(opC) == false
    @test is_linear(opC1) == true
    @test is_null(opC1) == false
    @test is_eye(opC1) == false
    @test is_diagonal(opC1) == false
    @test is_AcA_diagonal(opC1) == false
    @test is_AAc_diagonal(opC1) == false
    @test is_orthogonal(opC1) == false
    @test is_invertible(opC1) == false
    @test is_full_row_rank(opC1) == (is_full_row_rank(opC1.A[1]) && is_full_row_rank(opC1.A[2]))
    @test is_full_column_rank(opC1) == (is_full_column_rank(opC1.A[1]) && is_full_column_rank(opC1.A[2]))

    # properties special case
    d = randn(5)
    opC = DiagOp(d) * GetIndex((10,), 1:5)
    @test is_sliced(opC) == true
    @test is_diagonal(opC) == true
    @test diag(opC) == d

    # displacement test
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

    @test norm(opC * x - (A4 * (A3 * (A2 * (A1 * x + d1) .+ d2) .+ d3) - d4)) < 1e-9
    @test norm(displacement(opC) - (A4 * (A3 * (A2 * d1 .+ d2) .+ d3) - d4)) < 1e-9

    opA4 = MatrixOp(A4)
    opC = AffineAdd(Compose(Compose(Compose(opA4, opA3), opA2), opA1), d4)
    @test norm(opC * x - (A4 * (A3 * (A2 * (A1 * x + d1) .+ d2) .+ d3) + d4)) < 1e-9
    @test norm(displacement(opC) - (A4 * (A3 * (A2 * d1 .+ d2) .+ d3) + d4)) < 1e-9

    @test norm(remove_displacement(opC) * x - (A4 * (A3 * (A2 * (A1 * x))))) < 1e-9

    # Error paths: domain/codomain type/storage mismatch
    struct DummyOp <: LinearOperator end
    LinearAlgebra.size(::DummyOp) = ((2,), (2,))
    AbstractOperators.domain_type(::DummyOp) = Int
    AbstractOperators.codomain_type(::DummyOp) = Int
    AbstractOperators.is_linear(::DummyOp) = true
    AbstractOperators.is_diagonal(::DummyOp) = false
    AbstractOperators.is_null(::DummyOp) = false
    AbstractOperators.is_eye(::DummyOp) = false
    AbstractOperators.is_AcA_diagonal(::DummyOp) = false
    AbstractOperators.is_AAc_diagonal(::DummyOp) = false
    AbstractOperators.is_orthogonal(::DummyOp) = false
    AbstractOperators.is_invertible(::DummyOp) = false
    AbstractOperators.is_full_row_rank(::DummyOp) = false
    AbstractOperators.is_full_column_rank(::DummyOp) = false
    AbstractOperators.diag(::DummyOp) = 1
    AbstractOperators.fun_name(::DummyOp) = "D2"
    opint = DummyOp()
    @test_throws DomainError Compose(DiagOp(rand(2)), opint)

    # Show output patterns for Compose (2-term vs multi-term) instead of direct fun_name (non-exported)
    C2 = Compose(DiagOp(rand(2)), FiniteDiff((3,)))
    io_fn = IOBuffer(); show(io_fn, C2); str_fn = String(take!(io_fn))
    @test occursin("╲*δx", str_fn)
    C4 = DiagOp(rand(2)) * FiniteDiff((3,)) * DiagOp(rand(3))
    io_fn = IOBuffer(); show(io_fn, C4); str_fn = String(take!(io_fn))
    @test occursin("Π", str_fn)

    # opnorm/estimate_opnorm consistency (using DiagOp for simplicity)
    d = randn(2)
    D1 = DiagOp(d)
    D2 = FiniteDiff((3,))
    CC = Compose(D1, D2)
    @test estimate_opnorm(CC) ≈ estimate_opnorm(D1) * estimate_opnorm(D2)
    opnorm_CC = opnorm(CC)
    @test abs(opnorm_CC - estimate_opnorm(CC)) / opnorm_CC < 0.25

    # permute utility
    A1 = MatrixOp(randn(2,2))
    A2 = MatrixOp(randn(2,2))
    A3 = MatrixOp(randn(2,2))
    Cperm = Compose(A3, HCAT(A1, A2))
    p = [2,1]
    Cperm2 = AbstractOperators.permute(Cperm, p)
    @test size(Cperm2) == size(Cperm)
    x = ArrayPartition(randn(2), randn(2))
    y1 = Cperm * x
    y2 = Cperm2 * ArrayPartition(x.x[p]...)
    @test y1 == y2

    # remove_slicing utility
    S = Compose(DiagOp(randn(2)), GetIndex((5,), 1:2))
    S2 = AbstractOperators.remove_slicing(S)
    @test S2 isa DiagOp

    # get_operators utility
    ops = AbstractOperators.get_operators(CC)
    @test length(ops) == 2

    # Testing nonlinear Compose
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

    Y = A * (opB * (opC * x))
    @test norm(Y - y) < 1e-8

    ## NN
    m, n, l = 4, 7, 5
    b = randn(l)
    opS1 = Sigmoid(Float64, (n,), 2)
    x = ArrayPartition(randn(n, l), randn(n))
    r = randn(n)

    A1 = HCAT(LMatrixOp(b, n), Eye(n))
    op = Compose(opS1, A1)
    y, grad = test_NLop(op, x, r, verb)
end
