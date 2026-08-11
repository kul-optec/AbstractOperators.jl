@testitem "Sum: basic mul" tags = [:calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    A3 = randn(m, n)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opS = Sum(opA1, opA2, opA3)
    x1 = randn(n)
    y1 = test_op(opS, x1, randn(m), verb)
    @test norm(y1 - (A1 * x1 + A2 * x1 + A3 * x1)) <= 1.0e-12

    @test_throws Exception Sum(opA1, MatrixOp(randn(m, m)))
    @test is_full_row_rank(opS) == true
    @test is_full_column_rank(opS) == false
end

@testitem "Sum: displacement" tags = [:calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    A3 = randn(m, n)
    d1 = randn(m)
    d2 = pi
    d3 = randn(m)
    opA1 = AffineAdd(MatrixOp(A1), d1)
    opA2 = AffineAdd(MatrixOp(A2), d2)
    opA3 = AffineAdd(MatrixOp(A3), d3)
    opS = Sum(opA1, opA2, opA3)
    x1 = randn(n)
    @test norm(opS * x1 - (A1 * x1 + A2 * x1 + A3 * x1 + d1 .+ d2 + d3)) <= 1.0e-12
    @test norm(displacement(opS) - (d1 .+ d2 + d3)) <= 1.0e-12
    @test norm(remove_displacement(opS) * x1 - (A1 * x1 + A2 * x1 + A3 * x1)) <= 1.0e-12
end

@testitem "Sum: properties" tags = [:calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n = 5, 7
    Aeq = MatrixOp(randn(m, n))
    Beq = MatrixOp(randn(m, n))
    S1 = Sum(Aeq, Beq)
    S2 = Sum(Aeq, Beq)
    S3 = Sum(Beq, Aeq)
    @test S1 == S2 && S1 != S3
    @test Sum(Aeq) === Aeq

    z = Zeros(Float64, (n,), Float64, (m,))
    @test Sum(Aeq, z) === Aeq

    d = randn(10)
    op = Sum(Scale(-3.1, Eye(10)), DiagOp(d))
    @test is_diagonal(op) == true
    @test norm(diag(op) - (d .- 3.1)) < 1.0e-12

    @test domain_array_type(S1) !== nothing
    @test codomain_array_type(S1) !== nothing
    @test is_thread_safe(S1) == false
end

@testitem "Sum: nonlinear" tags = [:calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m = 5
    x = randn(m)
    r = randn(m)
    A = randn(m, m)
    opB = Sigmoid(Float64, (m,), 2)
    op = Sum(MatrixOp(A), opB)
    y, grad = test_NLop(op, x, r, verb)
    @test norm(A * x + opB * x - y) < 1.0e-8
end

@testitem "Sum (GPU)" tags = [:gpu, :calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n = 5
        opS = Sum(DiagOp(gpu_ones(backend, Float64, n)), DiagOp(to_gpu(backend, 2 .* ones(n))))
        test_op(opS, gpu_randn(backend, n), gpu_randn(backend, n), false)

        m, n2 = 5, 7
        A1 = gpu_randn(backend, m, n2)
        A2 = gpu_randn(backend, m, n2)
        opS2 = Sum(MatrixOp(A1), MatrixOp(A2))
        test_op(opS2, gpu_randn(backend, n2), gpu_randn(backend, m), false)
    end
end

@testitem "Sum: all-Zeros degenerate case (line 79)" tags = [:calculus, :Sum] setup = [TestUtils] begin
    using AbstractOperators
    n, m = 5, 4
    # Sum of two Zeros: @generated code returns A[1] when n_flat == 0 (line 79)
    z1 = Zeros(Float64, (n,), Float64, (m,))
    z2 = Zeros(Float64, (n,), Float64, (m,))
    result = Sum(z1, z2)
    @test result === z1
    x = randn(n)
    @test all(result * x .== 0)
end

@testitem "Sum: copy_operator" tags = [:calculus, :Sum] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(4)

    m, n = 5, 7
    A1 = randn(m, n)
    A2 = randn(m, n)
    opS = Sum(MatrixOp(A1), MatrixOp(A2))
    opS2 = copy_operator(opS; threaded = true)
    @test opS2 isa Sum
    x = randn(n)
    y1 = zeros(m)
    y2 = zeros(m)
    mul!(y1, opS, x)
    mul!(y2, opS2, x)
    @test y1 ≈ y2
end
