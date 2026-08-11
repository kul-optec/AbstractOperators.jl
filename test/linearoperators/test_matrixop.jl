@testitem "MatrixOp: basic mul" tags = [:linearoperator, :MatrixOp] setup = [TestUtils] begin
    using Random, SparseArrays, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, m = 5, 4
    A = randn(n, m)
    op = MatrixOp(A)
    test_op(op, randn(m), randn(n), verb)

    A = randn(n, m) + im * randn(n, m)
    op = MatrixOp(A)
    test_op(op, randn(m) + im * randn(m), randn(n) + im * randn(n), verb)

    # array_type keyword is ignored for element-type selection
    A = randn(n, m)
    op = MatrixOp(Float64, (m,), A; array_type = Array{ComplexF32, 2})
    @test domain_array_type(op) == Array{Float64}
    @test codomain_array_type(op) == Array{Float64}

    # complex matrix, real matrix input
    A = randn(n, m) + im * randn(n, m)
    c = 3
    op = MatrixOp(Float64, (m, c), A)
    x1 = randn(m, c)
    y1 = test_op(op, x1, randn(n, c) .+ im .* randn(n, c), verb)
    @test all(norm.(y1 .- A * x1) .<= 1.0e-12)
end

@testitem "MatrixOp: constructors" tags = [:linearoperator, :MatrixOp] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, m = 5, 4
    A = randn(n, m) + im * randn(n, m)
    c = 3
    op = MatrixOp(A)
    op = MatrixOp(Float64, A)
    op = MatrixOp(A, c)
    op = MatrixOp(Float64, A, c)
    op = convert(LinearOperator, A)
    op = convert(LinearOperator, A, c)
    op = convert(LinearOperator, Complex{Float64}, (m, c), A)

    @test_throws ErrorException MatrixOp(Float64, (m, c, 3), A)
    @test_throws MethodError MatrixOp(Float64, (m, c), randn(n, m, 2))
end

@testitem "MatrixOp: properties" tags = [:linearoperator, :MatrixOp] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, m = 5, 4
    A = randn(n, m) + im * randn(n, m)
    c = 3
    op = MatrixOp(Float64, (m, c), A)
    @test is_sliced(op) == false
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(MatrixOp(randn(Random.seed!(0), 3, 4))) == true
    @test is_full_column_rank(MatrixOp(randn(Random.seed!(0), 3, 4))) == false

    B = randn(4, 4)
    invop = MatrixOp(Float64, (4,), B)
    @test is_invertible(invop) == (det(B) != 0)

    Q, _ = qr(randn(5, 5))
    Qmat = Matrix(Q)
    @test is_orthogonal(MatrixOp(Float64, (5,), Qmat)) == true

    D = diagm(0 => randn(5))
    Dop = MatrixOp(Float64, (5,), D)
    @test is_diagonal(Dop) == true
    @test is_AcA_diagonal(Dop) == true
    @test is_AAc_diagonal(Dop) == true
end

@testitem "MatrixOp: adjoint and in-place" tags = [:linearoperator, :MatrixOp] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    Q, _ = qr(randn(5, 5))
    Qmat = Matrix(Q)
    Qop = MatrixOp(Float64, (5,), Qmat)
    xvec = randn(5)
    yvec = zeros(5)
    mul!(yvec, Qop, xvec)
    @test yvec ≈ Qmat * xvec
    Xmat = randn(5, 3)
    Ymat = zeros(5, 3)
    mul!(Ymat, Qop, Xmat)
    @test Ymat ≈ Qmat * Xmat

    Cmat = randn(3, 3) + im * randn(3, 3)
    Cop = MatrixOp(Float64, (3,), Cmat)
    xr = randn(3)
    @test Cop' * xr ≈ real.(Cmat' * xr)

    Nop = AbstractOperators.get_normal_op(Qop)
    v = randn(5)
    @test Nop * v ≈ Qop' * (Qop * v)

    @test opnorm(Qop) ≈ estimate_opnorm(Qop) rtol = 0.02
end

@testitem "MatrixOp (GPU)" tags = [:gpu, :linearoperator, :MatrixOp] setup = [TestUtils] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)
        n, m = 5, 4
        A = randn(n, m)
        test_op(MatrixOp(to_gpu(backend, A)), gpu_randn(backend, m), gpu_randn(backend, n), false)
        Ac = randn(n, m) + im * randn(n, m)
        test_op(
            MatrixOp(to_gpu(backend, Ac)),
            gpu_randn(backend, ComplexF64, m),
            gpu_randn(backend, ComplexF64, n),
            false,
        )
    end
end
