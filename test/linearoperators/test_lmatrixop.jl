@testitem "LMatrixOp: basic mul" tags = [:linearoperator, :LMatrixOp] setup = [TestUtils] begin
    using Random, AbstractOperators

    Random.seed!(0)
    verb && println(" --- Testing LMatrixOp: basic mul --- ")

    function test_lmatrixop_mul(conv, verb)
        n, m = 5, 6
        b = randn(m)
        op = LMatrixOp(Float64, (n, m), conv(b))
        test_op(op, conv(randn(n, m)), conv(randn(n)), verb)
    end

    test_lmatrixop_mul(identity, verb)

    n, m = 5, 6
    b = randn(m)
    op = LMatrixOp(Float64, (n, m), b)
    op_array_type = LMatrixOp(Float64, (n, m), b; array_type = Array{ComplexF32, 2})
    @test domain_array_type(op_array_type) == Array{Float64}
    @test codomain_array_type(op_array_type) == Array{Float64}
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = x1 * b
    @test all(norm.(y1 .- y2) .<= 1.0e-12)
    # size (codomain, domain)
    @test size(op) == ((n,), (n, m))
    @test domain_type(op) == Float64
    @test codomain_type(op) == Float64

    n, m = 5, 6
    b = randn(m) + im * randn(m)
    op = LMatrixOp(Complex{Float64}, (n, m), b)
    x1 = randn(n, m) + im * randn(n, m)
    y1 = test_op(op, x1, randn(n) + im * randn(n), verb)
    y2 = x1 * b
    @test all(norm.(y1 .- y2) .<= 1.0e-12)
    @test size(op) == ((n,), (n, m))

    n, m, l = 5, 6, 7
    b = randn(m, l)
    op = LMatrixOp(Float64, (n, m), b)
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(n, l), verb)
    y2 = x1 * b
    @test all(norm.(y1 .- y2) .<= 1.0e-12)
    @test size(op) == ((n, l), (n, m))

    n, m, l = 5, 6, 7
    b = randn(m, l) + im * randn(m, l)
    op = LMatrixOp(Complex{Float64}, (n, m), b)
    x1 = randn(n, m) + im * randn(n, m)
    y1 = test_op(op, x1, randn(n, l) + im * randn(n, l), verb)
    y2 = x1 * b
    @test all(norm.(y1 .- y2) .<= 1.0e-12)
    @test size(op) == ((n, l), (n, m))
end

@testitem "LMatrixOp: other constructors" tags = [:linearoperator, :LMatrixOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing LMatrixOp: other constructors --- ")

    n, m, l = 5, 6, 7

    ## other constructors (vector b and matrix b)
    bvec = randn(m)
    op_vec = LMatrixOp(bvec, n)
    @test size(op_vec) == ((n,), (n, m))
    bmat = randn(m, l)
    op_mat = LMatrixOp(bmat, n)
    @test size(op_mat) == ((n, l), (n, m))

    # In-place mul! (vector b)
    X = randn(n, m)
    y = zeros(n)
    mul!(y, op_vec, X)
    @test y ≈ X * bvec

    # In-place mul! (matrix b)
    X2 = randn(n, m)
    Y = zeros(n, l)
    mul!(Y, op_mat, X2)
    @test Y ≈ X2 * bmat

    # Adjoint application (vector b) : (⋅)b' acting on vector gives outer product
    z = collect(1.0:n)
    Zout = zeros(n, m)
    mul!(Zout, op_vec', z)
    @test Zout ≈ z * bvec'

    # Adjoint with matrix right-hand side
    b_mat = randn(m, 2)
    op_mat2 = LMatrixOp(b_mat, n)
    Zrhs = randn(n, 2)
    Zout2 = zeros(n, m)
    mul!(Zout2, op_mat2', Zrhs)
    @test Zout2 ≈ Zrhs * b_mat'
end

@testitem "LMatrixOp: scale and properties" tags = [:linearoperator, :LMatrixOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing LMatrixOp: scale and properties --- ")

    n, m, l = 5, 6, 7
    bvec = randn(m)
    op_vec = LMatrixOp(bvec, n)
    X = randn(n, m)

    # Scaling
    Sop = Scale(2.5, op_vec)
    @test Sop * X ≈ 2.5 * (op_vec * X)
    @test_throws ErrorException Scale(1 + 2im, op_vec)  # real domain, complex scale

    # Thread safety flag
    @test is_thread_safe(op_vec) == true

    # Error path: dimension mismatch (wrong inner dimension)
    Xbad = randn(n, m + 1)
    @test_throws DimensionMismatch (op_vec * Xbad)

    b = randn(m, l) + im * randn(m, l)
    op = LMatrixOp(Complex{Float64}, (n, m), b)

    ##properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == false
    @test is_full_column_rank(op) == false

    # Show output symbol
    io = IOBuffer()
    show(io, op_vec)
    s = String(take!(io))
    @test occursin("(⋅)b", s)
end

@testitem "LMatrixOp (GPU)" tags = [:linearoperator, :LMatrixOp, :gpu] setup = [TestUtils] begin
    using Random, AbstractOperators, GPUEnv
    for backend in gpu_backends()
        Random.seed!(0)
        n, m = 5, 6
        b = gpu_randn(backend, m)
        op = LMatrixOp(Float64, (n, m), b)
        test_op(op, gpu_randn(backend, n, m), gpu_randn(backend, n), false)
    end
end
