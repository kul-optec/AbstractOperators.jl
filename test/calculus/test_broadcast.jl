@testitem "BroadCast: basic mul" tags = [:calculus, :BroadCast] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing BroadCast --- ")

    m, n = 8, 4
    dim_out = (m, 10)
    A1 = randn(m, n)
    opA1 = MatrixOp(A1)
    opR = BroadCast(opA1, dim_out)
    x1 = randn(n)
    y1 = test_op(opR, x1, randn(dim_out), verb)
    y2 = zeros(dim_out)
    y2 .= A1 * x1
    @test norm(y1 - y2) <= 1.0e-12

    m, n, l, k = 8, 4, 5, 7
    dim_out = (m, n, l, k)
    opA1 = Eye(m, n)
    opR = BroadCast(opA1, dim_out)
    x1 = randn(m, n)
    y1 = test_op(opR, x1, randn(dim_out), verb)
    y2 = zeros(dim_out)
    y2 .= x1
    @test norm(y1 - y2) <= 1.0e-12
    @test_throws Exception BroadCast(opA1, (m, m))

    m, n = 8, 4
    dim_out = (m, 10)
    d1 = randn(m)
    opA1 = AffineAdd(MatrixOp(randn(m, n)), d1)
    opR = BroadCast(opA1, dim_out)
    x1 = randn(n)
    y1 = opR * x1
    y2 = zeros(dim_out)
    y2 .= opA1.A.A * x1 + d1
    @test norm(y1 - y2) <= 1.0e-12
    y3 = remove_displacement(opR) * x1
    y4 = zeros(dim_out)
    y4 .= opA1.A.A * x1
    @test norm(y3 - y4) <= 1.0e-12
end

@testitem "BroadCast: properties and storage" tags = [:calculus, :BroadCast] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n = 8, 4
    dim_out = (m, 10)
    opA1 = MatrixOp(randn(m, n))
    opR = BroadCast(opA1, dim_out)
    @test is_null(opR) == is_null(opA1)
    @test is_eye(opR) == false
    @test is_diagonal(opR) == false
    @test is_AcA_diagonal(opR) == false
    @test is_AAc_diagonal(opR) == false
    @test is_orthogonal(opR) == false
    @test is_invertible(opR) == false
    @test is_full_row_rank(opR) == false
    @test is_full_column_rank(opR) == false
    @test is_thread_safe(opR) == false
    @test domain_array_type(opR) !== nothing
    @test codomain_array_type(opR) !== nothing
    @test AbstractOperators.has_fast_opnorm(opR) == AbstractOperators.has_fast_opnorm(opA1)

    m = 3
    E = Eye(m)
    SB = BroadCast(E, (m, 2))
    @test SB isa AbstractOperators.NoOperatorBroadCast
    x = randn(m)
    y = SB * x
    @test y[:, 1] == x && y[:, 2] == x
    @test opnorm(SB) == √2
    @test remove_displacement(SB) === SB
end

@testitem "BroadCast: nonlinear" tags = [:calculus, :BroadCast] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, l = 4, 7
    x = randn(n)
    r = randn(n, l)
    opS = Sigmoid(Float64, (n,), 2)
    op = BroadCast(opS, (n, l))
    y, grad = test_NLop(op, x, r, verb)
    @test norm((opS * x) .* ones(n, l) - y) < 1.0e-8
end

@testitem "BroadCast error cases and edge paths" tags = [:calculus, :BroadCast] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m, n = 4, 3
    A = MatrixOp(randn(m, n))
    @test_throws DimensionMismatch BroadCast(A, (2,))

    E1 = Eye(3)
    E2 = Eye(3)
    B1 = BroadCast(E1, (3, 2); threaded = false)
    B2 = BroadCast(E2, (3, 2); threaded = false)
    B3 = BroadCast(E1, (3, 3); threaded = false)
    @test B1 == B2
    @test B1 != B3

    @test AbstractOperators.has_fast_opnorm(B1) == AbstractOperators.has_fast_opnorm(E1)
    A_op = MatrixOp(randn(3, 2))
    B_op = BroadCast(A_op, (3, 4); threaded = false)
    @test opnorm(B_op) ≈ opnorm(A_op)

    if Threads.nthreads() > 1
        B_op_t = BroadCast(A_op, (3, 4); threaded = true)
        @test opnorm(B_op_t) ≈ opnorm(A_op)
    end

    A = DiagOp(rand(4, 3, 2))
    @test_throws ErrorException BroadCast(A, (4, 2))
end

@testitem "Threaded NoOperatorBroadCast" tags = [:calculus, :BroadCast] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    if Threads.nthreads() > 1
        m = 1000
        E = Eye(m)
        B_threaded = BroadCast(E, (m, 100); threaded = true)
        @test B_threaded isa AbstractOperators.NoOperatorBroadCast
        x = randn(m)
        y = B_threaded * x
        for i in 1:100
            @test y[:, i] ≈ x
        end
        y_adj = randn(m, 100)
        x_back = B_threaded' * y_adj
        @test x_back ≈ dropdims(sum(y_adj, dims = 2), dims = 2)
    end
end

@testitem "Non-compact threaded OperatorBroadCast" tags = [:calculus, :BroadCast] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    if Threads.nthreads() > 1
        m, n = 3, 2
        A = reshape(MatrixOp(randn(m, n)), 1, m)
        dim_out = (4, m, 5)
        B_noncompact = BroadCast(A, dim_out; threaded = true)
        x = randn(n)
        y = B_noncompact * x
        ref = A * x
        for i in 1:4, j in 1:5
            @test y[i, :, j] ≈ vec(ref)
        end
        y_test = randn(dim_out)
        x_back = B_noncompact' * y_test
        @test size(x_back) == (n,)
        @test x_back ≈ A' * dropdims(sum(y_test, dims = (1, 3)), dims = 3)
    end
end

@testitem "BroadCast (GPU)" tags = [:gpu, :calculus, :BroadCast] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        m, n = 8, 4
        dim_out = (m, 10)
        A1 = gpu_randn(backend, m, n)
        opR = BroadCast(MatrixOp(A1), dim_out; threaded = false)
        test_op(opR, gpu_randn(backend, n), gpu_randn(backend, dim_out...), false)

        m2, n2 = 3, 3
        dim_out2 = (m2, n2, 5)
        opR2 = BroadCast(
            Eye(Float64, (m2, n2); array_type = gpu_wrapper(backend, Float64, m2, n2)),
            dim_out2;
            threaded = false,
        )
        test_op(opR2, gpu_randn(backend, m2, n2), gpu_randn(backend, dim_out2...), false)
    end
end

@testitem "BroadCast same-size returns operator unchanged" tags = [:calculus, :BroadCast] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    m, n = 5, 3
    A = MatrixOp(randn(m, n))
    # BroadCast with dim_out == size(A, 1) should return A unchanged (line 80)
    result = BroadCast(A, size(A, 1))
    @test result === A
    # Same for 2D Eye
    B = Eye(m, n)
    result2 = BroadCast(B, size(B, 1))
    @test result2 === B
end

@testitem "BroadCast non-compact adjoint reshape" tags = [:calculus, :BroadCast] setup = [TestUtils] begin
    using Random, AbstractOperators, LinearAlgebra
    Random.seed!(0)
    m, n = 3, 2
    # Reshape codomain to (1, m) so slices of dim_out=(4,m,5) won't match (line 144)
    A = reshape(MatrixOp(randn(m, n)), 1, m)
    dim_out = (4, m, 5)
    B_noncompact = BroadCast(A, dim_out; threaded = false)
    x = randn(n)
    y = B_noncompact * x
    @test size(y) == dim_out
    y_test = randn(dim_out)
    x_back = B_noncompact' * y_test
    @test size(x_back) == (n,)
end

@testitem "BroadCast: copy_operator" tags = [:calculus, :BroadCast] setup = [TestUtils] begin
    using Random, AbstractOperators, LinearAlgebra
    Random.seed!(7)

    m, n = 8, 4
    dim_out = (m, 10)

    # NoOperatorBroadCast branch (identity input)
    opEye = Eye(m)
    opNo = BroadCast(opEye, dim_out)
    opNo2 = copy_operator(opNo; threaded = true)
    @test opNo2 isa AbstractOperators.NoOperatorBroadCast
    x = randn(m)
    y1 = zeros(dim_out)
    y2 = zeros(dim_out)
    mul!(y1, opNo, x)
    mul!(y2, opNo2, x)
    @test y1 ≈ y2

    # OperatorBroadCast branch (wraps another operator)
    opA = MatrixOp(randn(m, n))
    opWrapped = BroadCast(opA, dim_out)
    opWrapped2 = copy_operator(opWrapped; threaded = true)
    @test opWrapped2 isa AbstractOperators.OperatorBroadCast
    x2 = randn(n)
    y3 = zeros(dim_out)
    y4 = zeros(dim_out)
    mul!(y3, opWrapped, x2)
    mul!(y4, opWrapped2, x2)
    @test y3 ≈ y4
end

@testitem "BroadCast: threading traits" tags = [:calculus, :BroadCast] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(9)

    # NoOperatorBroadCast: `is_threaded` reflects its own `Th` type parameter directly.
    m = 1000
    dim_out_big = (m, 300)   # m*300 = 300000 > THRESHOLD_MEMORY_BOUND (2^18)
    opNo_big = BroadCast(Eye(m), dim_out_big; threaded = true)
    @test supports_threading(opNo_big) == true
    if Threads.nthreads() > 1
        @test is_threaded(opNo_big) == true
    end
    opNo_small = BroadCast(Eye(4), (4, 2); threaded = true)
    @test is_threaded(opNo_small) == false

    # OperatorBroadCast: `is_threaded` is true either from its own `Th` or from a threaded
    # child, even when the broadcast's own size is below its threshold.
    n = 2000
    threaded_child = Sin(Float64, (n,); threaded = true)   # n > transcendental threshold
    dim_out_small = (n, 2)   # 4000 elements, below THRESHOLD_MEMORY_BOUND
    opWrapped_child = BroadCast(threaded_child, dim_out_small; threaded = false)
    @test supports_threading(opWrapped_child) == true
    if Threads.nthreads() > 1
        @test is_threaded(opWrapped_child) == true
    end

    serial_child = Cos(Float64, (n,); threaded = false)
    opWrapped_serial = BroadCast(serial_child, dim_out_small; threaded = false)
    @test is_threaded(opWrapped_serial) == false
end

@testitem "OperatorBroadCast: threaded construction allocates per-thread state" tags = [
    :calculus, :BroadCast,
] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(3)

    if Threads.nthreads() > 1
        # dim_out sized above THRESHOLD_MEMORY_BOUND (2^18) so `threaded = true` is actually
        # granted, exercising OperatorBroadCast's threaded-construction branch (per-thread
        # domain buffers and per-thread operator copies), not just the serial one.
        m, n = 8, 4
        dim_out = (m, 40000)   # 320000 > 2^18
        A = MatrixOp(randn(m, n))
        opR = BroadCast(A, dim_out; threaded = true)
        @test is_threaded(opR) == true
        x = randn(n)
        y = opR * x
        @test y[:, 1] ≈ A * x
        y_test = randn(dim_out)
        x_back = opR' * y_test
        @test x_back ≈ A' * dropdims(sum(y_test, dims = 2), dims = 2)
    end
end
