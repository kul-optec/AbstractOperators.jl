@testitem "Variation: basic mul" tags = [:linearoperator, :Variation] setup = [TestUtils] begin
    using Random, SparseArrays, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing Variation: basic mul --- ")

    function test_variation_mul(conv, verb)
        n, m = 10, 5
        op = Variation(conv(zeros(Float64, n, m)); threaded = false)
        test_op(op, conv(randn(n, m)), conv(randn(n * m, 2)), verb)
    end

    test_variation_mul(identity, verb)

    for threaded in (false, true)
        n, m = 10, 5
        verb && println("  - threaded = $threaded")
        op = Variation(Float64, (n, m); threaded)
        op_array_type = Variation(Float64, (n, m); threaded, array_type = Array{ComplexF32, 2})
        @test domain_array_type(op_array_type) == Array{Float64}
        @test codomain_array_type(op_array_type) == Array{Float64}
        x1 = randn(n, m)
        y1 = test_op(op, x1, randn(m * n, 2), verb)
        # size & types
        @test size(op) == ((n * m, 2), (n, m))
        @test domain_type(op) == Float64
        @test codomain_type(op) == Float64
        @test is_thread_safe(op) == true

        # Forward difference on simple ramp in first dimension, constant in second
        y1 = op * repeat(collect(range(0; stop = 1, length = n)), 1, m)
        @test all(norm.(y1[:, 1] .- 1 / (n - 1)) .<= 1.0e-12)
        @test all(norm.(y1[:, 2]) .<= 1.0e-12)
        # Constant input gives zero
        const_in = fill(3.14, n, m)
        @test op * const_in ≈ zeros(n * m, 2)

        Dx = spdiagm(0 => ones(n), -1 => -ones(n - 1))
        Dx[1, 1], Dx[1, 2] = -1, 1
        Dy = spdiagm(0 => ones(m), -1 => -ones(m - 1))
        Dy[1, 1], Dy[1, 2] = -1, 1

        Dxx = kron(sparse(I, m, m), Dx)
        Dyy = kron(Dy, sparse(I, n, n))
        TV = [Dxx; Dyy]

        x1 = randn(n, m)
        @test norm(op * x1 - reshape(TV * (x1[:]), n * m, 2)) < 1.0e-12
    end
end

@testitem "Variation: 3D mul and constructors" tags = [:linearoperator, :Variation] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing Variation: 3D mul and constructors --- ")

    for threaded in (false, true)
        n, m, l = 100, 50, 30
        verb && println("  - threaded = $threaded")
        op = Variation(Float64, (n, m, l); threaded)
        x1 = randn(n, m, l)
        y1 = test_op(op, x1, randn(m * n * l, 3), verb)
        @test size(op) == ((n * m * l, 3), (n, m, l))
        y1 = op * reshape(repeat(collect(range(0; stop = 1, length = n)), 1, m * l), n, m, l)
        @test all(norm.(y1[:, 1] .- 1 / (n - 1)) .<= 1.0e-12)
        @test all(norm.(y1[:, 2]) .<= 1.0e-12)
        @test all(norm.(y1[:, 3]) .<= 1.0e-12)
        # Constant 3D input zero output
        const3 = fill(-2.0, n, m, l)
        @test op * const3 ≈ zeros(n * m * l, 3)

        ### other constructors
        Variation(Float64, n, m)
        Variation((n, m))
        Variation(n, m)
        Variation(x1)

        ##errors
        @test_throws ErrorException Variation(Float64, (n,))
        badX = randn(n, m + 1)
        @test_throws DimensionMismatch op * badX
    end
end

@testitem "Variation: adjoint and properties" tags = [:linearoperator, :Variation] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing Variation: adjoint and properties --- ")

    for threaded in (false, true)
        n, m, l = 100, 50, 30
        op = Variation(Float64, (n, m, l); threaded)

        # Adjoint consistency: <Vx, Y> == <x, V'Y>
        x_test = randn(n, m)
        verb && println("  - threaded = $threaded")
        V = Variation(Float64, (n, m); threaded)
        Y = randn(n * m, 2)
        lhs = dot(vec(V * x_test), vec(Y))  # vec(Vx) ⋅ vec(Y)
        z = zeros(n, m)
        mul!(z, V', Y)
        rhs = dot(vec(x_test), vec(z))
        @test abs(lhs - rhs) <= 1.0e-10 * (1 + abs(lhs))

        # In-place mul! forward and adjoint
        Yf = zeros(n * m, 2)
        mul!(Yf, V, x_test)
        @test Yf == V * x_test
        Zb = zeros(n, m)
        mul!(Zb, V', Yf)
        @test Zb == V' * Yf

        # Scaling
        S = Scale(2.0, V)
        @test S * x_test ≈ 2.0 * (V * x_test)
        @test_throws ErrorException Scale(1 + 2im, V)

        # Show output symbol
        io = IOBuffer()
        show(io, V)
        s = String(take!(io))
        @test occursin("Ʋ", s)

        ###properties
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
    end
end

@testitem "Variation: adjoint matches dense transpose for every dimension size" tags = [
    :linearoperator, :Variation,
] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing Variation: adjoint vs dense transpose --- ")

    # Regression for a `BoundsError` in the adjoint whenever *any* dimension had size
    # exactly 2 (`Variation(Float64, (8, 2))'`): the `i == 2` branch was tested before the
    # `i == size(y, d)` branch, so the last column of a length-2 dimension took the interior
    # formula and read one element past the end. Shapes below deliberately place the 2 in
    # leading, trailing, and interior positions, and include the all-2s case.
    shapes = [
        (2, 2), (2, 3), (3, 2), (8, 2), (2, 8), (8, 3), (4, 4), (5, 7),
        (2, 2, 2), (8, 2, 3), (8, 3, 2), (2, 3, 4), (3, 2, 2), (4, 4, 4),
        (2, 2, 2, 2), (3, 2, 4, 2),
    ]
    for dims in shapes, T in (Float64, Float32), threaded in (false, true)
        op = Variation(T, dims; threaded)
        n, N = prod(dims), length(dims)

        # Dense forward matrix, built column by column from the images of the basis vectors.
        # The adjoint must equal its exact transpose -- a stronger check than the dot-product
        # identity alone, which can hide compensating sign errors.
        M = zeros(T, n * N, n)
        for j in 1:n
            e = zeros(T, dims)
            e[j] = one(T)
            M[:, j] = vec(op * e)
        end

        b = randn(T, n, N)
        z = zeros(T, dims)
        mul!(z, op', b)
        @test z ≈ reshape(M' * vec(b), dims)

        # <Vx, b> == <x, V'b>
        x = randn(T, dims)
        @test isapprox(
            dot(vec(op * x), vec(b)), dot(vec(x), vec(z)); rtol = sqrt(eps(T))
        )
    end
end

@testitem "Variation: adjoint allocates nothing" tags = [
    :linearoperator, :Variation,
] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    # Guards a regression that correctness tests cannot see. Sharing the adjoint body between
    # the threaded and serial methods is only safe if the shared function stays statically
    # dispatched: passing the dimension count as `Val(N)` made Polyester's `@batch` closure
    # build the `Val` from a runtime value, so every element dispatched dynamically and
    # allocated -- 293 MB at n = 2^22, turning a 4.5x speedup into 0.25x. Timings are too
    # noisy to assert on directly; the allocation count is not.
    for threaded in (false, true), dims in ((4096, 4), (64, 4), (8, 2, 3))
        op = Variation(Float64, dims; threaded)
        b = randn(prod(dims), length(dims))
        z = zeros(dims)
        adj = op'
        mul!(z, adj, b)  # warm up: first call compiles
        mul!(z, adj, b)
        @test minimum(@allocated(mul!(z, adj, b)) for _ in 1:5) == 0
    end
end

@testitem "Variation: rejects singleton dimensions" tags = [
    :linearoperator, :Variation,
] setup = [TestUtils] begin
    using AbstractOperators
    # A size-1 dimension has no finite difference to take; it used to fail with a
    # `BoundsError` from inside the forward kernel. Every constructor must reject it, the
    # array-based one included (it delegates rather than building the struct directly).
    for dims in [(4, 1), (1, 4), (1, 1), (3, 1, 3)]
        @test_throws ArgumentError Variation(Float64, dims)
        @test_throws ArgumentError Variation(zeros(Float64, dims))
    end
    @test_throws ArgumentError Variation(4, 1)
    @test_throws ArgumentError Variation((4, 1))
end

@testitem "Variation: copy_operator" tags = [:linearoperator, :Variation] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(6)

    n, m = 10, 5
    op = Variation(zeros(Float64, n, m); threaded = false)
    op2 = copy_operator(op; threaded = true)
    @test op2 isa Variation
    x = randn(n, m)
    y1 = zeros(n * m, 2)
    y2 = zeros(n * m, 2)
    mul!(y1, op, x)
    mul!(y2, op2, x)
    @test y1 ≈ y2
end

@testitem "Variation (GPU)" tags = [:gpu, :linearoperator, :Variation] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)
        n, m = 10, 5
        op = Variation(gpu_zeros(backend, Float64, n, m); threaded = false)
        test_op(op, gpu_randn(backend, n, m), gpu_randn(backend, n * m, 2), false)
    end
end
