@testitem "MyLinOp basic" tags = [:linearoperator, :MyLinOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m = 5, 4
    A = randn(n, m)
    op = MyLinOp(Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x))
    op_array_type = MyLinOp(
        Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x); array_type = Array{ComplexF32, 2}
    )
    @test domain_array_type(op_array_type) == Array{Float64}
    @test codomain_array_type(op_array_type) == Array{Float64}
    x1 = randn(m)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = A * x1
    @test y1 ≈ y2

    # size & types
    @test size(op) == ((n,), (m,))
    @test domain_type(op) == Float64
    @test codomain_type(op) == Float64
end

@testitem "MyLinOp mul and scaling" tags = [:linearoperator, :MyLinOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 4
    A = randn(n, m)
    op = MyLinOp(Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x))
    x1 = randn(m)
    z = randn(n)
    out_adj = similar(x1)
    mul!(out_adj, op', z)
    @test out_adj ≈ A' * z
    X = randn(m, 3)
    Y = zeros(n, 3)
    for j in 1:3
        mul!(view(Y, :, j), op, view(X, :, j))
    end
    @test Y ≈ A * X
    Sop = Scale(3.0, op)
    @test Sop * x1 ≈ 3.0 * (op * x1)
    @test_throws ErrorException Scale(1 + 2im, op)
end

@testitem "MyLinOp constructors and errors" tags = [:linearoperator, :MyLinOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 5, 4
    A = randn(n, m)
    op = MyLinOp(Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x))
    x1 = randn(m)
    @test_throws DimensionMismatch op * randn(m + 1)
    io = IOBuffer()
    show(io, op)
    s = String(take!(io))
    @test occursin("A", s)
    op2 = MyLinOp(Float64, (m,), Float64, (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x))
    @test size(op2) == ((n,), (m,))
    @test op2 * x1 ≈ A * x1
end

@testitem "AbstractOperator fallback properties" tags = [:misc, :MyLinOp] setup = [TestUtils] begin
    using AbstractOperators, LinearAlgebra, Random
    Random.seed!(0)
    n, m = 5, 4
    A = randn(n, m)
    op = MyLinOp(Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x))

    # is_thread_safe falls back to false for custom operators
    @test is_thread_safe(op) == false
    # is_sliced falls back to false
    @test is_sliced(op) == false
    # get_slicing_expr falls back to Colon() for non-null operators
    @test AbstractOperators.get_slicing_expr(op) == Colon()
    # get_slicing_mask throws for operators without specialization
    @test_throws ErrorException AbstractOperators.get_slicing_mask(op)
    # has_optimized_normalop falls back to false
    @test AbstractOperators.has_optimized_normalop(op) == false
    # get_normal_op falls back to L' * L
    normal = AbstractOperators.get_normal_op(op)
    x = randn(m)
    @test normal * x ≈ A' * (A * x)
    # diag throws for non-diagonal operators
    @test_throws ErrorException diag(op)
end

@testitem "MyLinOp (GPU)" tags = [:gpu, :linearoperator, :MyLinOp] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)
        n, m = 5, 4
        A = gpu_randn(backend, n, m)
        AT = gpu_wrapper(backend, randn(1))
        op = MyLinOp(
            Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x); array_type = AT
        )
        x = gpu_randn(backend, m)
        y = gpu_randn(backend, n)
        test_op(op, x, y, false)
        @test domain_array_type(op) <: AT
    end
end
