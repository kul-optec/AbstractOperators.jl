@testmodule FiniteDiffTestHelper begin
    using Test, AbstractOperators, LinearAlgebra

    export test_finitediff_mul

    function test_finitediff_mul(conv, verb, test_op)
        n = 10
        op = FiniteDiff(conv(zeros(Float64, n)))
        test_op(op, conv(randn(n)), conv(randn(n - 1)), verb)

        n, m = 10, 5
        op = FiniteDiff(conv(zeros(Float64, n, m)))
        test_op(op, conv(randn(n, m)), conv(randn(n - 1, m)), verb)

        n, m = 10, 5
        op = FiniteDiff(conv(zeros(Float64, n, m)), 2)
        test_op(op, conv(randn(n, m)), conv(randn(n, m - 1)), verb)
    end

end  # @testmodule FiniteDiffTestHelper

@testitem "FiniteDiff: basic mul" tags = [:linearoperator, :FiniteDiff] setup = [TestUtils, FiniteDiffTestHelper] begin
    using Random, SparseArrays, AbstractOperators
    Random.seed!(0)

    test_finitediff_mul(identity, verb, test_op)

    test_finitediff_mul(identity, verb, test_op)

    n = 10
    op = FiniteDiff(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n - 1), verb)
    y1 = op * collect(range(0; stop = 1, length = n))
    @test all(norm.(y1 .- 1 / 9) .<= 1.0e-12)

    I1, J1, V1 = SparseArrays.spdiagm_internal(0 => ones(n - 1))
    I2, J2, V2 = SparseArrays.spdiagm_internal(1 => ones(n - 1))
    B = -sparse(I1, J1, V1, n - 1, n) + sparse(I2, J2, V2, n - 1, n)
    @test norm(B * x1 - op * x1) <= 1.0e-8

    n, m = 10, 5
    op = FiniteDiff(Float64, (n, m), 2)
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(n, m - 1), verb)
    y1 = op * repeat(collect(range(0; stop = 1, length = n)), 1, m)
    @test all(norm.(y1) .<= 1.0e-12)

    @test_throws ErrorException FiniteDiff(Float64, (n, m), 4)
    FiniteDiff((n, m))
    FiniteDiff(x1)
end

@testitem "FiniteDiff: properties" tags = [:linearoperator, :FiniteDiff] setup = [TestUtils, FiniteDiffTestHelper] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, m, l, i = 5, 6, 2, 3
    op = FiniteDiff(Float64, (n, m, l, i), 4)
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == false

    n = 6
    F = FiniteDiff(Float64, (n,))
    g = randn(n - 1)
    x = randn(n)
    lhs = dot(F * x, g)
    tmp = zeros(n)
    mul!(tmp, F', g)
    rhs = dot(x, tmp)
    @test lhs ≈ rhs atol = 1.0e-10

    io = IOBuffer()
    show(io, F)
    @test occursin("δx", String(take!(io)))
    Fy = FiniteDiff(Float64, (3, 4), 2)
    io = IOBuffer()
    show(io, Fy)
    @test occursin("δy", String(take!(io)))
    @test size(FiniteDiff(Float64, (3, 4, 5), 2)) == ((3, 3, 5), (3, 4, 5))
end

@testitem "FiniteDiff (GPU)" tags = [:gpu, :linearoperator, :FiniteDiff] setup = [TestUtils] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n = 10
        op = FiniteDiff(Float64, (n,); array_type = gpu_wrapper(backend, Float64, n))
        test_op(op, gpu_randn(backend, n), gpu_randn(backend, n - 1), false)

        n, m = 10, 5
        op = FiniteDiff(Float64, (n, m); array_type = gpu_wrapper(backend, Float64, n, m))
        test_op(op, gpu_randn(backend, n, m), gpu_randn(backend, n - 1, m), false)

        op2 = FiniteDiff(Float64, (n, m), 2; array_type = gpu_wrapper(backend, Float64, n, m))
        test_op(op2, gpu_randn(backend, n, m), gpu_randn(backend, n, m - 1), false)
    end
end
