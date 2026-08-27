@testmodule ZeroPadTestHelper begin
    using Test, AbstractOperators, LinearAlgebra

    export test_zeropad_mul

    function test_zeropad_mul(conv, verb, test_op, to_cpu, norm)
        n = (3,)
        z = (5,)
        op = ZeroPad(conv(zeros(Float64, n)), z)
        x1 = conv(randn(n))
        y1 = test_op(op, x1, conv(randn(n .+ z)), verb)
        y2 = conv([collect(x1); zeros(5)])
        @test norm(to_cpu(y1) .- to_cpu(y2)) <= 1.0e-12

        n = (3, 2)
        z = (5, 3)
        op = ZeroPad(conv(zeros(Float64, n)), z)
        x1 = conv(randn(n))
        y1 = test_op(op, x1, conv(randn(n .+ z)), verb)
        y2c = zeros(n .+ z)
        y2c[1:n[1], 1:n[2]] = collect(x1)
        @test norm(to_cpu(y1) .- y2c) <= 1.0e-12

        n = (3, 2, 2)
        z = (5, 3, 1)
        op = ZeroPad(conv(zeros(Float64, n)), z)
        x1 = conv(randn(n))
        test_op(op, x1, conv(randn(n .+ z)), verb)
    end

end  # @testmodule ZeroPadTestHelper

@testitem "ZeroPad" tags = [:linearoperator, :ZeroPad] setup = [TestUtils, ZeroPadTestHelper] begin
    using Random, AbstractOperators
    Random.seed!(0)

    test_zeropad_mul(identity, verb, test_op, to_cpu, norm)

    # CPU-only: type-based constructors, properties, errors
    n = (3,)
    z = (5,)
    op = ZeroPad(Float64, n, z)
    x1 = randn(n)
    @test size(op) == (n .+ z, n)
    @test domain_type(op) == Float64
    @test codomain_type(op) == Float64
    @test is_thread_safe(op) == true
    @test AbstractOperators.has_fast_opnorm(op) == true
    @test opnorm(op) == 1

    n = (3, 2, 2)
    z = (5, 3, 1)
    op = ZeroPad(Float64, n, z)
    x1 = randn(n)

    # Normal operator should be identity on input space
    Nop = AbstractOperators.get_normal_op(op)
    @test Nop * x1 ≈ x1

    # Adjoint crop
    ybig = zeros(n .+ z)
    ybig[1:n[1], 1:n[2], 1:n[3]] .= x1
    xcropped = zeros(n)
    mul!(xcropped, op', ybig)
    @test xcropped ≈ x1

    # Scaling
    Sop = Scale(2.0, op)
    @test Sop * x1 ≈ 2.0 * (op * x1)
    @test_throws ErrorException Scale(1 + 2im, op)

    # other constructors
    ZeroPad(n, z...)
    ZeroPad(Float64, n, z...)
    ZeroPad(x1, z)
    ZeroPad(x1, z...)

    # errors
    @test_throws ErrorException ZeroPad(Float64, n, (1, 2))
    @test_throws ErrorException ZeroPad(Float64, n, (1, -2, 3))

    # properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == false
    @test is_full_column_rank(op) == true
    @test diag_AcA(op) == 1

    io = IOBuffer()
    show(io, op)
    s = String(take!(io))
    @test occursin("[I;0]", s)
end

@testitem "ZeroPad (GPU)" tags = [:gpu, :linearoperator, :ZeroPad] setup = [TestUtils, ZeroPadTestHelper, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)
        conv = x -> to_gpu(backend, x)
        test_zeropad_mul(conv, false, test_op, collect, norm)
    end
end
