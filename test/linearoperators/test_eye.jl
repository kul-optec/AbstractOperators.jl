@testmodule EyeTestHelper begin
    using Test, AbstractOperators, LinearAlgebra

    export test_eye_mul

    function test_eye_mul(conv, verb, test_op, to_cpu, norm)
        n = 4
        x1 = conv(randn(n))
        op = Eye(x1)
        y1 = test_op(op, x1, conv(randn(n)), verb)
        @test norm(to_cpu(y1) .- to_cpu(x1)) <= 1.0e-12

        x2 = conv(randn(n, n))
        op2 = Eye(x2)
        test_op(op2, x2, conv(randn(n, n)), verb)
    end

end  # @testmodule EyeTestHelper

@testitem "Eye" tags = [:linearoperator, :Eye] setup = [TestUtils, EyeTestHelper] begin
    using Random, AbstractOperators
    Random.seed!(0)

    test_eye_mul(identity, verb, test_op, to_cpu, norm)

    n = 4
    op = Eye(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)

    @test all(norm.(y1 .- x1) .<= 1.0e-12)

    # other constructors
    op = Eye(Float64, (n,))
    op = Eye((n,))
    op = Eye(n)
    op = Eye(x1)

    # properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == true
    @test is_diagonal(op) == true
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == true
    @test is_symmetric(op) == true
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true

    @test diag(op) == 1
    @test diag_AcA(op) == 1
    @test diag_AAc(op) == 1

    # Different element types
    opC = Eye(ComplexF64, (n,))
    xC = randn(ComplexF64, n)
    @test all(opC * xC .== xC)
    opI = Eye(Int, (n,))
    xI = rand(1:10, n)
    @test all(opI * xI .== xI)

    # Multi-dimensional
    op3 = Eye(Float64, (2, 3, 4))
    x3 = randn(2, 3, 4)
    @test all(op3 * x3 .== x3)
    @test size(op3) == ((2, 3, 4), (2, 3, 4))

    # Storage type helpers
    @test domain_type(op) == Float64
    @test codomain_type(op) == Float64
    @test domain_array_type(op) == Array{Float64}
    @test codomain_array_type(op) == Array{Float64}
    @test is_thread_safe(op) == true

    # Adjoint, opnorm, get_normal_op
    @test AdjointOperator(op) === op
    @test AbstractOperators.has_fast_opnorm(op) == true
    @test opnorm(op) == 1.0
    @test opnorm(op) == estimate_opnorm(op)
    @test AbstractOperators.has_optimized_normalop(op) == true
    @test AbstractOperators.get_normal_op(op) === op

    # In-place mul!
    y = similar(x1)
    fill!(y, 0)
    mul!(y, op, x1)
    @test all(y .== x1)
end

@testitem "Eye (GPU)" tags = [:gpu, :linearoperator, :Eye] setup = [TestUtils, EyeTestHelper, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)
        conv = x -> to_gpu(backend, x)
        test_eye_mul(conv, false, test_op, collect, norm)
    end
end
