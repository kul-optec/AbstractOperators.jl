@testmodule DiagOpTestHelper begin
    using Test, AbstractOperators, LinearAlgebra

    export test_diagop_mul

    function test_diagop_mul(conv, verb, test_op, to_cpu, norm)
        n = 4
        d = conv(randn(n))
        op = DiagOp(d)
        x1 = conv(randn(n))
        y1 = test_op(op, x1, conv(randn(n)), verb)
        y2 = d .* x1
        @test norm(to_cpu(y1) .- to_cpu(y2)) <= 1.0e-12

        d = conv(randn(n) + im * randn(n))
        op = DiagOp(d)
        x1 = conv(randn(n) .+ im * randn(n))
        y1 = test_op(op, x1, conv(randn(n) .+ im * randn(n)), verb)
        y2 = d .* x1
        @test norm(to_cpu(y1) .- to_cpu(y2)) <= 1.0e-12
    end

end  # @testmodule DiagOpTestHelper

@testitem "DiagOp: mul and constructors" tags = [:linearoperator, :DiagOp] setup = [TestUtils, DiagOpTestHelper] begin
    using Random, AbstractOperators
    Random.seed!(0)

    test_diagop_mul(identity, verb, test_op, to_cpu, norm)

    # scalar diagonal tests (CPU-only, not array-dependent)
    n = 4
    x1 = randn(n)
    d = pi
    op = DiagOp(Float64, (n,), d)
    y1 = test_op(op, x1, randn(n), verb)
    @test all(norm.(y1 .- d .* x1) .<= 1.0e-12)

    d = im
    op = DiagOp(Float64, (n,), d)
    y1 = test_op(op, x1, randn(n) + im * randn(n), verb)
    @test domain_type(op) == Float64
    @test codomain_type(op) == Complex{Float64}

    # other constructors
    op = DiagOp(randn(4))
    op = DiagOp(randn(4) .+ im)
    op = DiagOp((n,), pi)
end

@testitem "DiagOp: properties and helpers" tags = [:linearoperator, :DiagOp] setup = [TestUtils, DiagOpTestHelper] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n = 4
    op = DiagOp((n,), pi)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == true
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(DiagOp(ones(10))) == true
    @test is_invertible(DiagOp([ones(5); 0])) == false
    @test is_full_row_rank(op) == true
    @test is_full_row_rank(DiagOp([ones(5); 0])) == false
    @test is_full_column_rank(op) == true
    @test is_full_column_rank(DiagOp([ones(5); 0])) == false

    @test diag(op) == pi
    d = pi
    op = DiagOp((n,), d)
    x1 = randn(n)
    @test norm(op' * (op * x1) .- diag_AcA(op) .* x1) <= 1.0e-12
    @test norm(op * (op' * x1) .- diag_AAc(op) .* x1) <= 1.0e-12

    # Scale
    op_scaled = Scale(3.0, op)
    @test diag(op_scaled) == 3.0 .* diag(op)
    @test size(op_scaled) == size(op)

    # get_normal_op
    normal_op = AbstractOperators.get_normal_op(op)
    @test diag(normal_op) == abs2.(diag(op))
    @test is_diagonal(normal_op) == true

    # storage and type related helpers
    @test is_thread_safe(op) == true
    @test AbstractOperators.has_fast_opnorm(op) == true
    @test opnorm(op) == maximum(abs, diag(op))
    @test estimate_opnorm(op) == maximum(abs, diag(op))
    @test AbstractOperators.has_optimized_normalop(op) == true
    @test AbstractOperators.has_optimized_normalop(op') == true

    # invertibility false path
    op_sing = DiagOp([1.0, 0.0, 2.0, 3.0])
    @test is_invertible(op_sing) == false
    @test is_full_row_rank(op_sing) == false
    @test is_full_column_rank(op_sing) == false

    @test size(op) == ((n,), (n,))
end

@testitem "DiagOp: copy_operator" tags = [:linearoperator, :DiagOp] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(5)

    n = 6
    d = randn(n)
    op = DiagOp(d)
    op2 = copy_operator(op; threaded = true)
    @test op2 isa DiagOp
    # DiagOp's diagonal `d` is treated as read-only data and shared (not deep-copied)
    # when storage_type is not given, so op2 may be egal to op for these immutable structs.
    x = randn(n)
    y1 = zeros(n)
    y2 = zeros(n)
    mul!(y1, op, x)
    mul!(y2, op2, x)
    @test y1 ≈ y2

    op3 = copy_operator(op; storage_type = Array)
    @test op3 isa DiagOp
    @test op3.d !== op.d
end

@testitem "DiagOp (GPU)" tags = [:gpu, :linearoperator, :DiagOp] setup = [TestUtils, DiagOpTestHelper] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)
        conv = x -> to_gpu(backend, x)
        test_diagop_mul(conv, false, test_op, collect, norm)
        x = gpu_randn(backend, 4)
        op = DiagOp(x)
        @test domain_array_type(op) <: backend.array_type
        @test codomain_array_type(op) <: backend.array_type
    end
end
