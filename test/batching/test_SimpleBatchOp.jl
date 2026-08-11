@testmodule SimpleBatchOpHelpers begin
    using Random, BenchmarkTools, LinearAlgebra, AbstractOperators, JLArrays, Test

    function test_simple_batchop(op, batch_op, x, y, z, threaded)
        if threaded && Threads.nthreads() > 1
            @test batch_op.operator[1] == op
        else
            @test batch_op.operator == op
        end
        @test size(batch_op, 1) == size(y)
        @test size(batch_op, 2) == size(x)
        y2 = batch_op * x
        mul!(y2, batch_op, x)
        @test y == y2
        z2 = batch_op' * y
        return @test z == z2
    end

    function test_shape_keeping_simple_batch_op(threaded)
        op = DiagOp([1.0im, 1.0im])
        batch_op = BatchOp(op, (3, 4), (:_, :b, :b); threaded)
        x = rand(ComplexF64, 2, 3, 4)
        y = zeros(ComplexF64, 2, 3, 4)
        for i in 1:3, j in 1:4
            mul!(@view(y[:, i, j]), op, @view(x[:, i, j]))
        end
        return test_simple_batchop(op, batch_op, x, y, x, threaded)
    end

    function test_variation_simple_batch_op(threaded)
        op = Variation(3, 4, 5; threaded = false)
        batch_op = BatchOp(op, (2, 6), (:b, :_, :_, :_, :b) => (:b, :_, :b, :_); threaded)
        x = rand(2, 3, 4, 5, 6)
        y = zeros(2, 60, 6, 3)
        z = similar(x)
        for i in 1:2, j in 1:6
            mul!(@view(y[i, :, j, :]), op, @view(x[i, :, :, :, j]))
        end
        for i in 1:2, j in 1:6
            mul!(@view(z[i, :, :, :, j]), op', @view(y[i, :, j, :]))
        end
        return test_simple_batchop(op, batch_op, x, y, z, threaded)
    end

    function benchmark_threading(threaded)
        n = 10000
        op = Compose(DiagOp(randn(n - 1)), FiniteDiff((n,), 1))
        batch_op = BatchOp(op, (10, 10), (:_, :b, :b); threaded)
        y = zeros(n - 1, 10, 10)
        return @belapsed(mul!($y, $batch_op, x), setup = ($y .= 0; x = rand($n, 10, 10)))
    end

    function test_shape_changing_simple_batch_op(threaded)
        n = 16
        op = Compose(DiagOp(randn(n - 1)), FiniteDiff((n,), 1))
        batch_op = BatchOp(op, (5, 6), (:_, :b, :b); threaded)
        x = randn(n, 5, 6)
        y = zeros(n - 1, 5, 6)
        z = zeros(n, 5, 6)
        for i in 1:5, j in 1:6
            mul!(@view(y[:, i, j]), op, @view(x[:, i, j]))
        end
        for i in 1:5, j in 1:6
            mul!(@view(z[:, i, j]), op', @view(y[:, i, j]))
        end
        return test_simple_batchop(op, batch_op, x, y, z, threaded)
    end

    function other_tests(threaded)
        op = DiagOp([1.0, 2.0])
        batch_op = BatchOp(op, (2,); threaded)
        io = IOBuffer(); show(io, batch_op); s = String(take!(io))
        @test occursin("⟳", s)
        batch_op_copy = AbstractOperators.copy_operator(batch_op)
        @test batch_op == batch_op_copy
        @test isequal(batch_op, batch_op_copy)
        @test domain_array_type(batch_op) == domain_array_type(op)
        @test codomain_array_type(batch_op) == codomain_array_type(op)
        @test is_linear(batch_op) == is_linear(op)
        @test is_eye(batch_op) == is_eye(op)
        @test is_null(batch_op) == is_null(op)
        @test is_diagonal(batch_op) == is_diagonal(op)
        @test is_AcA_diagonal(batch_op) == is_AcA_diagonal(op)
        @test is_AAc_diagonal(batch_op) == is_AAc_diagonal(op)
        @test is_invertible(batch_op) == is_invertible(op)
        @test is_full_row_rank(batch_op) == is_full_row_rank(op)
        @test is_full_column_rank(batch_op) == is_full_column_rank(op)
        @test is_sliced(batch_op) == is_sliced(op)
        @test is_thread_safe(batch_op) == is_thread_safe(op)
        @test AbstractOperators.has_optimized_normalop(batch_op) == AbstractOperators.has_optimized_normalop(op)
        n_op = AbstractOperators.get_normal_op(batch_op)
        @test typeof(n_op) <: typeof(batch_op)
        @test AbstractOperators.has_fast_opnorm(batch_op) == AbstractOperators.has_fast_opnorm(op)
        @test opnorm(batch_op) == opnorm(op)
        @test estimate_opnorm(batch_op) == estimate_opnorm(op)
        @test estimate_opnorm(batch_op) == opnorm(batch_op)
        @test diag(batch_op) == [diag(op)'; diag(op)']'
        @test diag_AcA(batch_op) == [diag_AcA(op)'; diag_AcA(op)']'
        @test diag_AAc(batch_op) == [diag_AAc(op)'; diag_AAc(op)']'
        x_bad = rand(Int, 2, 2)
        y_bad = zeros(2, 2)
        @test_throws ArgumentError mul!(y_bad, batch_op, x_bad)
        x_bad2 = rand(2, 3)
        @test_throws DimensionMismatch mul!(y_bad, batch_op, x_bad2)
        y_bad2 = rand(Int, 2, 2)
        x_good = rand(2, 2)
        @test_throws ArgumentError mul!(y_bad2, batch_op, x_good)
        y_bad3 = zeros(3, 2)
        @test_throws DimensionMismatch mul!(y_bad3, batch_op, x_good)
        eye_batch = BatchOp(Eye(Float64, (2,)), (2,); threaded)
        @test diag(eye_batch) == 1.0
        @test diag_AcA(eye_batch) == 1.0
        @test diag_AAc(eye_batch) == 1.0
        return
    end
end

@testitem "SimpleBatchOp shape-keeping non-threaded" tags = [:batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random
    Random.seed!(0)
    SimpleBatchOpHelpers.test_shape_keeping_simple_batch_op(false)
end

@testitem "SimpleBatchOp shape-keeping threaded" tags = [:batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random
    Random.seed!(0)
    if Threads.nthreads() > 1
        SimpleBatchOpHelpers.test_shape_keeping_simple_batch_op(true)
    end
end

@testitem "SimpleBatchOp variation non-threaded" tags = [:batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random
    Random.seed!(0)
    SimpleBatchOpHelpers.test_variation_simple_batch_op(false)
end

@testitem "SimpleBatchOp variation threaded" tags = [:batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random
    Random.seed!(0)
    if Threads.nthreads() > 1
        SimpleBatchOpHelpers.test_variation_simple_batch_op(true)
    end
end

@testitem "SimpleBatchOp shape-changing non-threaded" tags = [:batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random
    Random.seed!(0)
    SimpleBatchOpHelpers.test_shape_changing_simple_batch_op(false)
end

@testitem "SimpleBatchOp shape-changing threaded" tags = [:batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random
    Random.seed!(0)
    if Threads.nthreads() > 1
        SimpleBatchOpHelpers.test_shape_changing_simple_batch_op(true)
    end
end

@testitem "SimpleBatchOp other tests non-threaded" tags = [:batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random
    Random.seed!(0)
    SimpleBatchOpHelpers.other_tests(false)
end

@testitem "SimpleBatchOp other tests threaded" tags = [:batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random
    Random.seed!(0)
    if Threads.nthreads() > 1
        SimpleBatchOpHelpers.other_tests(true)
    end
end

@testitem "SimpleBatchOpMultiThreaded properties" tags = [:batching, :SimpleBatchOp] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    # Directly construct SimpleBatchOpMultiThreaded to test property/diag methods
    # without requiring nthreads() > 1 at test time.
    op = DiagOp([1.0, 2.0])
    st = BatchOp(op, (2,); threaded = false)  # creates SimpleBatchOpSingleThreaded
    @assert st isa AbstractOperators.SimpleBatchOpSingleThreaded
    # Build MultiThreaded variant with same shape, 2 operator copies
    mt = let T = typeof(st)
        dT = T.parameters[1]
        cT = T.parameters[2]
        dM = T.parameters[3]
        cM = T.parameters[4]
        opT = typeof(op)
        N = length(st.domain_size)
        M = length(st.codomain_size)
        C = 2
        ops = (op, copy_operator(op))
        AbstractOperators.SimpleBatchOpMultiThreaded{dT, cT, dM, cM, opT, N, M, C}(
            ops, st.domain_size, st.codomain_size, CartesianIndices(st.batch_size)
        )
    end
    @test diag_AAc(mt) == diag_AAc(st)
    @test diag_AcA(mt) == diag_AcA(st)
    @test diag(mt) == diag(st)
    @test AbstractOperators.has_optimized_normalop(mt) == AbstractOperators.has_optimized_normalop(st)
    @test opnorm(mt) == opnorm(st)
    @test estimate_opnorm(mt) == estimate_opnorm(st)
    # Eye operator: scalar diag paths
    eye_op = Eye(Float64, (2,))
    eye_st = BatchOp(eye_op, (2,); threaded = false)
    eye_mt = let T = typeof(eye_st)
        dT, cT, dM, cM = T.parameters[1], T.parameters[2], T.parameters[3], T.parameters[4]
        opT = typeof(eye_op)
        N, M, C = length(eye_st.domain_size), length(eye_st.codomain_size), 2
        ops = (eye_op, copy_operator(eye_op))
        AbstractOperators.SimpleBatchOpMultiThreaded{dT, cT, dM, cM, opT, N, M, C}(
            ops, eye_st.domain_size, eye_st.codomain_size, CartesianIndices(eye_st.batch_size)
        )
    end
    @test diag(eye_mt) == 1.0
    @test diag_AcA(eye_mt) == 1.0
    @test diag_AAc(eye_mt) == 1.0
end

@testitem "SimpleBatchOp benchmark" tags = [:batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random
    Random.seed!(0)
    if Threads.nthreads() > 1 && get(ENV, "CI", "false") == "false"
        t_single_threaded = SimpleBatchOpHelpers.benchmark_threading(false)
        t_multi_threaded = SimpleBatchOpHelpers.benchmark_threading(true)
        @test t_multi_threaded < t_single_threaded
    end
end

@testitem "SimpleBatchOp (GPU)" tags = [:gpu, :batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)
        op = DiagOp(to_gpu(backend, [1.0, 2.0]))
        batch_op = BatchOp(op, (3, 4), (:_, :b, :b))
        x = gpu_ones(backend, Float64, 2, 3, 4)
        y_gpu = batch_op * x
        @test size(Array(y_gpu)) == (2, 3, 4)
        @test all(Array(y_gpu)[1, :, :] .≈ 1.0)
        @test all(Array(y_gpu)[2, :, :] .≈ 2.0)
        y_gpu2 = similar(y_gpu)
        mul!(y_gpu2, batch_op, x)
        @test Array(y_gpu2) ≈ Array(y_gpu)
    end
end
