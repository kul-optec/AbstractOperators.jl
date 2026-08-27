@testmodule SimpleBatchOpHelpers begin
    using Random, BenchmarkTools, LinearAlgebra, AbstractOperators, JLArrays, Test

    function test_simple_batchop(op, batch_op, x, y, z, threaded)
        # Read the wrapped operator through the accessor rather than assuming which struct
        # `threaded = true` produced. `threaded` is a permission, not a command: below the
        # batch policy's size gate a threaded request yields a SingleThreaded batch op,
        # whose `.operator` is the operator itself -- so `.operator[1]` would silently index
        # *into* it (slicing a DiagOp) instead of failing usefully.
        @test AbstractOperators._wrapped_operator(batch_op) == op
        @test is_threaded(batch_op) == (
            threaded && Threads.nthreads() > 1 &&
                AbstractOperators._should_thread(op)
        )
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

    # This test asserts that batch-level threading beats the serial batch loop. Two things
    # keep that assertion honest rather than lucky:
    #
    # 1. Minimum across repetitions. A single `@belapsed` of this workload has ~30% spread
    #    on the serial arm, enough to flip a bare `t_multi < t_single`.
    # 2. Enough work per batch item (n) that the parallel gain dominates dispatch overhead.
    #    Going the other way -- smaller items, more of them -- was *measured* to hurt:
    #    at n=2000/20x20 and n=1000/30x30 the threaded arm came out slower than serial.
    #
    # The margin needed widening when FiniteDiff stopped allocating: that made the serial
    # baseline ~3x faster, so the honest speedup shrank from ~2.5x to ~1.7x even though
    # both arms got faster in absolute terms.
    function benchmark_threading(threaded; repeats = 3)
        n = 40000
        op = Compose(DiagOp(randn(n - 1)), FiniteDiff((n,), 1))
        batch_op = BatchOp(op, (10, 10), (:_, :b, :b); threaded)
        y = zeros(n - 1, 10, 10)
        return minimum(
            @belapsed(mul!($y, $batch_op, x), setup = ($y .= 0; x = rand($n, 10, 10)))
                for _ in 1:repeats
        )
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

    # `benchmark_threading` is the only place in the test suite where the wrapped operator
    # is large enough to cross `MIN_BATCH_WORK_FOR_PARALLEL`, and it is skipped whenever
    # `CI == "true"` (it asserts on wall-clock time). That leaves the *real* threaded branch
    # of `create_BatchOp` -- `_resolve_threaded`/`_per_thread_operators` actually choosing
    # `SimpleBatchOpMultiThreaded`, and its `mul!` running for genuine work -- untested in
    # CI: every other test's operator domain is a handful of elements, well under the
    # threshold, so `BatchOp(...; threaded = true)` always resolves to the single-threaded
    # struct there regardless of the keyword. This test crosses the threshold with a
    # correctness check only (no timing), so it runs everywhere `Threads.nthreads() > 1`,
    # CI included.
    function test_real_multithreaded_construction()
        n = 1200  # > MIN_BATCH_WORK_FOR_PARALLEL (2^10 = 1024)
        op = DiagOp(randn(n))
        batch_op = BatchOp(op, (3,), (:_, :b); threaded = true)
        @test batch_op isa AbstractOperators.SimpleBatchOpMultiThreaded
        @test is_threaded(batch_op)
        x = rand(n, 3)
        y = zeros(n, 3)
        z = zeros(n, 3)
        for i in 1:3
            mul!(@view(y[:, i]), op, @view(x[:, i]))
        end
        for i in 1:3
            mul!(@view(z[:, i]), op', @view(y[:, i]))
        end
        return test_simple_batchop(op, batch_op, x, y, z, true)
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

    # MultiThreaded-specific trait/equality/copy_operator paths, exercised without needing
    # a workload large enough to actually cross the batch threshold.
    @test is_threaded(mt) == true
    @test is_threaded(st) == false
    @test AbstractOperators._wrapped_operator(mt) == op
    @test mt == mt
    # `==` compares structure, not execution strategy, so a hand-built MultiThreaded and a
    # SingleThreaded batch op over the same operator/shape do compare equal.
    @test mt == st

    # `copy_operator` always re-derives the threaded flag from the size policy, not from
    # `is_threaded(mt)` directly (`threaded = true` is a permission, not a command) -- this
    # tiny `op` is far below the batch threshold, so the copy comes back SingleThreaded even
    # though `mt` itself was hand-built as MultiThreaded. Numerical behaviour is unaffected.
    mt_copy = copy_operator(mt)
    x = rand(2, 2)
    @test mt_copy * x ≈ mt * x
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

@testitem "SimpleBatchOp real multi-threaded construction" tags = [:batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers] begin
    using Random
    Random.seed!(0)
    if Threads.nthreads() > 1
        SimpleBatchOpHelpers.test_real_multithreaded_construction()
    end
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

@testitem "SimpleBatchOp (GPU)" tags = [:gpu, :batching, :SimpleBatchOp] setup = [TestUtils, SimpleBatchOpHelpers, GpuEnvSetup] begin
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
