@testmodule SpreadingBatchOpHelpers begin
    using Random, BenchmarkTools, LinearAlgebra, AbstractOperators, Test

    function test_spreading_batchop(operators, batch_op, x, y, z, threaded)
        if !threaded
            @test batch_op.operators == operators
        end
        @test size(batch_op, 1) == size(y)
        @test size(batch_op, 2) == size(x)
        y2 = batch_op * x
        @test y == y2
        z2 = batch_op' * y
        return @test z == z2
    end

    function test_shape_keeping_threadsafe_spreading_batch_op(threaded)
        ops = [i * DiagOp([1.0im, 2.0im]) for i in 1:3]
        batch_op = BatchOp(ops, 4; threaded)
        x = rand(ComplexF64, 2, 3, 4)
        y = zeros(ComplexF64, 2, 3, 4)
        for i in 1:3, j in 1:4
            mul!(@view(y[:, i, j]), ops[i], @view(x[:, i, j]))
        end
        z = similar(x)
        for i in 1:3, j in 1:4
            mul!(@view(z[:, i, j]), ops[i]', @view(y[:, i, j]))
        end
        return test_spreading_batchop(ops, batch_op, x, y, z, threaded)
    end

    function test_shape_changing_threadsafe_spreading_batch_op(threaded)
        ops = [i * Variation(3, 4, 5) for i in 1:2]
        batch_op = BatchOp(ops, 6, (:s, :_, :_, :_, :b) => (:s, :_, :b, :_); threaded)
        x = rand(2, 3, 4, 5, 6)
        y = zeros(2, 60, 6, 3)
        z = similar(x)
        for i in 1:2, j in 1:6
            mul!(@view(y[i, :, j, :]), ops[i], @view(x[i, :, :, :, j]))
        end
        for i in 1:2, j in 1:6
            mul!(@view(z[i, :, :, :, j]), ops[i]', @view(y[i, :, j, :]))
        end
        return test_spreading_batchop(ops, batch_op, x, y, z, threaded)
    end

    function test_nonthreadsafe_spreading_batch_op(threaded, threading_strategy)
        n, m = 10, 15
        num_ops = Threads.nthreads() + 5
        ops = [DiagOp(rand(m - 1)) * FiniteDiff((m,)) for i in 1:num_ops]
        batch_op = BatchOp(ops, n, (:b, :s, :_); threaded, threading_strategy)
        x = rand(n, num_ops, m)
        y = zeros(n, num_ops, m - 1)
        z = similar(x)
        for i in 1:n, j in 1:num_ops
            mul!(@view(y[i, j, :]), ops[j], @view(x[i, j, :]))
        end
        for i in 1:n, j in 1:num_ops
            mul!(@view(z[i, j, :]), ops[j]', @view(y[i, j, :]))
        end
        return test_spreading_batchop(ops, batch_op, x, y, z, threaded)
    end

    function test_failing_nonthreadsafe_spreading_batch_op()
        n, m = 10, 15
        num_ops = Threads.nthreads() + 5
        op = GetIndex(Float64, (m - 1,), 1:6) * FiniteDiff((m,))
        ops = [reshape(i * op, 2, 3) for i in 1:num_ops]
        return @test_throws ArgumentError BatchOp(ops, n, (:b, :s, :_) => (:b, :s, :_, :_); threaded = true, threading_strategy = AbstractOperators.ThreadingStrategy.FIXED_OPERATOR)
    end

    function benchmark_threading_strategy(threaded, threading_strategy)
        n, m = 300, 500
        num_ops = Threads.nthreads() + 50
        ops = [DiagOp(rand(m - 1)) * FiniteDiff((m,)) for i in 1:num_ops]
        batch_op = BatchOp(ops, n, (:_, :s, :b); threaded, threading_strategy)
        y = zeros(m - 1, num_ops, n)
        return @belapsed(mul!($y, $batch_op, x), setup = ($y .= 0; x = rand($m, $num_ops, $n)))
    end

    function other_spreadingbatchop_tests(threaded)
        ops = [DiagOp([1.0, 2.0]) for _ in 1:3]
        bop = BatchOp(ops, 4; threaded = threaded)
        io = IOBuffer(); show(io, bop); s = String(take!(io))
        @test occursin("⟳", s)
        cod, dom = size(bop)
        @test cod == size(bop, 1) && dom == size(bop, 2)
        @test domain_array_type(bop) == domain_array_type(ops[1])
        @test codomain_array_type(bop) == codomain_array_type(ops[1])
        @test is_linear(bop) == is_linear(ops[1])
        @test is_eye(bop) == is_eye(ops[1])
        @test is_null(bop) == is_null(ops[1])
        @test is_diagonal(bop) == is_diagonal(ops[1])
        @test is_AcA_diagonal(bop) == is_AcA_diagonal(ops[1])
        @test is_AAc_diagonal(bop) == is_AAc_diagonal(ops[1])
        @test is_invertible(bop) == is_invertible(ops[1])
        @test is_full_row_rank(bop) == is_full_row_rank(ops[1])
        @test is_full_column_rank(bop) == is_full_column_rank(ops[1])
        @test is_sliced(bop) == is_sliced(ops[1])
        @test is_thread_safe(bop) == is_thread_safe(ops[1])
        @test AbstractOperators.has_optimized_normalop(bop) == AbstractOperators.has_optimized_normalop(ops[1])
        nbop = AbstractOperators.get_normal_op(bop)
        @test size(nbop, 1) == size(bop, 1) && size(nbop, 2) == size(bop, 2)
        @test opnorm(bop) == maximum(opnorm.(ops))
        @test estimate_opnorm(bop) == maximum(estimate_opnorm.(ops))
        @test estimate_opnorm(bop) == opnorm(bop)
        @test diag(bop) == repeat(diag(ops[1]), outer = (1, 3, 4))
        @test diag_AcA(bop) == repeat(diag_AcA(ops[1]), outer = (1, 3, 4))
        @test diag_AAc(bop) == repeat(diag_AAc(ops[1]), outer = (1, 3, 4))
        x = rand(2, 3, 4)
        y1 = bop * x
        y2 = similar(x)
        for i in 1:3, j in 1:4
            mul!(@view(y2[:, i, j]), ops[i], @view(x[:, i, j]))
        end
        @test y1 == y2
        x_bad_type = rand(Int, 2, 3, 4)
        y = zeros(2, 3, 4)
        @test_throws ArgumentError mul!(y, bop, x_bad_type)
        x_bad_size = rand(2, 3, 5)
        @test_throws DimensionMismatch mul!(y, bop, x_bad_size)
        y_bad_type = rand(Int, 2, 3, 4)
        @test_throws ArgumentError mul!(y_bad_type, bop, x)
        y_bad_size = zeros(3, 3, 4)
        @test_throws DimensionMismatch mul!(y_bad_size, bop, x)
        bad_ops = [DiagOp([1.0, 2.0]), DiagOp([1.0, 2.0, 3.0]), DiagOp([1.0, 2.0])]
        return @test_throws AssertionError BatchOp(bad_ops, 4)
    end
end

@testitem "SpreadingBatchOp shape-keeping non-threaded" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    SpreadingBatchOpHelpers.test_shape_keeping_threadsafe_spreading_batch_op(false)
end

@testitem "SpreadingBatchOp shape-keeping threaded" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        SpreadingBatchOpHelpers.test_shape_keeping_threadsafe_spreading_batch_op(true)
    end
end

@testitem "SpreadingBatchOp variation non-threaded" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    SpreadingBatchOpHelpers.test_shape_changing_threadsafe_spreading_batch_op(false)
end

@testitem "SpreadingBatchOp variation threaded" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        SpreadingBatchOpHelpers.test_shape_changing_threadsafe_spreading_batch_op(true)
    end
end

@testitem "SpreadingBatchOp non-threadsafe auto" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    SpreadingBatchOpHelpers.test_nonthreadsafe_spreading_batch_op(false, AbstractOperators.ThreadingStrategy.AUTO)
end

@testitem "SpreadingBatchOp threaded copying" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        SpreadingBatchOpHelpers.test_nonthreadsafe_spreading_batch_op(true, AbstractOperators.ThreadingStrategy.COPYING)
    end
end

@testitem "SpreadingBatchOp threaded locking" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        SpreadingBatchOpHelpers.test_nonthreadsafe_spreading_batch_op(true, AbstractOperators.ThreadingStrategy.LOCKING)
    end
end

@testitem "SpreadingBatchOp threaded fixed operator" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        SpreadingBatchOpHelpers.test_nonthreadsafe_spreading_batch_op(true, AbstractOperators.ThreadingStrategy.FIXED_OPERATOR)
        SpreadingBatchOpHelpers.test_failing_nonthreadsafe_spreading_batch_op()
    end
end

@testitem "SpreadingBatchOp other tests non-threaded" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    SpreadingBatchOpHelpers.other_spreadingbatchop_tests(false)
end

@testitem "SpreadingBatchOp other tests threaded" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        SpreadingBatchOpHelpers.other_spreadingbatchop_tests(true)
    end
end

@testitem "SpreadingBatchOp benchmark" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() >= 4 && get(ENV, "CI", "false") == "false"
        t_single_threaded = SpreadingBatchOpHelpers.benchmark_threading_strategy(false, AbstractOperators.ThreadingStrategy.AUTO)
        t_copying = SpreadingBatchOpHelpers.benchmark_threading_strategy(true, AbstractOperators.ThreadingStrategy.COPYING)
        t_fixed_operator = SpreadingBatchOpHelpers.benchmark_threading_strategy(true, AbstractOperators.ThreadingStrategy.FIXED_OPERATOR)
        @test t_copying < t_single_threaded
        @test t_fixed_operator < t_single_threaded
    end
end

@testitem "SpreadingBatchOpCopying property delegations" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        ops = [DiagOp(rand(5)) * FiniteDiff((6,)) for i in 1:3]
        bop = BatchOp(ops, 4, (:_, :s, :b); threaded = true, threading_strategy = AbstractOperators.ThreadingStrategy.COPYING)
        io = IOBuffer(); show(io, bop); s = String(take!(io)); @test occursin("⟳", s)
        @test domain_array_type(bop) == domain_array_type(ops[1])
        @test codomain_array_type(bop) == codomain_array_type(ops[1])
        @test is_linear(bop) == is_linear(ops[1])
        @test is_eye(bop) == is_eye(ops[1])
        @test is_AAc_diagonal(bop) == is_AAc_diagonal(ops[1])
        @test is_AcA_diagonal(bop) == is_AcA_diagonal(ops[1])
        @test is_full_row_rank(bop) == is_full_row_rank(ops[1])
        @test is_full_column_rank(bop) == is_full_column_rank(ops[1])
        @test is_sliced(bop) == is_sliced(ops[1])
        @test is_null(bop) == is_null(ops[1])
        @test is_diagonal(bop) == is_diagonal(ops[1])
        @test is_invertible(bop) == is_invertible(ops[1])
        @test is_orthogonal(bop) == is_orthogonal(ops[1])
        @test is_thread_safe(bop) == is_thread_safe(ops[1])
        @test AbstractOperators.has_optimized_normalop(bop) == AbstractOperators.has_optimized_normalop(ops[1])
        @test AbstractOperators.has_fast_opnorm(bop) == AbstractOperators.has_fast_opnorm(ops[1])
        operator_norm = opnorm(bop)
        @test operator_norm ≈ maximum(opnorm.(ops)) rtol = 5.0e-6
        @test estimate_opnorm(bop) ≈ operator_norm rtol = 0.05
        ops2 = [DiagOp(rand(5)) for i in 1:3]
        bop2 = BatchOp(ops2, 4, (:_, :s, :b); threaded = true, threading_strategy = AbstractOperators.ThreadingStrategy.COPYING)
        @test size(diag(bop2)) == (5, 3, 4)
        @test size(diag_AcA(bop2)) == (5, 3, 4)
        @test size(diag_AAc(bop2)) == (5, 3, 4)
    end
end

@testitem "Locking get_normal_op and reused operators" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        op = DiagOp(rand(6)) * FiniteDiff((7,))
        ops = [op, op, DiagOp(rand(6)) * FiniteDiff((7,))]
        bop = BatchOp(ops, 4, (:_, :s, :b); threaded = true, threading_strategy = AbstractOperators.ThreadingStrategy.LOCKING)
        y = bop * rand(7, 3, 4)
        @test size(y) == (6, 3, 4)
    end
end

@testitem "FixedOperator get_normal_op and get_spreading_dims" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        ops = [DiagOp(rand(6)) * FiniteDiff((7,)) for i in 1:3]
        bop = BatchOp(ops, 4, (:_, :s, :b); threaded = true, threading_strategy = AbstractOperators.ThreadingStrategy.FIXED_OPERATOR)
        y = bop * rand(7, 3, 4)
        @test size(y) == (6, 3, 4)
    end
end

@testitem "Orthogonal property for SpreadingBatchOp" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    ops = [Eye(Float64, 5) for i in 1:3]
    @test is_orthogonal(BatchOp(ops, 4; threaded = false)) == true
    if Threads.nthreads() > 1
        bop_threaded = BatchOp(ops, 4; threaded = true)
        @test is_orthogonal(bop_threaded) == true
        bop2 = BatchOp([Eye(Float64, 5) for i in 1:3], 4; threaded = true)
        @test is_orthogonal(bop2) == is_orthogonal(ops[1])
    end
end

@testitem "AUTO threading strategy triggering" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        ops = [FiniteDiff((11,)) for i in 1:3]
        bop = BatchOp(ops, 4, (:_, :s, :b); threaded = true, threading_strategy = AbstractOperators.ThreadingStrategy.AUTO)
        @test size(bop * rand(11, 3, 4)) == (10, 3, 4)
    end
end

@testitem "Scalar diagonal return paths" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    scale_val = 2.0
    ops = [scale_val * Eye(Float64, 5) for i in 1:3]
    bop = BatchOp(ops, 4; threaded = false)
    @test diag(bop) isa Number
    @test diag(bop) == scale_val
    @test diag_AcA(bop) isa Number
    @test diag_AcA(bop) == scale_val^2
    @test diag_AAc(bop) isa Number
    @test diag_AAc(bop) == scale_val^2
end

@testitem "SpreadingBatchOp (GPU)" tags = [:gpu, :batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)
        ops = [DiagOp(to_gpu(backend, [1.0, 2.0])) for _ in 1:4]
        bop = BatchOp(ops, 3, (:_, :s, :b))
        y_gpu = bop * gpu_ones(backend, Float64, 2, 4, 3)
        @test size(Array(y_gpu)) == (2, 4, 3)
        @test all(Array(y_gpu)[1, :, :] .≈ 1.0)
        @test all(Array(y_gpu)[2, :, :] .≈ 2.0)
    end
end

@testitem "BatchOp without explicit sizes (lines 107, 112)" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    # BatchOp(operators) with no size args → calls BatchOp(operators, ()) → line 112
    n = 5
    ops = [Eye(Float64, (n,)) for _ in 1:3]
    bop = BatchOp(ops; threaded = false)
    @test bop isa AbstractOperators.SpreadingBatchOp
    x = randn(n, 3)
    y = bop * x
    @test size(y) == (n, 3)
    @test y ≈ x
end

@testitem "BatchOp unsupported threading strategy for non-thread-safe ops (line 404)" tags = [:batching, :SpreadingBatchOp] setup = [TestUtils, SpreadingBatchOpHelpers] begin
    using Random, AbstractOperators
    Random.seed!(0)
    if Threads.nthreads() > 1
        n = 5
        # LBFGS is not thread-safe; an unknown strategy reaches the else branch at line 404
        ops = [LBFGS(zeros(n), 3) for _ in 1:3]
        @test_throws ArgumentError BatchOp(
            ops, (4,), (:b, :s, :_);
            threaded = true,
            threading_strategy = :UNKNOWN_STRATEGY,
        )
    end
end
