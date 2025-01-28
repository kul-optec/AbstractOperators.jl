function test_spreading_batchop(operators, batch_op, x, y, z, threaded)
	if !threaded
		@test batch_op.operators == operators
	end
	@test size(batch_op, 1) == size(y)
	@test size(batch_op, 2) == size(x)
	y2 = batch_op * x
	@test y == y2
	z2 = batch_op' * y
	@test z == z2
end

function test_shape_keeping_spreading_batch_op(threaded)
	ops = [i * DiagOp([1.0im, 2.0im]) for i in 1:3]
	batch_op = BatchOp(ops, (3, 4), (:_, :s, :b); threaded)
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

function test_variation_spreading_batch_op(threaded)
	ops = [i * Variation(3, 4, 5) for i in 1:2]
	batch_op = BatchOp(ops, (2, 6), (:s, :_, :_, :_, :b) => (:s, :_, :b, :_); threaded)
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

function test_filt_spreading_batch_op(threaded, threading_strategy)
	b, a = [1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 1.0; 1.0]
	num_ops = nthreads() + 5
	ops = [i * Filt(Float64, (15,), b, a) for i in 1:num_ops]
	batch_op = BatchOp(ops, (10, num_ops), (:b, :s, :_); threaded, threading_strategy)
	x = rand(10, num_ops, 15)
	y = zeros(10, num_ops, 15)
	z = similar(x)
	for i in 1:10, j in 1:num_ops
		mul!(@view(y[i, j, :]), ops[j], @view(x[i, j, :]))
	end
	for i in 1:10, j in 1:num_ops
		mul!(@view(z[i, j, :]), ops[j]', @view(y[i, j, :]))
	end
	return test_spreading_batchop(ops, batch_op, x, y, z, threaded)
end

function benchmark_threading_strategy(threaded, threading_strategy)
	b, a = [1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 1.0; 1.0]
	num_ops = nthreads() + 50
	ops = [i * Filt(Float64, (100,), b, a) for i in 1:num_ops]
	batch_op = BatchOp(ops, (num_ops, 40), (:_, :s, :b); threaded, threading_strategy)
	y = zeros(100, num_ops, 40)
	return @belapsed(mul!($y, $batch_op, x), setup = ($y .= 0; x = rand(100, $num_ops, 40)))
end

@testset "SpreadingBatchOp" begin
	@testset "Shape-keeping op (DiagOp)" begin
		@testset "non-threaded" begin
			test_shape_keeping_spreading_batch_op(false)
		end
		@testset "threaded (thread-safe)" begin
			test_shape_keeping_spreading_batch_op(true)
		end
	end
	@testset "Shape-changing op (Variation)" begin
		@testset "non-threaded" begin
			test_variation_spreading_batch_op(false)
		end
		@testset "threaded (thread-safe)" begin
			test_variation_spreading_batch_op(true)
		end
	end
	@testset "Non-threadsafe op (Filt)" begin
		@testset "non-threaded" begin
			test_filt_spreading_batch_op(false, ThreadingStrategy.AUTO)
		end
		@testset "threaded (copying)" begin
			test_filt_spreading_batch_op(true, ThreadingStrategy.COPYING)
		end
		@testset "threaded (locking)" begin
			test_filt_spreading_batch_op(true, ThreadingStrategy.LOCKING)
		end
		@testset "threaded (fixed operator)" begin
			test_filt_spreading_batch_op(true, ThreadingStrategy.FIXED_OPERATOR)
		end
	end
	if nthreads() > 1
		@testset "Benchmark" begin
			t_single_threaded = benchmark_threading_strategy(false, ThreadingStrategy.AUTO)
			t_copying = benchmark_threading_strategy(true, ThreadingStrategy.COPYING)
			t_fixed_operator = benchmark_threading_strategy(
				true, ThreadingStrategy.FIXED_OPERATOR
			)
			@test t_copying < t_single_threaded
			@test t_fixed_operator < t_single_threaded
		end
	end
end
