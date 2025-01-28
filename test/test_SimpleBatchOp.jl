function test_simple_batchop(op, batch_op, x, y, z, threaded)
	if threaded
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
	@test z == z2
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
	op = Variation(3, 4, 5)
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

function test_filt_simple_batch_op(threaded)
	b, a = [1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 1.0; 1.0]
	op = Filt(Float64, (15,), b, a)
	batch_op = BatchOp(op, (10, 20), (:b, :b, :_); threaded)
	x = rand(10, 20, 15)
	y = zeros(10, 20, 15)
	z = similar(x)
	for i in 1:10, j in 1:20
		mul!(@view(y[i, j, :]), op, @view(x[i, j, :]))
	end
	for i in 1:10, j in 1:20
		mul!(@view(z[i, j, :]), op', @view(y[i, j, :]))
	end
	return test_simple_batchop(op, batch_op, x, y, z, threaded)
end

function benchmark_threading(threaded)
	b, a = [1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 1.0; 1.0]
	op = Filt(Float64, (100,), b, a)
	batch_op = BatchOp(op, (30, 40), (:_, :b, :b); threaded)
	y = zeros(100, 30, 40)
	return @belapsed(mul!($y, $batch_op, x), setup = ($y .= 0; x = rand(100, 30, 40)))
end

@testset "SimpleBatchOp" begin
	@testset "Shape-keeping op (DiagOp)" begin
		@testset "non-threaded" begin
			test_shape_keeping_simple_batch_op(false)
		end
		@testset "threaded (thread-safe)" begin
			test_shape_keeping_simple_batch_op(true)
		end
	end
	@testset "Dimension count changing op (Variation)" begin
		@testset "non-threaded" begin
			test_variation_simple_batch_op(false)
		end
		@testset "threaded (thread-safe)" begin
			test_variation_simple_batch_op(true)
		end
	end
	@testset "Shape changing op (Filt)" begin
		@testset "non-threaded" begin
			test_filt_simple_batch_op(false)
		end
		@testset "threaded (copying)" begin
			test_filt_simple_batch_op(true)
		end
	end
	if nthreads() > 1
		@testset "Benchmark" begin
			t_single_threaded = benchmark_threading(false)
			t_multi_threaded = benchmark_threading(true)
			@test t_multi_threaded < t_single_threaded
		end
	end
end
