if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
using BenchmarkTools

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
	ops = [DiagOp(rand(m-1)) * FiniteDiff((m,)) for i in 1:num_ops]
	batch_op = BatchOp(ops, n, (:b, :s, :_); threaded, threading_strategy)
	x = rand(n, num_ops, m)
	y = zeros(n, num_ops, m-1)
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
	op = GetIndex(Float64, (m-1,), 1:6) * FiniteDiff((m,))
	ops = [reshape(i * op, 2, 3) for i in 1:num_ops]
	@test_throws ArgumentError BatchOp(ops, n, (:b, :s, :_) => (:b, :s, :_, :_); threaded=true, threading_strategy=ThreadingStrategy.FIXED_OPERATOR)
end

function benchmark_threading_strategy(threaded, threading_strategy)
	n, m = 40, 100
	num_ops = Threads.nthreads() + 50
	ops = [DiagOp(rand(m-1)) * FiniteDiff((m,)) for i in 1:num_ops]
	batch_op = BatchOp(ops, n, (:_, :s, :b); threaded, threading_strategy)
	y = zeros(m-1, num_ops, n)
	return @belapsed(mul!($y, $batch_op, x), setup = ($y .= 0; x = rand($m, $num_ops, $n)))
end

function other_spreadingbatchop_tests(threaded)
	ops = [DiagOp([1.0, 2.0]) for _ in 1:3]
	bop = BatchOp(ops, 4; threaded=threaded)
	# show output (fun_name indirectly)
	io = IOBuffer(); show(io, bop); s = String(take!(io))
	@test occursin("⟳", s)
	# size consistency
	cod, dom = size(bop)
	@test cod == size(bop, 1) && dom == size(bop, 2)
	# storage types
	@test domain_storage_type(bop) == domain_storage_type(ops[1])
	@test codomain_storage_type(bop) == codomain_storage_type(ops[1])
	# property queries mirror first operator
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
	# normal op
	@test AbstractOperators.has_optimized_normalop(bop) == AbstractOperators.has_optimized_normalop(ops[1])
	nbop = AbstractOperators.get_normal_op(bop)
	@test size(nbop, 1) == size(bop, 1) && size(nbop, 2) == size(bop, 2)
	# opnorm / estimate_opnorm aggregate as maximum
	@test opnorm(bop) == maximum(opnorm.(ops))
	@test estimate_opnorm(bop) == maximum(estimate_opnorm.(ops))
	# diag family collapses to first operator entries (identical)
	@test diag(bop) == repeat(diag(ops[1]), outer=(1,3,4))
	@test diag_AcA(bop) == repeat(diag_AcA(ops[1]), outer=(1,3,4))
	@test diag_AAc(bop) == repeat(diag_AAc(ops[1]), outer=(1,3,4))
	# Functional sanity: apply and compare with manual broadcast
	x = rand(2, 3, 4)
	y1 = bop * x
	y2 = similar(x)
	for i in 1:3, j in 1:4
		mul!(@view(y2[:, i, j]), ops[i], @view(x[:, i, j]))
	end
	@test y1 == y2
	# Error paths (type / size)
	x_bad_type = rand(Int, 2, 3, 4)
	y = zeros(2, 3, 4)
	@test_throws ArgumentError mul!(y, bop, x_bad_type)
	x_bad_size = rand(2, 3, 5) # wrong last dim
	@test_throws ArgumentError mul!(y, bop, x_bad_size)
	y_bad_type = rand(Int, 2, 3, 4)
	@test_throws ArgumentError mul!(y_bad_type, bop, x)
	y_bad_size = zeros(3, 3, 4)
	@test_throws ArgumentError mul!(y_bad_size, bop, x)
	# Constructor assertion error: mismatched operator shapes (different size)
	bad_ops = [DiagOp([1.0, 2.0]), DiagOp([1.0, 2.0, 3.0]), DiagOp([1.0, 2.0])]
	@test_throws AssertionError BatchOp(bad_ops, 4)
end

@testset "SpreadingBatchOp" begin
	@testset "Shape-keeping op (DiagOp)" begin
		@testset "non-threaded" begin
			test_shape_keeping_threadsafe_spreading_batch_op(false)
		end
		if Threads.nthreads() > 1
			@testset "threaded (thread-safe)" begin
				test_shape_keeping_threadsafe_spreading_batch_op(true)
			end
		end
	end
	@testset "Shape-changing op (Variation)" begin
		@testset "non-threaded" begin
			test_shape_changing_threadsafe_spreading_batch_op(false)
		end
		if Threads.nthreads() > 1
			@testset "threaded (thread-safe)" begin
				test_shape_changing_threadsafe_spreading_batch_op(true)
			end
		end
	end
	@testset "Non-threadsafe op (Compose)" begin
		@testset "non-threaded" begin
			test_nonthreadsafe_spreading_batch_op(false, ThreadingStrategy.AUTO)
		end
		if Threads.nthreads() > 1
			@testset "threaded (copying)" begin
				test_nonthreadsafe_spreading_batch_op(true, ThreadingStrategy.COPYING)
			end
			@testset "threaded (locking)" begin
				test_nonthreadsafe_spreading_batch_op(true, ThreadingStrategy.LOCKING)
			end
			@testset "threaded (fixed operator)" begin
				test_nonthreadsafe_spreading_batch_op(true, ThreadingStrategy.FIXED_OPERATOR)
				test_failing_nonthreadsafe_spreading_batch_op()
			end
		end
	end
	@testset "Other tests" begin
		@testset "non-threaded" begin
			other_spreadingbatchop_tests(false)
		end
		if Threads.nthreads() > 1
			@testset "threaded (thread-safe)" begin
				other_spreadingbatchop_tests(true)
			end
		end
	end
	if Threads.nthreads() > 1
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
