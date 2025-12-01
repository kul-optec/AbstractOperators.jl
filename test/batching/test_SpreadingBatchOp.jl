if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)
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
	# opnorm / estimate_opnorm aggregate as maximum -- exact solution is expected for both
	@test opnorm(bop) == maximum(opnorm.(ops))
	@test estimate_opnorm(bop) == maximum(estimate_opnorm.(ops))
	@test estimate_opnorm(bop) == opnorm(bop)
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
	if Threads.nthreads() > 1 && get(ENV, "CI", "false") == "false"
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

	@testset "SpreadingBatchOpCopying property delegations" begin
		if Threads.nthreads() > 1
			# Create non-threadsafe operators to trigger COPYING strategy
			ops = [DiagOp(rand(5)) * FiniteDiff((6,)) for i in 1:3]
			bop = BatchOp(ops, 4, (:_, :s, :b); threaded=true, threading_strategy=ThreadingStrategy.COPYING)
			
			# fun_name for Copying variant
			io = IOBuffer(); show(io, bop); s = String(take!(io))
			@test occursin("⟳", s)
			
			# Storage types for Copying
			@test domain_storage_type(bop) == domain_storage_type(ops[1])
			@test codomain_storage_type(bop) == codomain_storage_type(ops[1])
			
			# Properties for Copying
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
			
			# Normal op for Copying - requires optimized normal op
			@test AbstractOperators.has_optimized_normalop(bop) == AbstractOperators.has_optimized_normalop(ops[1])
			
			# opnorm methods for Copying (use approximate equality for floating point)
			@test AbstractOperators.has_fast_opnorm(bop) == AbstractOperators.has_fast_opnorm(ops[1])
			operator_norm = opnorm(bop)
			@test operator_norm ≈ maximum(opnorm.(ops)) rtol=5e-6
			@test estimate_opnorm(bop) ≈ operator_norm rtol=0.05
			
			# diag methods for Copying - use operators with diag
			ops2 = [DiagOp(rand(5)) for i in 1:3]
			bop2 = BatchOp(ops2, 4, (:_, :s, :b); threaded=true, threading_strategy=ThreadingStrategy.COPYING)
			d = diag(bop2)
			@test size(d) == (5, 3, 4)
			daca = diag_AcA(bop2)
			@test size(daca) == (5, 3, 4)
			daac = diag_AAc(bop2)
			@test size(daac) == (5, 3, 4)
		end
	end

	@testset "Locking get_normal_op and reused operators" begin
		if Threads.nthreads() > 1
			# Create scenario with reused operator to trigger haskey branch
			op = DiagOp(rand(6)) * FiniteDiff((7,))
			ops = [op, op, DiagOp(rand(6)) * FiniteDiff((7,))]  # First two are same instance
			bop = BatchOp(ops, 4, (:_, :s, :b); threaded=true, threading_strategy=ThreadingStrategy.LOCKING)
			
			# Verify it works
			x = rand(7, 3, 4)
			y = bop * x
			@test size(y) == (6, 3, 4)
		end
	end

	@testset "FixedOperator get_normal_op and get_spreading_dims" begin
		if Threads.nthreads() > 1
			ops = [DiagOp(rand(6)) * FiniteDiff((7,)) for i in 1:3]
			bop = BatchOp(ops, 4, (:_, :s, :b); threaded=true, threading_strategy=ThreadingStrategy.FIXED_OPERATOR)
			
			# Verify get_spreading_dims is called (indirectly via operations)
			x = rand(7, 3, 4)
			y = bop * x
			@test size(y) == (6, 3, 4)
		end
	end

	@testset "Orthogonal property for SpreadingBatchOp" begin
		# Use Identity operators which are orthogonal
		ops = [Eye(Float64, 5) for i in 1:3]
		bop = BatchOp(ops, 4; threaded=false)
		@test is_orthogonal(bop) == true
		
		if Threads.nthreads() > 1
			# Test with threaded version
			bop_threaded = BatchOp(ops, 4; threaded=true)
			@test is_orthogonal(bop_threaded) == true
			
			# Test with Copying variant - need non-threadsafe orthogonal operator
			# Use Compose with Eye operators (still orthogonal but might not be thread-safe depending on implementation)
			ops2 = [Eye(Float64, 5) for i in 1:3]
			bop2 = BatchOp(ops2, 4; threaded=true)
			@test is_orthogonal(bop2) == is_orthogonal(ops2[1])
		end
	end

	@testset "AUTO threading strategy triggering" begin
		if Threads.nthreads() > 1
			# Create scenario where AUTO should pick a strategy
			# Small operators with FiniteDiff (non-threadsafe)
			ops = [FiniteDiff((11,)) for i in 1:3]
			bop = BatchOp(ops, 4, (:_, :s, :b); threaded=true, threading_strategy=ThreadingStrategy.AUTO)
			
			# Should work regardless of chosen strategy
			x = rand(11, 3, 4)
			y = bop * x
			@test size(y) == (10, 3, 4)
		end
	end

	@testset "Scalar diagonal return paths" begin
		# Create operators where all have identical scalar diagonals
		scale_val = 2.0
		ops = [scale_val * Eye(Float64, 5) for i in 1:3]
		bop = BatchOp(ops, 4; threaded=false)
		
		# These should return scalars when all operators have identical scalar diagonals
		d = diag(bop)
		@test d isa Number  # Expect scalar
		@test d == scale_val
		
		daca = diag_AcA(bop)
		@test daca isa Number  # Expect scalar
		@test daca == scale_val^2
		
		daac = diag_AAc(bop)
		@test daac isa Number  # Expect scalar
		@test daac == scale_val^2
	end
end
