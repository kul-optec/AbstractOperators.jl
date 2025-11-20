if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
using BenchmarkTools

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
	op = Variation(3, 4, 5; threaded=false)
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
	n = 1000
	op = Compose(DiagOp(randn(n - 1)), FiniteDiff((n,), 1))
	batch_op = BatchOp(op, (30, 40), (:_, :b, :b); threaded)
	y = zeros(n - 1, 30, 40)
	return @belapsed(mul!($y, $batch_op, x), setup = ($y .= 0; x = rand($n, 30, 40)))
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
	# show (fun_name)
	io = IOBuffer(); show(io, batch_op); s = String(take!(io))
	@test occursin("⟳", s)
	# == and isequal
	batch_op_copy = AbstractOperators.copy_op(batch_op)
	@test batch_op == batch_op_copy
	@test isequal(batch_op, batch_op_copy)
	# storage types
	@test domain_storage_type(batch_op) == domain_storage_type(op)
	@test codomain_storage_type(batch_op) == codomain_storage_type(op)
	# property queries
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
	# has_optimized_normalop, get_normal_op
	@test AbstractOperators.has_optimized_normalop(batch_op) == AbstractOperators.has_optimized_normalop(op)
	n_op = AbstractOperators.get_normal_op(batch_op)
	@test typeof(n_op) <: typeof(batch_op)
	# opnorm, estimate_opnorm
	@test opnorm(batch_op) ≈ opnorm(op)
	@test estimate_opnorm(batch_op) == estimate_opnorm(op)
	# diag, diag_AcA, diag_AAc
	@test diag(batch_op) == [diag(op)'; diag(op)']'
	@test diag_AcA(batch_op) == [diag_AcA(op)'; diag_AcA(op)']'
	@test diag_AAc(batch_op) == [diag_AAc(op)'; diag_AAc(op)']'
	# error path: mismatched input type
	x_bad = rand(Int, 2, 2)
	y_bad = zeros(2, 2)
	@test_throws ArgumentError mul!(y_bad, batch_op, x_bad)
	# error path: mismatched input size
	x_bad2 = rand(2, 3)
	@test_throws ArgumentError mul!(y_bad, batch_op, x_bad2)
	# error path: mismatched output type
	y_bad2 = rand(Int, 2, 2)
	x_good = rand(2, 2)
	@test_throws ArgumentError mul!(y_bad2, batch_op, x_good)
	# error path: mismatched output size
	y_bad3 = zeros(3, 2)
	@test_throws ArgumentError mul!(y_bad3, batch_op, x_good)
end

@testset "SimpleBatchOp" begin
	@testset "Shape-keeping op (DiagOp)" begin
		@testset "non-threaded" begin
			test_shape_keeping_simple_batch_op(false)
		end
		if Threads.nthreads() > 1
			@testset "threaded (thread-safe)" begin
				test_shape_keeping_simple_batch_op(true)
			end
		end
	end
	@testset "Dimension count changing op (Variation)" begin
		@testset "non-threaded" begin
			test_variation_simple_batch_op(false)
		end
		if Threads.nthreads() > 1
			@testset "threaded (thread-safe)" begin
				test_variation_simple_batch_op(true)
			end
		end
	end
	@testset "Shape-changing op (DiagOp∘FiniteDiff)" begin
		@testset "non-threaded" begin
			test_shape_changing_simple_batch_op(false)
		end
		if Threads.nthreads() > 1
			@testset "threaded (thread-safe)" begin
				test_shape_changing_simple_batch_op(true)
			end
		end
	end
	@testset "Other tests" begin
		@testset "non-threaded" begin
			other_tests(false)
		end
		if Threads.nthreads() > 1
			@testset "threaded (thread-safe)" begin
				other_tests(true)
			end
		end
	end
	if Threads.nthreads() > 1 && get(ENV, "CI", "false") == "false"
		@testset "Benchmark" begin
			t_single_threaded = benchmark_threading(false)
			t_multi_threaded = benchmark_threading(true)
			@test t_multi_threaded < t_single_threaded
		end
	end
end
