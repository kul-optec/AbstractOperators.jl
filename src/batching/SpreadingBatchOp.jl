export ThreadingStrategy

abstract type SpreadingBatchOp{dT,cT,dM,cM} <: BatchOp{dT,cT,dM,cM} end

"""
Threading strategy for `SpreadingBatchOp` operators.
- `AUTO`: Automatically choose the best threading strategy based on the size of the operators and the input arrays.
- `COPYING`: Create a copy of the operators for each thread.
- `LOCKING`: Use locks to ensure thread safety.
- `FIXED_OPERATOR`: Use a fixed set of operators for each thread, i.e. each operator is assigned to a specific thread.
"""
module ThreadingStrategy
const AUTO = :auto
const COPYING = :copying
const LOCKING = :locking
const FIXED_OPERATOR = :fixed_operator
end

struct SpreadingBatchOpSingleThreaded{dT,cT,dM,cM,sD,N,M,B,opT} <:
	   SpreadingBatchOp{dT,cT,dM,cM}
	operators::Array{opT}
	domain_size::NTuple{N,Int}
	codomain_size::NTuple{M,Int}
	batch_size::NTuple{B,Int}
end

struct SpreadingBatchOpThreadSafe{dT,cT,dM,cM,sD,N,M,opT} <: SpreadingBatchOp{dT,cT,dM,cM}
	operators::Array{opT}
	domain_size::NTuple{N,Int}
	codomain_size::NTuple{M,Int}
	batch_indices::CartesianIndices
end

struct SpreadingBatchOpCopying{dT,cT,dM,cM,sD,N,M,opT} <: SpreadingBatchOp{dT,cT,dM,cM}
	operators::Vector{Array{opT}}
	domain_size::NTuple{N,Int}
	codomain_size::NTuple{M,Int}
	batch_indices::CartesianIndices
end

struct SpreadingBatchOpLocking{dT,cT,dM,cM,sD,N,M,opT} <: SpreadingBatchOp{dT,cT,dM,cM}
	operators::Array{opT}
	domain_size::NTuple{N,Int}
	codomain_size::NTuple{M,Int}
	batch_indices::CartesianIndices
	locks::Array{AbstractLock}
end

struct SpreadingBatchOpFixedOperator{dT,cT,dM,cM,sD,N,M,opT} <:
	   SpreadingBatchOp{dT,cT,dM,cM}
	operators::Array{opT}
	domain_size::NTuple{N,Int}
	codomain_size::NTuple{M,Int}
	batch_indices::CartesianIndices
	operator_indices::CartesianIndices
end

# Public constructors

"""
	function BatchOp(
		operators::Array{<:AbstractOperators.AbstractOperator},
		batch_size::NTuple{N,Int},
		batch_dim_mask::Union{NTuple{M,Symbol}, Pair{NTuple{M1,Symbol},NTuple{M2,Symbol}};
		threaded::Bool=nthreads() > 1,
		threading_strategy::Symbol=ThreadingStrategy.AUTO,
	)

Creates a "spreading" `BatchOp` from an array of `AbstractOperator`s. The `BatchOp` can be used to apply the operators to an array over
the selected batch dimensions. The "spreading" dimensions will be used to select the operator to apply, and the batch dimensions will be used to
repeat the operator application over the batch dimensions.

# Arguments
- `operators::Array{<:AbstractOperators.AbstractOperator}`: The operators to be batched.
- `batch_size::NTuple{N,Int}`: The size of the batch dimensions.
- `batch_dim_mask::Union{NTuple{M,Symbol}, Pair{NTuple{M1,Symbol},NTuple{M2,Symbol}}}`: The mask specifying which dimensions are batch
dimensions. The symbols can be `:b` for batch dimensions, `:s` for spreading dimensions or `:_` for dimensions on which the operator acts.
If the mask is a pair, the first element specifies the domain batch dimensions and the second element specifies the codomain batch dimensions.
If no mask is provided, the following order is assumed: (operator dims..., spreading dims..., batch dims...)
- `threaded::Bool`: If `true`, the operator will execute in parallel over the batch dimensions. Default is `nthreads() > 1`.
- `threading_strategy::Symbol`: The threading strategy to use. Default is `ThreadingStrategy.AUTO`, which will automatically choose the
best strategy based on the size of the operators and the input arrays.

# Examples
```jldoctest
julia> ops = [i * DiagOp([1.0im, 2.0im]) for i in 1:3];

julia> batch_op = BatchOp(ops, 4) # operator dims: (2,), spreading dims: (3,), batch dims: (4,)
⟳╲  ℂ^(2, 3, 4) -> ℂ^(2, 3, 4)

julia> x = rand(ComplexF64, 2, 3, 4);

julia> y = similar(x);

julia> [mul!(@view(y[:, i, j]), ops[i], @view(x[:, i, j])) for i in 1:3, j in 1:4];

julia> y == batch_op * x
true

julia> ops = [i * Variation(3, 4, 5) for i in 1:2];

julia> batch_op = BatchOp(ops, 6, (:s, :_, :_, :_, :b) => (:s, :_, :b, :_))
⟳Ʋ  ℝ^(2, 3, 4, 5, 6) -> ℝ^(2, 60, 6, 3)

```
"""
function BatchOp(
	operators::Array{<:AbstractOperators.AbstractOperator};
	threaded::Bool=nthreads() > 1,
	threading_strategy::Symbol=ThreadingStrategy.AUTO,
)
	return BatchOp(operators, (); threaded, threading_strategy)
end

function BatchOp(
	operators::Array{<:AbstractOperators.AbstractOperator},
	batch_size;
	threaded::Bool=nthreads() > 1,
	threading_strategy::Symbol=ThreadingStrategy.AUTO,
)
	op_domain_dims = ndims(operators[1], 2)
	op_codomain_dims = ndims(operators[1], 1)
	spreading_dims = ndims(operators)
	batch_domain_dim_mask = get_batch_dim_mask(
		op_domain_dims, spreading_dims, length(batch_size)
	)
	batch_codomain_dim_mask = get_batch_dim_mask(
		op_codomain_dims, spreading_dims, length(batch_size)
	)
	return BatchOp(
		operators,
		batch_size,
		batch_domain_dim_mask => batch_codomain_dim_mask;
		threaded,
		threading_strategy,
	)
end

function BatchOp(
	operators::Array{<:AbstractOperators.AbstractOperator},
	batch_dim_mask::NTuple{M,Symbol};
	threaded::Bool=nthreads() > 1,
	threading_strategy::Symbol=ThreadingStrategy.AUTO,
) where {M}
	return BatchOp(
		operators,
		(),
		batch_dim_mask;
		threaded,
		threading_strategy,
	)
end

function BatchOp(
	operators::Array{<:AbstractOperators.AbstractOperator},
	batch_size,
	batch_dim_mask::NTuple{M,Symbol};
	threaded::Bool=nthreads() > 1,
	threading_strategy::Symbol=ThreadingStrategy.AUTO,
) where {M}
	return BatchOp(
		operators,
		batch_size,
		batch_dim_mask => batch_dim_mask;
		threaded,
		threading_strategy,
	)
end

function BatchOp(
	operators::Array{<:AbstractOperators.AbstractOperator},
	batch_dim_mask::Pair{NTuple{M1,Symbol},NTuple{M2,Symbol}};
	threaded::Bool=nthreads() > 1,
	threading_strategy::Symbol=ThreadingStrategy.AUTO,
) where {M1,M2}
	return BatchOp(
		operators,
		(),
		batch_dim_mask;
		threaded,
		threading_strategy,
	)
end

function BatchOp(
	operators::Array{<:AbstractOperators.AbstractOperator},
	batch_size,
	batch_dim_mask::Pair{NTuple{M1,Symbol},NTuple{M2,Symbol}};
	threaded::Bool=nthreads() > 1,
	threading_strategy::Symbol=ThreadingStrategy.AUTO,
) where {M1,M2}
	batch_dims = [m for m in batch_dim_mask.first if m == :s || m == :b]
	spreading_dims = Tuple(i for i in eachindex(batch_dims) if batch_dims[i] == :s)
	spreading_and_batch_dims = ()
	batch_counter = 1
	spreading_counter = 1
	for i in eachindex(batch_dims)
		if batch_dims[i] == :b
			spreading_and_batch_dims = (
				spreading_and_batch_dims..., batch_size[batch_counter]
			)
			batch_counter += 1
		else
			spreading_and_batch_dims = (
				spreading_and_batch_dims..., size(operators, spreading_counter)
			)
			spreading_counter += 1
		end
	end
	domain_batch_dim_mask = symbol_mask_to_bool(batch_dim_mask.first)
	codomain_batch_dim_mask = symbol_mask_to_bool(batch_dim_mask.second)
	domain_size, domain_batch_dim_mask, codomain_size, codomain_batch_dim_mask = calculate_shapes(
		operators[1], spreading_and_batch_dims, domain_batch_dim_mask => codomain_batch_dim_mask
	)
	return create_BatchOp(
		operators,
		domain_size,
		domain_batch_dim_mask,
		codomain_size,
		codomain_batch_dim_mask,
		spreading_dims;
		threaded,
		threading_strategy,
	)
end

function create_BatchOp(
	operators::Array{opT},
	domain_size::NTuple{N,Int},
	domain_batch_dim_mask::NTuple{N,Bool},
	codomain_size::NTuple{M,Int},
	codomain_batch_dim_mask::NTuple{M,Bool},
	spreading_dims::NTuple{K,Int};
	threaded::Bool=nthreads() > 1,
	threading_strategy::Symbol=ThreadingStrategy.AUTO,
) where {M,N,K,opT<:AbstractOperators.AbstractOperator}
	batch_size, dType, cdType, opType = prepare_SpreadingBatchOp(
		operators,
		domain_size,
		domain_batch_dim_mask,
		codomain_size,
		codomain_batch_dim_mask,
		spreading_dims,
	)
	if threaded && nthreads() > 1
		type_args = (
			dType,
			cdType,
			domain_batch_dim_mask,
			codomain_batch_dim_mask,
			spreading_dims,
			N,
			M,
			opType,
		)
		return create_threaded_SpreadingBatchOp(
			operators,
			(domain_size, codomain_size),
			type_args,
			batch_size,
			spreading_dims,
			threading_strategy,
		)
	else
		B = length(batch_size)
		type_args = (
			dType,
			cdType,
			domain_batch_dim_mask,
			codomain_batch_dim_mask,
			spreading_dims,
			N,
			M,
			B,
			opType,
		)
		return SpreadingBatchOpSingleThreaded{type_args...}(
			operators, domain_size, codomain_size, batch_size
		)
	end
end

function prepare_SpreadingBatchOp(
	operators,
	domain_size,
	domain_batch_dim_mask,
	codomain_size,
	codomain_batch_dim_mask,
	spreading_dims,
)
	@assert all(op -> domain_type(op) == domain_type(operators[1]), operators) "All operators must have the same domain type"
	@assert all(op -> codomain_type(op) == codomain_type(operators[1]), operators) "All operators must have the same codomain type"
	@assert all(
		op -> domain_storage_type(op) == domain_storage_type(operators[1]), operators
	) "All operators must have the same storage type"
	@assert all(
		op -> codomain_storage_type(op) == codomain_storage_type(operators[1]), operators
	) "All operators must have the same storage type"
	@assert all(op -> size(op, 2) == size(operators[1], 2), operators) "All operators must have the same input size"
	@assert all(op -> size(op, 1) == size(operators[1], 1), operators) "All operators must have the same output size"
	@assert length(spreading_dims) == ndims(operators) "Number of spreading dimensions must match the number of dimensions in operators array"
	@assert Tuple(unique(spreading_dims)) == spreading_dims "Spreading dimensions must be unique"
	batch_size, dType, cdType = prepare_batch_op(
		operators[1],
		domain_size,
		domain_batch_dim_mask,
		codomain_size,
		codomain_batch_dim_mask,
	)
	opType = promote_type(typeof.(operators)...)
	@assert minimum(spreading_dims) > 0 "Spreading dimensions must be positive"
	@assert maximum(spreading_dims) <= length(batch_size) "Spreading dimensions must be less or equal to the number of batch dimensions"
	@assert size(operators) == Tuple(batch_size[d] for d in spreading_dims) "Shape of operators array must match the shape of spreading dimensions"
	return batch_size, dType, cdType, opType
end

function guess_optimal_threading_strategy(operators, batch_size)
	copied_ops_size = Base.summarysize(operators) * min(nthreads(), prod(batch_size))
	domain_array_size = sizeof(domain_type(operators[1])) * prod(size(operators[1], 2))
	codomain_array_size = sizeof(codomain_type(operators[1])) * prod(size(operators[1], 1))
	if copied_ops_size < 10e6 ||
		copied_ops_size < max(domain_array_size, codomain_array_size) * 2 #= 10MB =#
		threading_strategy = ThreadingStrategy.COPYING
	else
		threading_strategy = ThreadingStrategy.LOCKING
	end
	return threading_strategy
end

function create_threaded_SpreadingBatchOp(
	operators, sizes, type_args, batch_size, spreading_dims, threading_strategy
)
	if all(is_thread_safe, operators)
		return SpreadingBatchOpThreadSafe{type_args...}(
			operators, sizes..., CartesianIndices(batch_size)
		)
	else
		if threading_strategy == ThreadingStrategy.AUTO
			threading_strategy = guess_optimal_threading_strategy(operators, batch_size)
		end
		if threading_strategy == ThreadingStrategy.COPYING
			operators = [
				i == 1 ? operators : [copy_op(op) for op in operators] for
				i in 1:min(nthreads(), prod(batch_size))
			]
			return SpreadingBatchOpCopying{type_args...}(
				operators, sizes..., CartesianIndices(batch_size)
			)
		elseif threading_strategy == ThreadingStrategy.LOCKING
			d = Dict{eltype(operators),Int}()
			locks = Array{AbstractLock}(undef, size(operators))
			for i in eachindex(operators)
				if haskey(d, operators[i])
					locks[i] = locks[d[operators[i]]]
				else
					d[operators[i]] = i
					locks[i] = ReentrantLock()
				end
			end
			return SpreadingBatchOpLocking{type_args...}(
				operators, sizes..., CartesianIndices(batch_size), locks
			)
		elseif threading_strategy == ThreadingStrategy.FIXED_OPERATOR
			# TODO recurse into each operator and check if all non-threadsafe sub-operators are unique
			non_threadsafe_ops = []
			for op in operators
				get_nested_ops(op, non_threadsafe_ops)
			end
			if length(unique(non_threadsafe_ops)) < length(non_threadsafe_ops)
				throw(ArgumentError(
					"A reused (potentially nested) non-thread-safe operator was detected. ThreadingStrategy.FIXED_OPERATOR requires all non-thread-safe operators to be unique.",
				))
			end
			D = length(batch_size)
			batch_indices = CartesianIndices(
				Tuple(d in spreading_dims ? 1 : batch_size[d] for d in 1:D)
			)
			spreading_indices = CartesianIndices(
				Tuple(d in spreading_dims ? batch_size[d] : 1 for d in 1:D)
			)
			operators = reshape(operators, size(spreading_indices))
			return SpreadingBatchOpFixedOperator{type_args...}(
				operators, sizes..., batch_indices, spreading_indices
			)
		else
			throw(
				ArgumentError(
					"Threading strategy $threading_strategy is not supported for non-thread-safe operators",
				),
			)
		end
	end
end

function get_nested_ops(op, collection)
	for fieldname in fieldnames(typeof(op))
		field = getfield(op, fieldname)
		if field isa AbstractOperators.AbstractOperator
			!is_thread_safe(field) && push!(collection, field)
			get_nested_ops(field, collection)
		elseif field isa AbstractArray{<:AbstractOperators.AbstractOperator}
			for sub_op in field
				!is_thread_safe(sub_op) && push!(collection, sub_op)
				get_nested_ops(sub_op, collection)
			end
		elseif field isa Tuple || field isa NTuple
			for sub_field in field
				if sub_field isa AbstractOperators.AbstractOperator
					!is_thread_safe(sub_field) && push!(collection, sub_field)
					get_nested_ops(sub_field, collection)
				end
			end
		end
	end
	return collection
end

@generated function mul!(
	out::AbstractArray,
	op::SpreadingBatchOpSingleThreaded,
	inp::AbstractArray,
)
	left_mask = get_codomain_batch_dim_mask(op)
	right_mask = get_domain_batch_dim_mask(op)
	spreading_dims = get_spreading_dims(op)
	return quote
		check(out, op, inp)
		$(get_single_threaded_loop_expr(left_mask, right_mask, false, spreading_dims))
	end
end

@generated function mul!(
	out::AbstractArray, op::AdjointOperator{opT}, inp::AbstractArray
) where {opT<:SpreadingBatchOpSingleThreaded}
	left_mask = get_domain_batch_dim_mask(opT)
	right_mask = get_codomain_batch_dim_mask(opT)
	spreading_dims = get_spreading_dims(opT)
	return quote
		check(out, op, inp)
		op = op.A
		$(get_single_threaded_loop_expr(left_mask, right_mask, true, spreading_dims))
	end
end

macro threaded_mul_body(is_adj, get_operators_expr)
	input_view_func = is_adj ? :get_codomain_view : :get_domain_view
	output_view_func = is_adj ? :get_domain_view : :get_codomain_view
	return esc(
		quote
			num_threads = min(nthreads(), length(op.batch_indices))
			@restrict_threading @threads for j in 1:num_threads
				@inbounds for i in j:num_threads:length(op.batch_indices)
					idx = op.batch_indices[i]
					mul!(
						$output_view_func(out, op, idx),
						$get_operators_expr,
						$input_view_func(inp, op, idx),
					)
				end
			end
			return out
		end,
	)
end

function mul!(out::AbstractArray, op::SpreadingBatchOpThreadSafe, inp::AbstractArray)
	check(out, op, inp)
	@threaded_mul_body(false, get_threadsafe_spreading_operator(op, idx))
end

function mul!(
	out::AbstractArray,
	op::AdjointOperator{<:SpreadingBatchOpThreadSafe},
	inp::AbstractArray,
)
	check(out, op, inp)
	op = op.A
	@threaded_mul_body(true, get_threadsafe_adj_spreading_operator(op, idx))
end

function mul!(out::AbstractArray, op::SpreadingBatchOpCopying, inp::AbstractArray)
	check(out, op, inp)
	@threaded_mul_body(false, get_copying_spreading_operator(op, idx, j))
end

function mul!(
	out::AbstractArray, op::AdjointOperator{<:SpreadingBatchOpCopying}, inp::AbstractArray
)
	check(out, op, inp)
	op = op.A
	@threaded_mul_body(true, get_copying_adj_spreading_operator(op, idx, j))
end

function mul!(out::AbstractArray, op::SpreadingBatchOpLocking, inp::AbstractArray)
	check(out, op, inp)
	@restrict_threading @sync for idx in op.batch_indices
		op_lock = get_lock(op, idx)
		@spawn begin
			lock(op_lock)
			try
				@inbounds mul!(
					get_codomain_view(out, op, idx),
					get_threadsafe_spreading_operator(op, idx),
					get_domain_view(inp, op, idx),
				)
			finally
				unlock(op_lock)
			end
		end
	end
	return out
end

function mul!(
	out::AbstractArray, op::AdjointOperator{<:SpreadingBatchOpLocking}, inp::AbstractArray
)
	check(out, op, inp)
	op = op.A
	@restrict_threading @sync for idx in op.batch_indices
		op_lock = get_lock(op, idx)
		@spawn begin
			lock(op_lock)
			try
				@inbounds mul!(
					get_domain_view(out, op, idx),
					get_threadsafe_adj_spreading_operator(op, idx),
					get_codomain_view(inp, op, idx),
				)
			finally
				unlock(op_lock)
			end
		end
	end
	return out
end

function mul!(out::AbstractArray, op::SpreadingBatchOpFixedOperator, inp::AbstractArray)
	check(out, op, inp)
	@restrict_threading @threads for op_idx in op.operator_indices
		current_op = op.operators[op_idx]
		@inbounds for batch_idx in op.batch_indices
			idx = merge_indices(op_idx, batch_idx)
			mul!(get_codomain_view(out, op, idx), current_op, get_domain_view(inp, op, idx))
		end
	end
	return out
end

function mul!(
	out::AbstractArray,
	op::AdjointOperator{<:SpreadingBatchOpFixedOperator},
	inp::AbstractArray,
)
	check(out, op, inp)
	op = op.A
	@restrict_threading @threads for op_idx in op.operator_indices
		current_op = op.operators[op_idx]
		@inbounds for batch_idx in op.batch_indices
			idx = merge_indices(op_idx, batch_idx)
			mul!(
				get_domain_view(out, op, idx), current_op', get_codomain_view(inp, op, idx)
			)
		end
	end
	return out
end

# Properties

fun_name(L::SpreadingBatchOp) = "⟳" * fun_name(L.operators[1])
fun_name(L::SpreadingBatchOpCopying) = "⟳" * fun_name(L.operators[1][1])

size(L::SpreadingBatchOp) = L.codomain_size, L.domain_size

domain_storage_type(L::SpreadingBatchOp) = domain_storage_type(L.operators[1])
domain_storage_type(L::SpreadingBatchOpCopying) = domain_storage_type(L.operators[1][1])
codomain_storage_type(L::SpreadingBatchOp) = codomain_storage_type(L.operators[1])
codomain_storage_type(L::SpreadingBatchOpCopying) = codomain_storage_type(L.operators[1][1])

is_linear(L::SpreadingBatchOp) = is_linear(L.operators[1])
is_linear(L::SpreadingBatchOpCopying) = is_linear(L.operators[1][1])
is_eye(L::SpreadingBatchOp) = is_eye(L.operators[1])
is_eye(L::SpreadingBatchOpCopying) = is_eye(L.operators[1][1])
is_AAc_diagonal(L::SpreadingBatchOp) = is_AAc_diagonal(L.operators[1])
is_AAc_diagonal(L::SpreadingBatchOpCopying) = is_AAc_diagonal(L.operators[1][1])
is_AcA_diagonal(L::SpreadingBatchOp) = is_AcA_diagonal(L.operators[1])
is_AcA_diagonal(L::SpreadingBatchOpCopying) = is_AcA_diagonal(L.operators[1][1])
is_full_row_rank(L::SpreadingBatchOp) = is_full_row_rank(L.operators[1])
is_full_row_rank(L::SpreadingBatchOpCopying) = is_full_row_rank(L.operators[1][1])
is_full_column_rank(L::SpreadingBatchOp) = is_full_column_rank(L.operators[1])
is_full_column_rank(L::SpreadingBatchOpCopying) = is_full_column_rank(L.operators[1][1])
is_sliced(L::SpreadingBatchOp) = is_sliced(L.operators[1])
is_sliced(L::SpreadingBatchOpCopying) = is_sliced(L.operators[1][1])
is_null(L::SpreadingBatchOp) = is_null(L.operators[1])
is_null(L::SpreadingBatchOpCopying) = is_null(L.operators[1][1])
is_diagonal(L::SpreadingBatchOp) = is_diagonal(L.operators[1])
is_diagonal(L::SpreadingBatchOpCopying) = is_diagonal(L.operators[1][1])
is_invertible(L::SpreadingBatchOp) = is_invertible(L.operators[1])
is_invertible(L::SpreadingBatchOpCopying) = is_invertible(L.operators[1][1])
is_orthogonal(L::SpreadingBatchOp) = is_orthogonal(L.operators[1])
is_orthogonal(L::SpreadingBatchOpCopying) = is_orthogonal(L.operators[1][1])

function extend_single_diags(single_diags, L::BatchOp{T1,T2,dM,cM}) where {T1,T2,dM,cM}
	input_size = L.domain_size
	output_size = L.codomain_size
	spreading_dims = get_spreading_dims(typeof(L))
	input = allocate_in_domain(L)
	diag_ops = [DiagOp(single_diag) for single_diag in single_diags]
	dT = domain_type(diag_ops[1])
	cT = codomain_type(diag_ops[1])
	sD = spreading_dims
	N = length(input_size)
	M = length(output_size)
	if L isa SpreadingBatchOpSingleThreaded
		batch_size = L.batch_size
	else
		batch_size = size(L.batch_indices)
	end
	B = length(batch_size)
	opT = typeof(diag_ops[1])
	op = SpreadingBatchOpSingleThreaded{dT,cT,dM,cM,sD,N,M,B,opT}(diag_ops, input_size, output_size, batch_size)
	input .= 1
	return op * input
end

function diag_AAc(L::SpreadingBatchOp)
	single_diags = [diag_AAc(op) for op in L.operators]
	if all(
		single_diag isa Number && single_diag == single_diags[1] for
		single_diag in single_diags
	)
		return single_diags[1]
	else
		return extend_single_diags(single_diags, L)
	end
end
function diag_AAc(L::SpreadingBatchOpCopying)
	single_diags = [diag_AAc(op) for op in L.operators[1]]
	if all(
		single_diag isa Number && single_diag == single_diags[1] for
		single_diag in single_diags
	)
		return single_diags[1]
	else
		return extend_single_diags(single_diags, L)
	end
end
function diag_AcA(L::SpreadingBatchOp)
	single_diags = [diag_AcA(op) for op in L.operators]
	if all(
		single_diag isa Number && single_diag == single_diags[1] for
		single_diag in single_diags
	)
		return single_diags[1]
	else
		return extend_single_diags(single_diags, L)
	end
end
function diag_AcA(L::SpreadingBatchOpCopying)
	single_diags = [diag_AcA(op) for op in L.operators[1]]
	if all(
		single_diag isa Number && single_diag == single_diags[1] for
		single_diag in single_diags
	)
		return single_diags[1]
	else
		return extend_single_diags(single_diags, L)
	end
end
function diag(L::SpreadingBatchOp)
	single_diags = [diag(op) for op in L.operators]
	if all(
		single_diag isa Number && single_diag == single_diags[1] for
		single_diag in single_diags
	)
		return single_diags[1]
	else
		return extend_single_diags(single_diags, L)
	end
end
function diag(L::SpreadingBatchOpCopying)
	single_diags = [diag(op) for op in L.operators[1]]
	if all(
		single_diag isa Number && single_diag == single_diags[1] for
		single_diag in single_diags
	)
		return single_diags[1]
	else
		return extend_single_diags(single_diags, L)
	end
end

is_thread_safe(L::SpreadingBatchOp) = is_thread_safe(L.operators[1])
is_thread_safe(L::SpreadingBatchOpCopying) = is_thread_safe(L.operators[1][1])

has_optimized_normalop(L::SpreadingBatchOp) = has_optimized_normalop(L.operators[1])
function has_optimized_normalop(L::SpreadingBatchOpCopying)
	has_optimized_normalop(L.operators[1][1])
end
function get_normal_op(
	L::SpreadingBatchOpSingleThreaded{dT,cT,dM,cM,sD,N,M,B,opT}
) where {dT,cT,dM,cM,sD,N,M,B,opT}
	new_ops = get_normal_op.(L.operators)
	return SpreadingBatchOpSingleThreaded{dT,dT,dM,dM,sD,N,N,B,typeof(new_ops[1])}(
		new_ops, L.domain_size, L.domain_size, L.batch_size
	)
end
function get_normal_op(
	L::SpreadingBatchOpThreadSafe{dT,cT,dM,cM,sD,N,M,opT}
) where {dT,cT,dM,cM,sD,N,M,opT}
	new_ops = get_normal_op.(L.operators)
	return SpreadingBatchOpThreadSafe{dT,cT,dM,dM,sD,N,N,typeof(new_ops[1])}(
		new_ops, L.domain_size, L.codomain_size, L.batch_indices
	)
end
function get_normal_op(
	L::SpreadingBatchOpCopying{dT,cT,dM,cM,sD,N,M,opT}
) where {dT,cT,dM,cM,sD,N,M,opT}
	new_ops = get_normal_op.(L.operators)
	return SpreadingBatchOpCopying{dT,cT,dM,dM,sD,N,N,typeof(new_ops[1])}(
		new_ops, L.domain_size, L.domain_size, L.batch_indices
	)
end
function get_normal_op(
	L::SpreadingBatchOpLocking{dT,cT,dM,cM,sD,N,M,opT}
) where {dT,cT,dM,cM,sD,N,M,opT}
	new_ops = get_normal_op.(L.operators)
	return SpreadingBatchOpLocking{dT,cT,dM,dM,sD,N,N,typeof(new_ops[1])}(
		new_ops, L.domain_size, L.domain_size, L.batch_indices, L.locks
	)
end
function get_normal_op(
	L::SpreadingBatchOpFixedOperator{dT,cT,dM,cM,sD,N,M,opT}
) where {dT,cT,dM,cM,sD,N,M,opT}
	new_ops = get_normal_op.(L.operators)
	return SpreadingBatchOpFixedOperator{dT,cT,dM,dM,sD,N,N,typeof(new_ops[1])}(
		new_ops, L.domain_size, L.domain_size, L.batch_indices, L.operator_indices
	)
end

has_fast_opnorm(L::SpreadingBatchOp) = all(has_fast_opnorm.(L.operators))
LinearAlgebra.opnorm(L::SpreadingBatchOp) = maximum(LinearAlgebra.opnorm.(L.operators))
has_fast_opnorm(L::SpreadingBatchOpCopying) = all(has_fast_opnorm.(L.operators[1]))
LinearAlgebra.opnorm(L::SpreadingBatchOpCopying) = maximum(LinearAlgebra.opnorm.(L.operators[1]))
estimate_opnorm(L::SpreadingBatchOp) = maximum(estimate_opnorm.(L.operators))
estimate_opnorm(L::SpreadingBatchOpCopying) = maximum(estimate_opnorm.(L.operators[1]))

# Utility

function get_batch_dim_mask(op_dims, spreading_dims, batch_size)
	return ntuple(
		i -> if i ≤ op_dims
			:_
		elseif i ≤ op_dims + spreading_dims
			:s
		else
			:b
		end,
		op_dims + spreading_dims + length(batch_size),
	)
end

function get_spreading_dims(
	::Type{<:SpreadingBatchOpSingleThreaded{T1,T2,dM,cM,sD}}
) where {T1,T2,dM,cM,sD}
	return sD
end
function get_spreading_dims(
	::Type{<:SpreadingBatchOpThreadSafe{T1,T2,dM,cM,sD}}
) where {T1,T2,dM,cM,sD}
	return sD
end
function get_spreading_dims(
	::Type{<:SpreadingBatchOpCopying{T1,T2,dM,cM,sD}}
) where {T1,T2,dM,cM,sD}
	return sD
end
function get_spreading_dims(
	::Type{<:SpreadingBatchOpLocking{T1,T2,dM,cM,sD}}
) where {T1,T2,dM,cM,sD}
	return sD
end
function get_spreading_dims(
	::Type{<:SpreadingBatchOpFixedOperator{T1,T2,dM,cM,sD}}
) where {T1,T2,dM,cM,sD}
	return sD
end

@generated function get_threadsafe_spreading_operator(op, batch_idx)
	return get_spreading_operator_item_expr(get_spreading_dims(op), :(op.operators))
end

@generated function get_threadsafe_adj_spreading_operator(op, batch_idx)
	return Expr(
		Symbol("'"),
		get_spreading_operator_item_expr(get_spreading_dims(op), :(op.operators)),
	)
end

@generated function get_copying_spreading_operator(op, batch_idx, thread_idx)
	return get_spreading_operator_item_expr(
		get_spreading_dims(op), :(op.operators[thread_idx])
	)
end

@generated function get_copying_adj_spreading_operator(op, batch_idx, thread_idx)
	return Expr(
		Symbol("'"),
		get_spreading_operator_item_expr(
			get_spreading_dims(op), :(op.operators[thread_idx])
		),
	)
end

function get_spreading_operator_item_expr(spreading_dims, op_expr)
	return Expr(:ref, op_expr, [Expr(:ref, :batch_idx, i) for i in spreading_dims]...)
end

@generated function get_lock(op, batch_idx)
	return get_spreading_operator_item_expr(get_spreading_dims(op), :(op.locks))
end

@generated function merge_indices(
	op_index::CartesianIndex{N}, batch_index::CartesianIndex{N}
) where {N}
	return Expr(
		:call,
		:CartesianIndex,
		[
			Expr(:call, :-, Expr(:call, :+, :(op_index[$i]), :(batch_index[$i])), 1) for
			i in 1:N
		]...,
	)
end
