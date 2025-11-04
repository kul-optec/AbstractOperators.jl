export BatchOp, ThreadingStrategy

abstract type BatchOp{dT,cT,dM,cM} <: AbstractOperators.AbstractOperator end

# Properties

domain_type(::BatchOp{dT}) where {dT} = dT
codomain_type(::BatchOp{dT,cT}) where {dT,cT} = cT

# Utility

function ensure_batch_size_is_tuple(batch_size::Int)
	return (batch_size,)
end
function ensure_batch_size_is_tuple(batch_size::AbstractVector)
	return ntuple(i -> batch_size[i], length(batch_size))
end
function ensure_batch_size_is_tuple(batch_size::NTuple)
	return batch_size
end
function ensure_batch_size_is_tuple(::Any)
	error("Batch size must be an Int, Vector, or NTuple")
end

get_domain_batch_dim_mask(::Type{<:BatchOp{T1,T2,dM,cM}}) where {T1,T2,dM,cM} = dM
get_codomain_batch_dim_mask(::Type{<:BatchOp{T1,T2,dM,cM}}) where {T1,T2,dM,cM} = cM

copy_op(x) = deepcopy(x)
function copy_op(x::T) where {T<:AbstractOperator}
	return if AbstractOperators.is_thread_safe(x)
		x
	else
		T.name.wrapper([copy_op(getfield(x, k)) for k in fieldnames(T)]...)
	end
end

function prepare_batch_op(
	operator::AbstractOperators.AbstractOperator,
	domain_size::NTuple{M1,Int},
	domain_batch_dim_mask::NTuple{M2,Bool},
	codomain_size::NTuple{N1,Int},
	codomain_batch_dim_mask::NTuple{N2,Bool},
) where {M1,M2,N1,N2}
	@assert M1 == M2 "Domain size and domain batch dimension mask must have the same length"
	@assert N1 == N2 "Codomain size and codomain batch dimension mask must have the same length"
	@assert sum(domain_batch_dim_mask) == sum(codomain_batch_dim_mask) "Number of batch dimensions in input and output must match"
	domain_batch_size = [
		domain_size[d] for d in 1:length(domain_size) if domain_batch_dim_mask[d]
	]
	codomain_batch_size = [
		codomain_size[d] for d in 1:length(codomain_size) if codomain_batch_dim_mask[d]
	]
	@assert domain_batch_size == codomain_batch_size "Batch dimensions must have the same size in input and output"
	input_block_size = tuple(
		[domain_size[d] for d in 1:length(domain_size) if !domain_batch_dim_mask[d]]...
	)
	@assert size(operator, 2) == input_block_size "Operator input size does not match input block size"
	output_block_size = tuple(
		[
			codomain_size[d] for d in 1:length(codomain_size) if !codomain_batch_dim_mask[d]
		]...,
	)
	@assert size(operator, 1) == output_block_size "Operator output size does not match output block size"
	dType, cdType = domain_type(operator), codomain_type(operator)
	return tuple(domain_batch_size...), dType, cdType
end

function symbol_mask_to_bool(mask::NTuple{N,Symbol}) where {N}
	return tuple([m == :b || m == :s for m in mask]...)
end
function symbol_mask_to_bool(mask::Pair{NTuple{N1,Symbol},NTuple{N2,Symbol}}) where {N1,N2}
	return Pair(symbol_mask_to_bool(mask.first), symbol_mask_to_bool(mask.second))
end

function calculate_shapes(
	operator::AbstractOperators.AbstractOperator,
	batch_size::NTuple{N,Int},
	batch_dim_mask::Pair{NTuple{M1,Bool},NTuple{M2,Bool}},
) where {N,M1,M2}
	@assert ndims(operator, 2) == sum(.!, batch_dim_mask.first) "Number of domain non-batch dimensions in batch_dim_mask must match operator domain size"
	@assert ndims(operator, 1) == sum(.!, batch_dim_mask.second) "Number of codomain non-batch dimensions in batch_dim_mask must match operator codomain size"
	@assert length(batch_size) == sum(batch_dim_mask.first) "Number of domain batch dimensions in batch_dim_mask must match batch size"
	@assert length(batch_size) == sum(batch_dim_mask.second) "Number of codomain batch dimensions in batch_dim_mask must match batch size"
	domain_ndim = length(batch_dim_mask.first)
	codomain_ndim = length(batch_dim_mask.second)
	domain_op_size = size(operator, 2)
	domain_size = Vector{Int}(undef, domain_ndim)
	batch_dim_counter, operator_dim_counter = 1, 1
	for i in 1:domain_ndim
		if batch_dim_mask.first[i]
			domain_size[i] = batch_size[batch_dim_counter]
			batch_dim_counter += 1
		else
			domain_size[i] = domain_op_size[operator_dim_counter]
			operator_dim_counter += 1
		end
	end
	codomain_op_size = size(operator, 1)
	codomain_size = Vector{Int}(undef, codomain_ndim)
	batch_dim_counter, operator_dim_counter = 1, 1
	for i in 1:codomain_ndim
		if batch_dim_mask.second[i]
			codomain_size[i] = batch_size[batch_dim_counter]
			batch_dim_counter += 1
		else
			codomain_size[i] = codomain_op_size[operator_dim_counter]
			operator_dim_counter += 1
		end
	end
	return tuple(domain_size...),
	batch_dim_mask.first, tuple(codomain_size...),
	batch_dim_mask.second
end

function get_view(array_expr, batch_idx_expr, m)
	N = length(m)
	indices = Vector{Union{Colon,Expr}}(undef, N)
	counter = 1
	for i in 1:N
		if m[i]
			indices[i] = Expr(:ref, batch_idx_expr, counter)
			counter += 1
		else
			indices[i] = Colon()
		end
	end
	return Expr(:call, :view, array_expr, indices...)
end

@generated function get_domain_view(array, op, batch_idx)
	return get_view(:array, :batch_idx, get_domain_batch_dim_mask(op))
end

@generated function get_codomain_view(array, op, batch_idx)
	return get_view(:array, :batch_idx, get_codomain_batch_dim_mask(op))
end

function get_single_threaded_loop_expr(
	left_mask, right_mask, is_adj, spreading_dims=nothing
)
	batch_ndims = sum(left_mask)
	loop_vars = [Symbol("i_", i) for i in 1:batch_ndims]
	loop_var_exprs = Expr(
		:block, [:($(loop_vars[i]) = 1:op.batch_size[$i]) for i in 1:batch_ndims]...
	)
	left_view_indices = Vector{Any}(undef, length(left_mask))
	batch_counter = 1
	for i in eachindex(left_mask)
		if left_mask[i]
			left_view_indices[i] = loop_vars[batch_counter]
			batch_counter += 1
		else
			left_view_indices[i] = Colon()
		end
	end
	right_view_indices = Vector{Any}(undef, length(right_mask))
	batch_counter = 1
	for i in eachindex(right_mask)
		if right_mask[i]
			right_view_indices[i] = loop_vars[batch_counter]
			batch_counter += 1
		else
			right_view_indices[i] = Colon()
		end
	end
	left_view = Expr(:call, :view, :out, left_view_indices...)
	right_view = Expr(:call, :view, :inp, right_view_indices...)
	if spreading_dims === nothing
		op_expr = is_adj ? :(op.operator') : :(op.operator)
	else
		operator_indices = [loop_vars[i] for i in spreading_dims]
		op_expr = Expr(:ref, Expr(:(.), :op, :(:operators)), operator_indices...)
		if is_adj
			op_expr = Expr(Symbol("'"), op_expr)
		end
	end
	loop_body = Expr(:block, Expr(:call, :mul!, left_view, op_expr, right_view))
	loop_expr = Expr(:for, loop_var_exprs, loop_body)
	return quote
		@inbounds $loop_expr
		return out
	end
end
