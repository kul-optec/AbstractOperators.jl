abstract type SimpleBatchOp{dT,cT,dM,cM} <: BatchOp{dT,cT,dM,cM} end

struct SimpleBatchOpSingleThreaded{dT,cT,dM,cM,opT,N,M,B} <: SimpleBatchOp{dT,cT,dM,cM}
	operator::opT
	domain_size::NTuple{N,Int}
	codomain_size::NTuple{M,Int}
	batch_size::NTuple{B,Int}
end

struct SimpleBatchOpMultiThreaded{dT,cT,dM,cM,opT,N,M,C} <: SimpleBatchOp{dT,cT,dM,cM}
	operator::NTuple{C,opT}
	domain_size::NTuple{N,Int}
	codomain_size::NTuple{M,Int}
	batch_indices::CartesianIndices
end

# Public constructors

"""
	function BatchOp(
		operator::AbstractOperators.AbstractOperator,
		batch_size::NTuple{N,Int},
		batch_dim_mask::Union{NTuple{M,Symbol}, Pair{NTuple{M1,Symbol},NTuple{M2,Symbol}}};
		threaded::Bool=nthreads() > 1,
	)

Creates a "simple" `BatchOp` from an `AbstractOperator`. The `BatchOp` can be used to apply the operator to an array over
the selected batch dimensions.

# Arguments
- `operator::AbstractOperators.AbstractOperator`: The operator to be batched.
- `batch_size::NTuple{N,Int}`: The size of the batch dimensions.
- `batch_dim_mask::Union{NTuple{M,Symbol}, Pair{NTuple{M1,Symbol},NTuple{M2,Symbol}}}`: The mask specifying which dimensions are batch dimensions. The symbols can be `:b` for batch dimensions or `:_` for dimensions on which the operator acts. If a pair is provided, the first tuple specifies the domain mask and the second tuple specifies the codomain mask.
- `threaded::Bool`: If `true`, the operator will execute in parallel over the batch dimensions. Default is `nthreads() > 1`.

# Examples
```jldoctest
julia> op = DiagOp([1.0im, 1.0im])
╲  ℂ^2 -> ℂ^2

julia> batch_op = BatchOp(op, (3, 4), (:_, :b, :b))
⟳╲  ℂ^(2, 3, 4) -> ℂ^(2, 3, 4)

julia> x = rand(ComplexF64, 2, 3, 4);

julia> y = similar(x);

julia> [mul!(@view(y[:, i, j]), op, @view(x[:, i, j])) for i in 1:3, j in 1:4];

julia> y == batch_op*x
true

julia> op = Variation(3, 4, 5)
Ʋ  ℝ^(3, 4, 5) -> ℝ^(60, 3)

julia> batch_op = BatchOp(op, (2, 6), (:b, :_, :_, :_, :b) => (:b, :_, :b, :_))
⟳Ʋ  ℝ^(2, 3, 4, 5, 6) -> ℝ^(2, 60, 6, 3)
	
```
"""
function BatchOp(
	operator::AbstractOperators.AbstractOperator,
	batch_size::NTuple{N,Int},
	batch_dim_mask::NTuple{M,Symbol};
	threaded::Bool=nthreads() > 1,
) where {N,M}
	return create_BatchOp(
		operator, batch_size, symbol_mask_to_bool(batch_dim_mask); threaded
	)
end

function BatchOp(
	operator::AbstractOperators.AbstractOperator,
	batch_size::NTuple{N,Int},
	batch_dim_mask::Pair{NTuple{M1,Symbol},NTuple{M2,Symbol}};
	threaded::Bool=nthreads() > 1,
) where {N,M1,M2}
	return create_BatchOp(
		operator, batch_size, symbol_mask_to_bool(batch_dim_mask); threaded
	)
end

# Internal constructors

function create_BatchOp(
	operator::AbstractOperators.AbstractOperator,
	batch_size::NTuple{N,Int},
	batch_dim_mask::NTuple{M,Bool};
	threaded::Bool=nthreads() > 1,
) where {N,M}
	@assert ndims(operator, 1) == ndims(operator, 2) "Operator must be square or batch_dim_mask must be a pair of domain and codomain masks"
	return create_BatchOp(operator, batch_size, batch_dim_mask => batch_dim_mask; threaded)
end

function create_BatchOp(
	operator::AbstractOperators.AbstractOperator,
	batch_size::NTuple{N,Int},
	batch_dim_mask::Pair{NTuple{M1,Bool},NTuple{M2,Bool}};
	threaded::Bool=nthreads() > 1,
) where {N,M1,M2}
	return create_BatchOp(
		operator, calculate_shapes(operator, batch_size, batch_dim_mask)...; threaded
	)
end

function create_BatchOp(
	operator::AbstractOperators.AbstractOperator,
	domain_size::NTuple{N,Int},
	domain_batch_dim_mask::NTuple{N2,Bool},
	codomain_size::NTuple{M,Int},
	codomain_batch_dim_mask::NTuple{M2,Bool};
	threaded::Bool=nthreads() > 1,
) where {N,N2,M,M2}
	@assert M == M2 "Domain size and domain batch dimension mask must have the same length"
	@assert N == N2 "Codomain size and codomain batch dimension mask must have the same length"
	batch_size, dType, cdType, opType = prepare_batch_op(
		operator, domain_size, domain_batch_dim_mask, codomain_size, codomain_batch_dim_mask
	)
	return if threaded && nthreads() > 1
		batch_length = prod(batch_size)
		operators = tuple(
			[
				i == 1 ? operator : copy_op(operator) for
				i in 1:min(nthreads(), batch_length)
			]...,
		)
		C = length(operators)
		SimpleBatchOpMultiThreaded{
			dType,cdType,domain_batch_dim_mask,codomain_batch_dim_mask,opType,N,M,C
		}(
			operators, domain_size, codomain_size, CartesianIndices(batch_size)
		)
	else
		B = length(batch_size)
		SimpleBatchOpSingleThreaded{
			dType,cdType,domain_batch_dim_mask,codomain_batch_dim_mask,opType,N,M,B
		}(
			operator, domain_size, codomain_size, batch_size
		)
	end
end

# mul! implementations

@generated function mul!(
	out::AbstractArray, op::SimpleBatchOpSingleThreaded, inp::AbstractArray
)
	left_mask = get_codomain_batch_dim_mask(op)
	right_mask = get_domain_batch_dim_mask(op)
	return quote
		check(inp, out, op)
		$(get_single_threaded_loop_expr(left_mask, right_mask, false))
	end
end

@generated function mul!(
	out::AbstractArray, op::AdjointOperator{opT}, inp::AbstractArray
) where {opT<:SimpleBatchOpSingleThreaded}
	left_mask = get_domain_batch_dim_mask(opT)
	right_mask = get_codomain_batch_dim_mask(opT)
	return quote
		op = op.A
		check(out, inp, op)
		$(get_single_threaded_loop_expr(left_mask, right_mask, true))
	end
end

function mul!(out::AbstractArray, op::SimpleBatchOpMultiThreaded, inp::AbstractArray)
	check(inp, out, op)
	@threads for j in 1:length(op.operator)
		@inbounds for i in j:length(op.operator):length(op.batch_indices)
			idx = op.batch_indices[i]
			mul!(
				get_codomain_view(out, op, idx),
				op.operator[j],
				get_domain_view(inp, op, idx),
			)
		end
	end
	return out
end

function mul!(
	out::AbstractArray,
	op::AdjointOperator{<:SimpleBatchOpMultiThreaded},
	inp::AbstractArray,
)
	op = op.A
	check(out, inp, op)
	@threads for j in 1:length(op.operator)
		@inbounds for i in j:length(op.operator):length(op.batch_indices)
			idx = op.batch_indices[i]
			mul!(
				get_domain_view(out, op, idx),
				op.operator[j]',
				get_codomain_view(inp, op, idx),
			)
		end
	end
	return out
end

function get_normal_op(L::SimpleBatchOpSingleThreaded)
	return SimpleBatchOpSingleThreaded(get_normal_op(L.operator), L.domain_size, L.codomain_size, L.batch_size)
end
function get_normal_op(L::SimpleBatchOpMultiThreaded)
	return SimpleBatchOpMultiThreaded(
		get_normal_op.(L.operator), L.domain_size, L.codomain_size, L.batch_indices
	)
end

# Properties

fun_name(L::SimpleBatchOpSingleThreaded) = "⟳" * fun_name(L.operator)
fun_name(L::SimpleBatchOpMultiThreaded) = "⟳" * fun_name(L.operator[1])

size(L::SimpleBatchOp) = L.codomain_size, L.domain_size

function domain_storage_type(L::SimpleBatchOpSingleThreaded)
	return extend_domain_storage_type(L, L.operator)
end
function domain_storage_type(L::SimpleBatchOpMultiThreaded)
	return extend_domain_storage_type(L, L.operator[1])
end
function codomain_storage_type(L::SimpleBatchOpSingleThreaded)
	return extend_codomain_storage_type(L, L.operator)
end
function codomain_storage_type(L::SimpleBatchOpMultiThreaded)
	return extend_codomain_storage_type(L, L.operator[1])
end

is_linear(L::SimpleBatchOpSingleThreaded) = is_linear(L.operator)
is_linear(L::SimpleBatchOpMultiThreaded) = is_linear(L.operator[1])
is_eye(L::SimpleBatchOpSingleThreaded) = is_eye(L.operator)
is_eye(L::SimpleBatchOpMultiThreaded) = is_eye(L.operator[1])

is_thread_safe(L::SimpleBatchOpSingleThreaded) = is_thread_safe(L.operator)
is_thread_safe(L::SimpleBatchOpMultiThreaded) = is_thread_safe(L.operator[1])

LinearAlgebra.opnorm(L::SimpleBatchOpSingleThreaded) = LinearAlgebra.opnorm(L.operator)
LinearAlgebra.opnorm(L::SimpleBatchOpMultiThreaded) = LinearAlgebra.opnorm(L.operator[1])
