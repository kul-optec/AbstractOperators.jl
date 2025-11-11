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
	BatchOp(
		operator::AbstractOperators.AbstractOperator,
		batch_dims::NTuple{N,Int},
		batch_dim_mask::Union{NTuple{M,Symbol}, Pair{NTuple{M1,Symbol},NTuple{M2,Symbol}}};
		threaded::Bool=nthreads() > 1,
	)

Creates a "simple" `BatchOp` from an `AbstractOperator`. The `BatchOp` can be used to apply the operator to an array over
the selected batch dimensions.

# Arguments
- `operator::AbstractOperators.AbstractOperator`: The operator to be batched.
- `batch_dims::NTuple{N,Int}`: The size of the batch dimensions. This tuple should contain the sizes of the batch dimensions in the order they
   appear in the operator's domain and codomain.
- `batch_dim_mask::Union{NTuple{M,Symbol}, Pair{NTuple{M1,Symbol},NTuple{M2,Symbol}}}`: The mask specifying which dimensions are batch dimensions.
   The symbols can be `:b` for batch dimensions or `:_` for dimensions on which the operator acts. If a pair is provided, the first tuple specifies
   the domain mask and the second tuple specifies the codomain mask. When omitted, the batch dimensions are assumed to proceed the operator's
   domain and codomain dimensions.
- `threaded::Bool`: If `true`, the operator will execute in parallel over the batch dimensions. Default is `nthreads() > 1`.

# Examples
```jldoctest
julia> op = DiagOp([1.0im, 1.0im])
╲  ℂ^2 -> ℂ^2

julia> batch_op = BatchOp(op, (3, 4), (:_, :b, :b))
⟳╲  ℂ^(2, 3, 4) -> ℂ^(2, 3, 4)

julia> batch_op = BatchOp(op, (3, 4)) # we don't need to specify the mask as the default is to assume the batch dimensions are at the end
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
	batch_size;
	threaded::Bool=nthreads() > 1,
)
	batch_size = ensure_batch_size_is_tuple(batch_size)
	N = length(batch_size)
	@assert N > 0 "Batch size must be a non-empty tuple"
	in_dims = ndims(operator, 2)
	out_dims = ndims(operator, 1)
	batch_dim_mask = ntuple(i -> i <= in_dims ? :_ : :b, in_dims + N) => ntuple(i -> i <= out_dims ? :_ : :b, out_dims + N)
	return create_BatchOp(
		operator, batch_size, symbol_mask_to_bool(batch_dim_mask); threaded
	)
end

function BatchOp(
	operator::AbstractOperators.AbstractOperator,
	batch_size,
	batch_dim_mask::NTuple{M,Symbol};
	threaded::Bool=nthreads() > 1,
) where {M}
	batch_size = ensure_batch_size_is_tuple(batch_size)
	N = length(batch_size)
	@assert N > 0 "Batch size must be a non-empty tuple"
	return create_BatchOp(
		operator, batch_size, symbol_mask_to_bool(batch_dim_mask); threaded
	)
end

function BatchOp(
	operator::AbstractOperators.AbstractOperator,
	batch_size,
	batch_dim_mask::Pair{NTuple{M1,Symbol},NTuple{M2,Symbol}};
	threaded::Bool=nthreads() > 1,
) where {M1,M2}
	batch_size = ensure_batch_size_is_tuple(batch_size)
	N = length(batch_size)
	@assert N > 0 "Batch size must be a non-empty tuple"
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
	batch_size, dType, cdType = prepare_batch_op(
		operator, domain_size, domain_batch_dim_mask, codomain_size, codomain_batch_dim_mask
	)
	opType = typeof(operator)
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
		check(out, op, inp)
		$(get_single_threaded_loop_expr(left_mask, right_mask, false))
	end
end

@generated function mul!(
	out::AbstractArray, op::AdjointOperator{opT}, inp::AbstractArray
) where {opT<:SimpleBatchOpSingleThreaded}
	left_mask = get_domain_batch_dim_mask(opT)
	right_mask = get_codomain_batch_dim_mask(opT)
	return quote
		check(out, op, inp)
		op = op.A
		$(get_single_threaded_loop_expr(left_mask, right_mask, true))
	end
end

function mul!(out::AbstractArray, op::SimpleBatchOpMultiThreaded, inp::AbstractArray)
	check(out, op, inp)
	@restrict_threading @threads for j in 1:length(op.operator)
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
	check(out, op, inp)
	op = op.A
	@restrict_threading @threads for j in 1:length(op.operator)
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

# Properties

fun_name(L::SimpleBatchOpSingleThreaded) = "⟳" * fun_name(L.operator)
fun_name(L::SimpleBatchOpMultiThreaded) = "⟳" * fun_name(L.operator[1])

size(L::SimpleBatchOp) = L.codomain_size, L.domain_size

domain_storage_type(L::SimpleBatchOpSingleThreaded) = domain_storage_type(L.operator)
domain_storage_type(L::SimpleBatchOpMultiThreaded) = domain_storage_type(L.operator[1])
codomain_storage_type(L::SimpleBatchOpSingleThreaded) = codomain_storage_type(L.operator)
codomain_storage_type(L::SimpleBatchOpMultiThreaded) = codomain_storage_type(L.operator[1])

is_linear(L::SimpleBatchOpSingleThreaded) = is_linear(L.operator)
is_linear(L::SimpleBatchOpMultiThreaded) = is_linear(L.operator[1])
is_eye(L::SimpleBatchOpSingleThreaded) = is_eye(L.operator)
is_eye(L::SimpleBatchOpMultiThreaded) = is_eye(L.operator[1])
is_null(L::SimpleBatchOpSingleThreaded) = is_null(L.operator)
is_null(L::SimpleBatchOpMultiThreaded) = is_null(L.operator[1])
is_AAc_diagonal(L::SimpleBatchOpSingleThreaded) = is_AAc_diagonal(L.operator)
is_AAc_diagonal(L::SimpleBatchOpMultiThreaded) = is_AAc_diagonal(L.operator[1])
is_AcA_diagonal(L::SimpleBatchOpSingleThreaded) = is_AcA_diagonal(L.operator)
is_AcA_diagonal(L::SimpleBatchOpMultiThreaded) = is_AcA_diagonal(L.operator[1])
is_invertible(L::SimpleBatchOpSingleThreaded) = is_invertible(L.operator)
is_invertible(L::SimpleBatchOpMultiThreaded) = is_invertible(L.operator[1])
is_full_row_rank(L::SimpleBatchOpSingleThreaded) = is_full_row_rank(L.operator)
is_full_row_rank(L::SimpleBatchOpMultiThreaded) = is_full_row_rank(L.operator[1])
is_full_column_rank(L::SimpleBatchOpSingleThreaded) = is_full_column_rank(L.operator)
is_full_column_rank(L::SimpleBatchOpMultiThreaded) = is_full_column_rank(L.operator[1])
is_sliced(L::SimpleBatchOpSingleThreaded) = is_sliced(L.operator)
is_sliced(L::SimpleBatchOpMultiThreaded) = is_sliced(L.operator[1])
is_diagonal(L::SimpleBatchOpSingleThreaded) = is_diagonal(L.operator)
is_diagonal(L::SimpleBatchOpMultiThreaded) = is_diagonal(L.operator[1])

function extend_single_diag(single_diag, domain_mask, output_size)
	output = similar(single_diag, output_size)
	reshape_size = map(eachindex(output_size)) do dim
		domain_mask[dim] ? 1 : output_size[dim]
	end
	output .= reshape(single_diag, reshape_size...)
	return output
end

function diag_AAc(L::SimpleBatchOpSingleThreaded)
	single_diag = diag_AAc(L.operator)
	if single_diag isa Number
		return single_diag
	else
		domain_mask = get_codomain_batch_dim_mask(typeof(L))
		return extend_single_diag(single_diag, domain_mask, L.codomain_size)
	end
end
function diag_AAc(L::SimpleBatchOpMultiThreaded)
	single_diag = diag_AAc(L.operator[1])
	if single_diag isa Number
		return single_diag
	else
		domain_mask = get_codomain_batch_dim_mask(typeof(L))
		return extend_single_diag(single_diag, domain_mask, L.codomain_size)
	end
end
function diag_AcA(L::SimpleBatchOpSingleThreaded)
	single_diag = diag_AcA(L.operator)
	if single_diag isa Number
		return single_diag
	else
		domain_mask = get_domain_batch_dim_mask(typeof(L))
		return extend_single_diag(single_diag, domain_mask, L.domain_size)
	end
end
function diag_AcA(L::SimpleBatchOpMultiThreaded)
	single_diag = diag_AcA(L.operator[1])
	if single_diag isa Number
		return single_diag
	else
		domain_mask = get_domain_batch_dim_mask(typeof(L))
		return extend_single_diag(single_diag, domain_mask, L.domain_size)
	end
end
function diag(L::SimpleBatchOpSingleThreaded)
	single_diag = diag(L.operator)
	if single_diag isa Number
		return single_diag
	else
		domain_mask = get_domain_batch_dim_mask(typeof(L))
		return extend_single_diag(single_diag, domain_mask, L.domain_size)
	end
end
function diag(L::SimpleBatchOpMultiThreaded)
	single_diag = diag(L.operator[1])
	if single_diag isa Number
		return single_diag
	else
		domain_mask = get_domain_batch_dim_mask(typeof(L))
		return extend_single_diag(single_diag, domain_mask, L.domain_size)
	end
end

is_thread_safe(L::SimpleBatchOpSingleThreaded) = is_thread_safe(L.operator)
is_thread_safe(L::SimpleBatchOpMultiThreaded) = is_thread_safe(L.operator[1])

has_optimized_normalop(L::SimpleBatchOpSingleThreaded) = has_optimized_normalop(L.operator)
function has_optimized_normalop(L::SimpleBatchOpMultiThreaded)
	return has_optimized_normalop(L.operator[1])
end
function get_normal_op(
	L::SimpleBatchOpSingleThreaded{dT,cT,dM,cM,opT,N,M,C}
) where {dT,cT,dM,cM,opT,N,M,C}
	new_op = get_normal_op(L.operator)
	return SimpleBatchOpSingleThreaded{dT,cT,dM,dM,typeof(new_op),N,N,C}(
		new_op, L.domain_size, L.domain_size, L.batch_size
	)
end
function get_normal_op(
	L::SimpleBatchOpMultiThreaded{dT,cT,dM,cM,opT,N,M,C}
) where {dT,cT,dM,cM,opT,N,M,C}
	new_op = get_normal_op(L.operator[1])
	new_ops = tuple([i == 1 ? new_op : copy_op(new_op) for i in 1:length(L.operator)]...)
	return SimpleBatchOpMultiThreaded{dT,cT,dM,dM,typeof(new_op),N,N,C}(
		new_ops, L.domain_size, L.domain_size, L.batch_indices
	)
end

has_fast_opnorm(L::SimpleBatchOpSingleThreaded) = has_fast_opnorm(L.operator)
has_fast_opnorm(L::SimpleBatchOpMultiThreaded) = has_fast_opnorm(L.operator[1])
function LinearAlgebra.opnorm(L::SimpleBatchOpSingleThreaded)
	return LinearAlgebra.opnorm(L.operator)
end
function LinearAlgebra.opnorm(L::SimpleBatchOpMultiThreaded)
	return LinearAlgebra.opnorm(L.operator[1])
end
function estimate_opnorm(L::SimpleBatchOpSingleThreaded)
	return estimate_opnorm(L.operator)
end
function estimate_opnorm(L::SimpleBatchOpMultiThreaded)
	return estimate_opnorm(L.operator[1])
end
