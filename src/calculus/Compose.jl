export Compose

"""
	Compose(A::AbstractOperator,B::AbstractOperator)

Shorthand constructor:

	A*B

Compose different `AbstractOperator`s. Notice that the domain and codomain of the operators `A` and `B` must match, i.e. `size(A,2) == size(B,1)` and `domain_type(A) == codomain_type(B)`.

```jldoctest
julia> Compose(Variation((4,4)), FiniteDiff((5,4),1))
Ʋ*δx  ℝ^(5, 4) -> ℝ^(16, 2)

julia> MatrixOp(randn(20,10))*FiniteDiff((11,))
▒*δx  ℝ^11 -> ℝ^20

```
"""
struct Compose{N,M,L<:Tuple,T<:Tuple} <: AbstractOperator
	A::L
	buf::T       # memory in the buffer of the operators
	function Compose(A::L, buf::T) where {L<:Tuple,T<:Tuple}
		if length(A) - 1 != length(buf)
			throw(
				DimensionMismatch(
					"number of operators ($(length(A))) and buffers ($(length(buf))) do not match",
				),
			)
		end
		# check for adjacent operators that can be combined
		i = 1
		while i < length(A)
			should_be_combined = false
			triple_combination = false
			if (
				A[i + 1] isa AdjointOperator &&
				A[i + 1].A == A[i] &&
				has_optimized_normalop(A[i])
			)
				DEBUG_COMPOSE[] && print("Replacing $(typeof((A[i])).name.wrapper) and $(typeof((A[i + 1])).name.wrapper) with normal operator")
				new_op = get_normal_op(A[i])
				should_be_combined = true
			elseif can_be_combined(A[i + 1], A[i])
				DEBUG_COMPOSE[] && print("Combining $(typeof((A[i + 1])).name.wrapper) and $(typeof((A[i])).name.wrapper)")
				new_op = combine(A[i + 1], A[i])
				if DEBUG_COMPOSE[]
					new_ops = new_op isa Compose ? get_operators(new_op) : (new_op,)
					new_ops_strs = map(op -> typeof(op).name.wrapper, new_ops)
					new_ops_str = length(new_ops) > 1 ? "($(join(new_ops_strs, ", ")))" : new_ops_strs[1]
					println(" to $new_ops_str")
				end
				should_be_combined = true
			elseif i + 2 <= length(A) && can_be_combined(A[i + 2], A[i + 1], A[i])
				DEBUG_COMPOSE[] && print("Combining $(typeof((A[i + 2])).name.wrapper), $(typeof((A[i + 1])).name.wrapper) and $(typeof((A[i])).name.wrapper)")
				new_op = combine(A[i + 2], A[i + 1], A[i])
				if DEBUG_COMPOSE[]
					new_ops = new_op isa Compose ? get_operators(new_op) : (new_op,)
					new_ops_strs = map(op -> typeof(op).name.wrapper, new_ops)
					new_ops_str = length(new_ops) > 1 ? "($(join(new_ops_strs, ", ")))" : new_ops_strs[1]
					println(" to $new_ops_str")
				end
				should_be_combined = true
				triple_combination = true
			end
			next_i = i + (triple_combination ? 2 : 1)
			if should_be_combined && new_op isa Compose
				DEBUG_COMPOSE[] && println("Current operators: $(map(op -> typeof(op).name.wrapper, A))")
				A = (A[1:(i - 1)]..., new_op.A..., A[(next_i + 1):end]...)
				DEBUG_COMPOSE[] && println("New operators: $(map(op -> typeof(op).name.wrapper, A))")
				buf = (buf[1:(i - 1)]..., new_op.buf..., buf[next_i:end]...)
				if i > 1
					i -= 1 # maybe the previous operator can be combined with the new one
				end
			elseif should_be_combined
				DEBUG_COMPOSE[] && println("Current operators: $(map(op -> typeof(op).name.wrapper, A))")
				A = (A[1:(i - 1)]..., new_op, A[(next_i + 1):end]...)
				DEBUG_COMPOSE[] && println("New operators: $(map(op -> typeof(op).name.wrapper, A))")
				buf = (buf[1:(i - 1)]..., buf[next_i:end]...)
				if i > 1 && length(buf) >= i
					if buf[i - 1] === buf[i] # check for re-used buffers that might have become adjacent
						buf = (
							buf[1:(i - 1)]...,
							allocate_in_codomain(new_op),
							buf[next_i:end]...,
						)
					end
					i -= 1 # maybe the previous operator can be combined with the new one
				end
			else
				i += 1
			end
		end
		if length(A) == 1
			return A[1]
		end
		return new{length(A),length(buf),typeof(A),typeof(buf)}(
			A, buf
		)
	end
end

# Constructors

function Compose(L1::AbstractOperator, L2::AbstractOperator)
	if size(L1, 2) != size(L2, 1)
		msg = "cannot compose operators with different domain and codomain sizes"
		throw(DomainError((size(L1, 2), size(L2, 1)), msg))
	end
	if domain_type(L1) != codomain_type(L2)
		msg = "cannot compose operators with different domain and codomain types"
		throw(DomainError((domain_type(L1), codomain_type(L2)), msg))
	end
	if domain_storage_type(L1) != codomain_storage_type(L2)
		msg = "cannot compose operators with different input and output storage types"
		throw(DomainError((domain_storage_type(L1), codomain_storage_type(L2)), msg))
	end
	if L1 isa AdjointOperator && L1.A == L2 && has_optimized_normalop(L2)
		return get_normal_op(L2)
	end
	if !(L1 isa Compose || L2 isa Compose) && can_be_combined(L1, L2)
		return combine(L1, L2)
	end
	available_bufs = ()
	if L1 isa Compose && length(L1.A) > 1
		available_bufs = L1.buf[2:end]
	end
	if L2 isa Compose && length(L2.A) > 1
		available_bufs = (available_bufs..., L2.buf[1:(end - 1)]...)
	end
	compatible_bufs =
		x ->
			x isa codomain_storage_type(L1) &&
			size(x) == size(L2, 1) &&
			eltype(x) == codomain_type(L2)
	new_buf_pos = findfirst(compatible_bufs, available_bufs)
	new_buf =
		new_buf_pos === nothing ? allocate_in_codomain(L2) : available_bufs[new_buf_pos]
	return Compose(L1, L2, new_buf)
end

function Compose(L1::AbstractOperator, L2::AbstractOperator, buf::AbstractArray)
	return Compose((L2, L1), (buf,))
end

function Compose(L1::Compose, L2::AbstractOperator, buf::AbstractArray)
	return Compose((L2, L1.A...), (buf, L1.buf...))
end

function Compose(L1::AbstractOperator, L2::Compose, buf::AbstractArray)
	return Compose((L2.A..., L1), (L2.buf..., buf))
end

function Compose(L1::Compose, L2::Compose, buf::AbstractArray)
	return Compose((L2.A..., L1.A...), (L2.buf..., buf, L1.buf...))
end

#special cases
Compose(::Eye, L2::AbstractOperator) = L2
Compose(L1::AbstractOperator, ::Eye) = L1
Compose(L1::Eye, ::Eye) = L1

function Scale(coeff, L::Compose; threaded=default_should_thread(L))
	if coeff == 1
		return L
	end
	combinable = filter(A -> A isa Union{Scale,DiagOp,MatrixOp,LMatrixOp}, L.A)
	if all(is_linear.(L.A)) && !isempty(combinable)
		op_to_combine = argmin(op -> prod(size(op, 1)), combinable)
		i = findfirst(==(op_to_combine), L.A)
		A = ()
		if i > 1
			A = (L.A[1:(i - 1)]...,)
		end
		A = (A..., Scale(coeff, L.A[i]; threaded),)
		if i < length(L.A)
			A = (A..., L.A[(i + 1):end]...)
		end
		return Compose(A, L.buf)
	else
		return Scale(coeff, conj(coeff), L; threaded)
	end
end

# Mappings

@generated function mul!(y::C, L::Compose{N,M,T1,T2}, b::D) where {N,M,T1,T2,C,D}
	ex = :(mul!(L.buf[1], L.A[1], b))
	for i in 2:M
		ex = quote
			$ex
			mul!(L.buf[$i], L.A[$i], L.buf[$(i - 1)])
		end
	end
	ex = quote
		$ex
		mul!(y, L.A[N], L.buf[M])
		return y
	end
end

@generated function mul!(
	y::D, L::AdjointOperator{Compose{N,M,T1,T2}}, b::C
) where {N,M,T1,T2,C,D}
	ex = :(mul!(L.A.buf[M], L.A.A[N]', b))
	for i in M:-1:2
		ex = quote
			$ex
			mul!(L.A.buf[$(i - 1)], L.A.A[$i]', L.A.buf[$i])
		end
	end
	ex = quote
		$ex
		mul!(y, L.A.A[1]', L.A.buf[1])
		return y
	end
end

has_optimized_normalop(L::Compose) = has_optimized_normalop(L.A[end])
function get_normal_op(L::Compose)
	if has_optimized_normalop(L.A[end])
		combined = get_normal_op(L.A[end])
		ops = (L.A[1:(end - 1)]..., combined, reverse(adjoint.(L.A[1:(end - 1)]))...)
		bufs = (L.buf..., copy(L.buf[end]), reverse(L.buf[1:(end - 1)])...)
	else
		ops = (L.A..., reverse(adjoint.(L.A))...)
		bufs = (L.buf..., allocate_in_codomain(L), reverse(L.buf)...)
	end
	return Compose(ops, bufs)
end

# Properties

Base.:(==)(L1::Compose{N,M,L,T}, L2::Compose{N,M,L,T}) where {N,M,L,T} = all(L1.A .== L2.A)
size(L::Compose) = (size(L.A[end], 1), size(L.A[1], 2))

fun_name(L::Compose) = length(L.A) == 2 ? fun_name(L.A[2]) * "*" * fun_name(L.A[1]) : "Π"

domain_type(L::Compose) = domain_type(L.A[1])
codomain_type(L::Compose) = codomain_type(L.A[end])
domain_storage_type(L::Compose) = domain_storage_type(L.A[1])
codomain_storage_type(L::Compose) = codomain_storage_type(L.A[end])
is_thread_safe(::Compose) = false

is_linear(L::Compose) = all(is_linear.(L.A))
function is_diagonal(L::Compose)
	return  all(is_diagonal.(L.A[is_sliced(L) ? (2:end) : (1:end)]))
end
is_invertible(L::Compose) = all(is_invertible.(L.A))
is_AAc_diagonal(L::Compose) = all(is_AAc_diagonal.(L.A))
is_AcA_diagonal(L::Compose) = all(is_AcA_diagonal.(L.A))

is_sliced(L::Compose) = is_sliced(L.A[1])
get_slicing_expr(L::Compose) = get_slicing_expr(L.A[1])
get_slicing_mask(L::Compose) = get_slicing_mask(L.A[1])
function remove_slicing(L::Compose)
	if L.A[1] isa GetIndex
		return if length(L.A) == 2
			L.A[2]
		else
			Compose(tuple(L.A[2:end]...), tuple(L.buf[2:end]...))
		end
	elseif is_sliced(L.A[1])
		new_first = AbstractOperators.remove_slicing(L.A[1])
		if is_eye(new_first)
			return if length(L.A) == 2
				L.A[2]
			else
				Compose(tuple(L.A[2:end]...), tuple(L.buf[2:end]...))
			end
		else
			A = (new_first, L.A[2:end]...)
			return Compose(A, L.buf)
		end
	else
		throw(ArgumentError("First operator is not a GetIndex"))
	end
end

diag(L::Compose) = is_sliced(L) ? diag(L.A[2]) : prod(diag.(L.A))
function diag_AAc(L::Compose)
	return if is_AAc_diagonal(L)
		diag_AAc(L.A[2])
	else
		error("is_AAc_diagonal( $(typeof(L) ) ) == false")
	end
end

# utils
function permute(C::Compose, p::AbstractVector{Int})
	i = findfirst(x -> ndoms(x, 2) > 1, C.A)
	P = permute(C.A[i], p)
	AA = (C.A[1:(i - 1)]..., P, C.A[(i + 1):end]...)
	return Compose(AA, C.buf)
end

remove_displacement(C::Compose) = Compose(remove_displacement.(C.A), C.buf)

get_operators(C::Compose) = C.A
