export BroadCast

abstract type AbstractBroadCast{T,N,M,Threaded} <: AbstractOperator end

struct NoOperatorBroadCast{T,N,M,Threaded,S} <: AbstractBroadCast{T,N,M,Threaded}
	dim_in::NTuple{N,Int}
	reshaped_dim_in::NTuple{M,Int}
	dim_out::NTuple{M,Int}
	function NoOperatorBroadCast(
		T::Type, S, dim_in::NTuple{N,Int}, reshaped_dim_in::NTuple{M,Int}, dim_out::NTuple{M,Int}; threaded::Bool=true
	) where {N,M}
		Base.Broadcast.check_broadcast_shape(dim_out, reshaped_dim_in)
		compact = all(reshaped_dim_in[d] == dim_out[d] for d in 1:N)
		threaded = threaded && Threads.nthreads() > 1 && compact && prod(dim_in) * sizeof(T) > 2^16
		return new{T,N,M,threaded,S}(dim_in, reshaped_dim_in, dim_out)
	end
end

struct OperatorBroadCast{T,N,M,Threaded,Compact,Imask,L,C,D,K} <: AbstractBroadCast{T,N,M,Threaded}
	A::L
	dim_out::NTuple{M,Int}
	idxs::CartesianIndices{K}
	bufC::C
	bufD::D
	function OperatorBroadCast(
		A, dim_out::NTuple{M,Int}; threaded::Bool=true
	) where {M}
		Base.Broadcast.check_broadcast_shape(dim_out, size(A, 1))
		threaded = threaded && Threads.nthreads() > 1
		N = ndims(A, 1)
		T = codomain_type(A)
		dim_in = size(A, 1)
		Imask = Tuple(d ≤ N && (dim_out[d] == dim_in[d]) for d in 1:M)
		broadcast_dims = Tuple(Imask[d] ? 1 : dim_out[d] for d in 1:length(dim_out))
		idxs = CartesianIndices(broadcast_dims)
		compact = all(Imask[1:N])
		bufC = allocate_in_codomain(A)
		if threaded
			bufD = [allocate_in_domain(A) for _ in 1:Threads.nthreads()]
			A = [i == 1 ? A : copy_op(A) for i in 1:Threads.nthreads()]
		else
			bufD = allocate_in_domain(A)
		end
		L = typeof(A)
		C = typeof(bufC)
		D = typeof(bufD)
		K = length(broadcast_dims)
		return new{T,N,M,threaded,compact,Imask,L,C,D,K}(A, dim_out, idxs, bufC, bufD)
	end
end

# Constructors

"""
	BroadCast(A::AbstractOperator, dim_out...)

BroadCast the codomain dimensions of an `AbstractOperator`.

```jldoctest
julia> A = Eye(2)
I  ℝ^2 -> ℝ^2

julia> B = BroadCast(A,(2,3))
.I  ℝ^2 -> ℝ^(2, 3)

julia> B*[1.;2.]
2×3 Matrix{Float64}:
 1.0  1.0  1.0
 2.0  2.0  2.0
	
```
"""
function BroadCast(
	A::L, dim_out::NTuple{N,Int}; threaded::Bool=true
) where {N,L<:AbstractOperator}
	if length(dim_out) < ndims(A, 1)
		error("dim_out must have at least as many dimensions as the codomain of A")
	end
	if dim_out == size(A, 1)
		return A
	elseif is_eye(A)
		dim_in = size(A, 2)
		reshaped_dim_in = ntuple(d -> d <= ndims(A, 1) ? size(A, 1)[d] : 1, length(dim_out))
		return NoOperatorBroadCast(domain_type(A), domain_storage_type(A), dim_in, reshaped_dim_in, dim_out; threaded)
	else
		return OperatorBroadCast(A, dim_out; threaded)
	end
end

# Mappings

# Simple threaded broadcast (all broadcasting dimensions at the end)
function tbroadcast!(y, x)
	_x = vec(x)
	_y = reshape(y, length(x), :)
	@batch for k in axes(_y, 2)
		_y[:, k] .= _x
	end
	_y
end

# NoOperatorBroadCast
function mul!(y, A::NoOperatorBroadCast{T,N,M,false}, b) where {T,N,M}
	check(y, A, b)
	b = reshape(b, A.reshaped_dim_in)
	return y .= b # non-threaded broadcasting
end

function mul!(y, A::NoOperatorBroadCast{T,N,M,true}, b) where {T,N,M}
	check(y, A, b)
	return tbroadcast!(y, b) # threaded broadcasting, handles reshaping (threading is only enabled when compact)
end

function mul!(y, A::AdjointOperator{<:NoOperatorBroadCast}, b) # there is no threaded option here
	check(y, A, b)
	y = reshape(y, A.A.reshaped_dim_in)
	return sum!(y, b)
end

# OperatorBroadCast

function mul!(y, R::OperatorBroadCast, b) # Non-threaded
	check(y, R, b)
	mul!(R.bufC, R.A, b)
	return y .= R.bufC # non-threaded broadcasting
end

function mul!(y, R::OperatorBroadCast{T,N,M,true,Compact}, b) where {T,N,M,Compact} # Threaded
	check(y, R, b)
	mul!(R.bufC, R.A[1], b)
	if Compact
		return tbroadcast!(y, R.bufC) # threaded broadcasting
	else
		return y .= R.bufC # non-threaded broadcasting
	end
end

function mul!(y, A::AdjointOperator{<:OperatorBroadCast{T,N,M,false}}, b) where {T,N,M} # Non-threaded
	check(y, A, b)
	R = A.A
	for idx in R.idxs
		b_slice = get_input_slice(R, idx, b)
		mul!(R.bufD, R.A', b_slice)
		if idx == first(R.idxs)
			y .= R.bufD
		else
			y .+= R.bufD
		end
	end
	return y
end

function mul!(y, A::AdjointOperator{<:OperatorBroadCast{T,N,M,true}}, b) where {T,N,M} # Threaded
	check(y, A, b)
	R = A.A
	fill!(y, 0)
	lock = ReentrantLock()
	thread_count = min(Threads.nthreads(), length(R.idxs))
	batch_size = length(R.idxs) / thread_count
	@threads for t in 1:thread_count
		idx_start = max(1, floor(Int, (t - 1) * batch_size + 1))
		idx_end = min(length(R.idxs), floor(Int, t * batch_size))
		for i in idx_start:idx_end
			b_slice = get_input_slice(R, R.idxs[i], b)
			mul!(R.bufD[t], R.A[t]', b_slice)
			@lock lock y .+= R.bufD[t]
		end
	end
	return y
end

# Properties
function Base.:(==)(R1::NoOperatorBroadCast{T,N,M}, R2::NoOperatorBroadCast{T,N,M}) where {T,N,M}
	R1.dim_in == R2.dim_in && R1.dim_out == R2.dim_out
end
function Base.:(==)(
	R1::OperatorBroadCast{T,N,M}, R2::OperatorBroadCast{T,N,M}
) where {T,N,M}
	R1.A == R2.A && R1.dim_out == R2.dim_out
end

size(R::NoOperatorBroadCast) = (R.dim_out, R.dim_in)
size(R::OperatorBroadCast{T,N,M,false}) where {T,N,M} = (R.dim_out, size(R.A, 2))
size(R::OperatorBroadCast{T,N,M,true}) where {T,N,M} = (R.dim_out, size(R.A[1], 2))

domain_type(::NoOperatorBroadCast{T}) where {T} = T
domain_type(R::OperatorBroadCast{T,N,M,false}) where {T,N,M} = domain_type(R.A)
domain_type(R::OperatorBroadCast{T,N,M,true}) where {T,N,M} = domain_type(R.A[1])
codomain_type(::NoOperatorBroadCast{T}) where {T} = T
codomain_type(R::OperatorBroadCast{T,N,M,false}) where {T,N,M} = codomain_type(R.A)
codomain_type(R::OperatorBroadCast{T,N,M,true}) where {T,N,M} = codomain_type(R.A[1])
domain_storage_type(::NoOperatorBroadCast{T,N,M,Threaded,S}) where {T,N,M,Threaded,S} = S
domain_storage_type(R::OperatorBroadCast{T,N,M,false}) where {T,N,M} = domain_storage_type(R.A)
domain_storage_type(R::OperatorBroadCast{T,N,M,true}) where {T,N,M} = domain_storage_type(R.A[1])
codomain_storage_type(::NoOperatorBroadCast{T,N,M,Threaded,S}) where {T,N,M,Threaded,S} = S
codomain_storage_type(R::OperatorBroadCast{T,N,M,false}) where {T,N,M} = codomain_storage_type(R.A)
codomain_storage_type(R::OperatorBroadCast{T,N,M,true}) where {T,N,M} = codomain_storage_type(R.A[1])

is_thread_safe(::NoOperatorBroadCast) = true
is_thread_safe(::OperatorBroadCast) = false

is_linear(::NoOperatorBroadCast) = true
is_linear(R::OperatorBroadCast{T,N,M,false}) where {T,N,M} = is_linear(R.A)
is_linear(R::OperatorBroadCast{T,N,M,true}) where {T,N,M} = is_linear(R.A[1])
is_null(R::NoOperatorBroadCast) = false
is_null(R::OperatorBroadCast{T,N,M,false}) where {T,N,M} = is_null(R.A)
is_null(R::OperatorBroadCast{T,N,M,true}) where {T,N,M} = is_null(R.A[1])

fun_name(::NoOperatorBroadCast) = ".I"
fun_name(R::OperatorBroadCast{T,N,M,false}) where {T,N,M} = "." * fun_name(R.A)
fun_name(R::OperatorBroadCast{T,N,M,true}) where {T,N,M} = "." * fun_name(R.A[1])
remove_displacement(R::NoOperatorBroadCast) = R
function remove_displacement(R::OperatorBroadCast{T,N,M,false,Imask}) where {T,N,M,Imask}
	new_A = remove_displacement(R.A)
	return OperatorBroadCast(new_A, R.dim_out, threaded=false)
end
function remove_displacement(R::OperatorBroadCast{T,N,M,true,Imask}) where {T,N,M,Imask}
	new_A = remove_displacement(R.A[1])
	return OperatorBroadCast(new_A, R.dim_out, threaded=true)
end

has_fast_opnorm(::NoOperatorBroadCast) = true
has_fast_opnorm(::OperatorBroadCast{T,N,M,false}) where {T,N,M} = has_fast_opnorm(R.A)
has_fast_opnorm(::OperatorBroadCast{T,N,M,true}) where {T,N,M} = has_fast_opnorm(R.A[1])
function LinearAlgebra.opnorm(R::NoOperatorBroadCast{T,N,M}) where {T,N,M}
	real(T)(sqrt(prod(R.dim_out[d] for d in 1:M if R.dim_out[d] != R.reshaped_dim_in[d])))
end
LinearAlgebra.opnorm(R::OperatorBroadCast{T,N,M,false}) where {T,N,M} = LinearAlgebra.opnorm(R.A)
LinearAlgebra.opnorm(R::OperatorBroadCast{T,N,M,true}) where {T,N,M} = LinearAlgebra.opnorm(R.A[1])

# utils

function permute(R::OperatorBroadCast{T,N,M,false}, p::AbstractVector{Int}) where {T,N,M}
	return BroadCast(permute(R.A, p), R.dim_out; threaded=false)
end
function permute(R::OperatorBroadCast{T,N,M,true}, p::AbstractVector{Int}) where {T,N,M}
	return BroadCast([permute(A, p) for A in R.A], R.dim_out; threaded=true)
end

@generated function get_input_slice(
	::OperatorBroadCast{T,N,M,Threaded,Compact,Imask}, idx::CartesianIndex, b
) where {T,N,M,Threaded,Compact,Imask}
	return quote
		@ncall($M, view, b, d -> Imask[d] ? Colon() : idx[d])
	end
end
