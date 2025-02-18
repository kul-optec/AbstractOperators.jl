export BroadcastingDiagOp

"""
	BroadcastingDiagOp(domain_type::Type, dim_in::Tuple, w::AbstractArray)
	BroadcastingDiagOp(x::AbstractArray, w::AbstractArray)

Creates a `LinearOperator` which, when multiplied with an array `x`, returns the elementwise product `w.*x`,
broadcasting weights `w` over the singleton dimensions of `x`.

```jldoctest
julia> S = BroadcastingDiagOp(Float64, (2, 2, 1), reshape(1:2*2*3, 2, 2, 3))
╲╲╲ ℝ^(2, 2, 1) -> ℝ^(2, 2, 3)

julia> S*ones(2, 2, 1)
2×2×3 Array{Float64, 3}:
[:, :, 1] =
 1.0  3.0
 2.0  4.0

[:, :, 2] =
 5.0  7.0
 6.0  8.0

[:, :, 3] =
 9.0  11.0
 10.0  12.0

```
"""

struct BroadcastingDiagOp{N,D,C} <: LinearOperator
	dim_in::NTuple{N,Int}
	w::AbstractArray{C,N}
	buf::AbstractArray{C,N}
end

# Constructors

function BroadcastingDiagOp(
	domain_type::Type, dim_in::NTuple{N,Int}, w::AbstractArray{C,N}
) where {N,C}
	C2 = eltype(w) <: Complex ? complex(domain_type) : domain_type
	return BroadcastingDiagOp{N,domain_type,C2}(dim_in, w, similar(w))
end

function BroadcastingDiagOp(x::AbstractArray, w::AbstractArray)
	dim_in = size(x)
	@assert length(dim_in) == ndims(w)
	@assert(all(d -> dim_in[d] == 1 || dim_in[d] == size(w, d), eachindex(dim_in)))
	C = eltype(w) <: Complex ? complex(eltype(x)) : eltype(x)
	return BroadcastingDiagOp{length(dim_in),eltype(x),C}(dim_in, w, similar(w))
end

# Mappings

function mul!(
	y::AbstractArray{C,N}, L::BroadcastingDiagOp{N,D,C}, b::AbstractArray{D,N}
) where {N,D,C}
	return y .= L.w .* b
end

function mul!(
	y::AbstractArray{D,N},
	L::AdjointOperator{BroadcastingDiagOp{N,D,C}},
	b::AbstractArray{C,N},
) where {N,D<:Real,C}
	L.A.buf .= conj.(L.A.w) .* b
	return sum!(real, y, L.A.buf)
end

function mul!(
	y::AbstractArray{D,N},
	L::AdjointOperator{BroadcastingDiagOp{N,D,C}},
	b::AbstractArray{C,N},
) where {N,D<:Complex,C}
	L.A.buf .= conj.(L.A.w) .* b
	return sum!(y, L.A.buf)
end

# Properties

domainType(::BroadcastingDiagOp{N,D,C}) where {N,D,C} = D
codomainType(::BroadcastingDiagOp{N,D,C}) where {N,D,C} = C
is_thread_safe(::BroadcastingDiagOp) = false

size(L::BroadcastingDiagOp) = size(L.w), L.dim_in

fun_name(L::BroadcastingDiagOp) = "╲╲╲"

is_orthogonal(L::BroadcastingDiagOp) = all(≈(1), sum(abs2, L.w; dims=broadcasting_dims(L)))
is_invertible(L::BroadcastingDiagOp) = all(>(0), L.w)
is_full_row_rank(L::BroadcastingDiagOp) = false
is_full_column_rank(L::BroadcastingDiagOp) = all(>(0), L.w)

# Utils

function broadcasting_dims(L::BroadcastingDiagOp)
	return findall(d -> L.dim_in[d] != size(L.w, d), eachindex(L.dim_in))
end
