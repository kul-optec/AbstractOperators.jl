export GetIndex

"""
	GetIndex([domainType=Float64::Type,] dim_in::Tuple, idx...)
	GetIndex(x::AbstractArray, idx::Tuple)

Creates a `LinearOperator` which, when multiplied with `x`, returns `x[idx]`.

```jldoctest
julia> x = collect(range(1,stop=10,length=10));

julia> G = GetIndex(Float64,(10,), 1:3)
↓  ℝ^10 -> ℝ^3

julia> G*x
3-element Vector{Float64}:
 1.0
 2.0
 3.0

julia> GetIndex(randn(10,20,30),(1:2,1:4))
↓  ℝ^(10, 20, 30) -> ℝ^(2, 4)
	
```
"""
struct GetIndex{N,M,T} <: LinearOperator
	domainType::Type
	dim_out::NTuple{N,Int}
	dim_in::NTuple{M,Int}
	idx::T
end

# Constructors
# default
function GetIndex(domainType::Type, dim_in::NTuple{M,Int}, idx::T) where {M,T<:Tuple}
	length(idx) > M && error("cannot slice object of dimension $dim_in with $idx")
	dim_out = get_dim_out(dim_in, idx...)
	if dim_out == dim_in
		return Eye(domainType, dim_in)
	else
		return GetIndex(domainType, dim_out, dim_in, idx)
	end
end

function GetIndex(domainType::Type, dim_in::NTuple{M,Int}, mask::T) where {M,T<:BitArray}
	size(mask) != dim_in && error("cannot slice object of dimension $dim_in with this bitmask")
	dim_out = (sum(mask),)
	if dim_out[1] == prod(dim_in)
		return reshape(Eye(domainType, dim_in), dim_out)
	else
		return GetIndex(domainType, dim_out, dim_in, mask)
	end
end

GetIndex(domainType::Type, dim_in::Tuple, idx...) = GetIndex(domainType, dim_in, idx)
GetIndex(dim_in::Tuple, idx...) = GetIndex(Float64, dim_in, idx)
GetIndex(dim_in::Tuple, idx::Tuple) = GetIndex(Float64, dim_in, idx)
GetIndex(x::AbstractArray, idx::Tuple) = GetIndex(eltype(x), size(x), idx)
GetIndex(dim_in::Tuple, mask::T) where {T<:BitArray} = GetIndex(Float64, dim_in, mask)
GetIndex(x::AbstractArray, mask::T) where {T<:BitArray} = GetIndex(eltype(x), size(x), mask)

# Mappings

function mul!(
	y::AbstractArray{T1,N}, L::GetIndex{N,M,T2}, b::AbstractArray{T1,M}
) where {T1,N,M,T2<:Tuple}
	return y .= view(b, L.idx...)
end

function mul!(
	y::AbstractArray{T1,N}, L::GetIndex{N,M,T2}, b::AbstractArray{T1,M}
) where {T1,N,M,T2<:BitArray}
	return y .= view(b, L.idx)
end

function mul!(
	y::AbstractArray{T1,M}, L::AdjointOperator{GetIndex{N,M,T2}}, b::AbstractArray{T1,N}
) where {T1,N,M,T2<:Tuple}
	fill!(y, 0.0)
	return setindex!(y, b, L.A.idx...)
end

function mul!(
	y::AbstractArray{T1,M}, L::AdjointOperator{GetIndex{N,M,T2}}, b::AbstractArray{T1,N}
) where {T1,N,M,T2<:BitArray}
	fill!(y, 0.0)
	return setindex!(y, b, L.A.idx)
end

struct NormalGetIndex{N,M,T} <: LinearOperator
	domainType::Type
	dim_in::NTuple{M,Int}
	idx::T
end

function mul!(
	y::AbstractArray{T1,M}, L::NormalGetIndex{N,M,T2}, b::AbstractArray{T1,M}
) where {T1,N,M,T2<:Tuple}
	fill!(y, 0.0)
	setindex!(y, view(b, L.idx), L.idx...)
end
function mul!(
	y::AbstractArray{T1,M}, L::NormalGetIndex{N,M,T2}, b::AbstractArray{T1,M}
) where {T1,N,M,T2<:BitArray}
	fill!(y, 0.0)
	setindex!(y, view(b, L.idx), L.idx)
end

# Properties
diag_AAc(L::GetIndex) = 1.0

domainType(L::GetIndex) = L.domainType
codomainType(L::GetIndex) = L.domainType
is_thread_safe(L::GetIndex) = true

size(L::GetIndex) = (L.dim_out, L.dim_in)

fun_name(L::GetIndex) = "↓"

is_AAc_diagonal(L::GetIndex) = true
is_full_row_rank(L::GetIndex) = true
is_sliced(L::GetIndex) = true
get_slicing_expr(L::GetIndex) = L.idx
get_slicing_mask(L::GetIndex{N,M,<:BitArray}) where {N,M} = L.idx
function get_slicing_mask(L::GetIndex{N,M,<:Tuple}) where {N,M}
	mask = falses(L.dim_in)
	mask[L.idx...] .= true
	return mask
end
remove_slicing(L::GetIndex) = Eye(L.domainType, L.dim_out)

has_optimized_normalop(L::GetIndex) = true
get_normal_op(L::GetIndex) = NormalGetIndex(L.domainType, L.dim_in, L.idx)

LinearAlgebra.opnorm(L::GetIndex) = one(real(domainType(L)))

# Utils

get_idx(L::GetIndex) = L.idx

function get_dim_out(dim, args...)
	if length(args) != 1
		dim2 = ()
		for i in eachindex(args)
			if args[i] != Colon()
				!(typeof(args[i]) <: Int) && (dim2 = (dim2..., length(args[i])))
			else
				dim2 = (dim2..., dim[i])
			end
		end
		return dim2
	else
		if args[1] == Colon()
			return dim
		else
			return tuple(length(args[1]))
		end
	end
end
