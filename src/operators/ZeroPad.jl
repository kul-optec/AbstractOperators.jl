export ZeroPad

immutable ZeroPad{T,N} <: LinearOperator
	dim_in::NTuple{N,Int}
	zp::NTuple{N,Int}
end

# Constructors
#standard constructor
function ZeroPad{N,M}(domainType::Type, dim_in::NTuple{N,Int}, zp::NTuple{M,Int})
	M != N && error("dim_in and zp must have the same length")
	any([zp...].<0) && error("zero padding cannot be negative")
	N > 3 && error("currently ZeroPad not implemented for Arrays with ndims > 3")
	ZeroPad{domainType,N}(dim_in,zp)
end

ZeroPad{N}(dim_in::Tuple, zp::NTuple{N,Int}) = ZeroPad(Float64, dim_in, zp)
ZeroPad{N}(domainType::Type, dim_in::Tuple, zp::Vararg{Int,N}) = ZeroPad(domainType, dim_in, zp)
ZeroPad{N}(dim_in::Tuple, zp::Vararg{Int,N}) = ZeroPad(Float64, dim_in, zp)
ZeroPad{N}(x::AbstractArray, zp::NTuple{N,Int}) = ZeroPad(eltype(x), size(x), zp)
ZeroPad{N}(x::AbstractArray, zp::Vararg{Int,N}) = ZeroPad(eltype(x), size(x), zp)

# Mappings

function A_mul_B!{T}(y::AbstractVector{T}, L::ZeroPad{T,1}, b::AbstractVector{T})
	for i in eachindex(y)
		y[i] = i <= length(b) ? b[i] : 0.
	end
end

function Ac_mul_B!{T}(y::AbstractVector{T}, L::ZeroPad{T,1}, b::AbstractVector{T})
	for i in eachindex(y)
		y[i] = b[i]
	end
end

function A_mul_B!{T}(y::AbstractArray{T,2}, L::ZeroPad{T,2}, b::AbstractArray{T,2})
	for l = 1:size(y,1), m = 1:size(y,2)
		y[l,m] = l <= size(b,1) && m <= size(b,2) ? b[l,m] : 0.
	end
end

function Ac_mul_B!{T}(y::AbstractArray{T,2}, L::ZeroPad{T,2}, b::AbstractArray{T,2})
	for l = 1:size(y,1), m = 1:size(y,2)
		y[l,m] = b[l,m]
	end
end

function A_mul_B!{T}(y::AbstractArray{T,3}, L::ZeroPad{T,3}, b::AbstractArray{T,3})
	for l = 1:size(y,1), m = 1:size(y,2), n = 1:size(y,3)
		y[l,m,n] = l <= size(b,1) && m <= size(b,2) && n <= size(b,3) ? b[l,m,n] : 0.
	end
end

function Ac_mul_B!{T}(y::AbstractArray{T,3}, L::ZeroPad{T,3}, b::AbstractArray{T,3})
	for l = 1:size(y,1), m = 1:size(y,2), n = 1:size(y,3)
		y[l,m,n] = b[l,m,n]
	end
end

# Properties

domainType{T}(L::ZeroPad{T}) = T
codomainType{T}(L::ZeroPad{T}) = T

size(L::ZeroPad) = L.dim_in .+ L.zp, L.dim_in

fun_name(L::ZeroPad)       = "[I;0]"
is_AcA_diagonal(L::ZeroPad) = true
diag_AcA(L::ZeroPad) = 1

is_full_column_rank(L::ZeroPad) = true
