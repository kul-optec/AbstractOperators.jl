export GetIndex

immutable GetIndex{N,M,T<:Tuple} <: LinearOperator
	domainType::Type
	dim_out::NTuple{N,Int}
	dim_in::NTuple{M,Int}
	idx::T
end

# Constructors
# default 
function GetIndex{M,T<:Tuple}(domainType::Type,dim_in::NTuple{M,Int},idx::T)
	length(idx) > M && error("cannot slice object of dimension $dim_in with $idx")
	dim_out = get_dim_out(dim_in,idx...)
	if dim_out == dim_in
		return Eye(domainType,dim_in)
	else
		return GetIndex{length(dim_out),M,T}(domainType,dim_out,dim_in,idx)
	end
end

GetIndex(domainType::Type,dim_in::Tuple, idx...) = GetIndex(domainType, dim_in, idx)
GetIndex(dim_in::Tuple, idx...) = GetIndex(Float64, dim_in, idx)
GetIndex(dim_in::Tuple, idx::Tuple) = GetIndex(Float64, dim_in, idx)
GetIndex(x::AbstractArray, idx::Tuple) = GetIndex(eltype(x), size(x), idx)

# Mappings

function A_mul_B!{T1,N,M,T2}(y::Array{T1,N},L::GetIndex{N,M,T2},b::Array{T1,M})
	y .= view(b,L.idx...)
end

function Ac_mul_B!{T1,N,M,T2}(y::Array{T1,M},L::GetIndex{N,M,T2},b::AbstractArray{T1,N})
	y .= 0.
	setindex!(y,b,L.idx...)
end

# Properties
diag_AAc(L::GetIndex) = 1.

domainType(L::GetIndex) = L.domainType
codomainType(L::GetIndex) = L.domainType

size(L::GetIndex) = (L.dim_out,L.dim_in)

fun_name(L::GetIndex) = "↓"

is_AAc_diagonal(L::GetIndex)   = true
is_full_row_rank(L::GetIndex)  = true


# Utils

get_idx(L::GetIndex) = L.idx

function get_dim_out(dim,args...)
	if length(args) != 1
		dim2 = [dim[1:length(args)]...]
		for i = 1:length(args)
			if args[i] != Colon() dim2[i] = length(args[i]) end
		end
		return tuple(dim2...)
	else
		if args[1] == Colon()
			return dim
		else
			return tuple(length(args[1]))
		end
	end
end
