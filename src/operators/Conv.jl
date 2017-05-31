export Conv

immutable Conv{T,H <: AbstractVector{T}} <: LinearOperator
	dim_in::Tuple{Int}
	h::H
end

# Constructors

###standard constructor
function Conv{H<:AbstractVector, N}(DomainType::Type, DomainDim::NTuple{N,Int},  h::H) 
	eltype(h) != DomainType && error("eltype(h) is $(eltype(h)), should be $(DomainType)")
	N != 1 && error("Conv treats only SISO, check Filt and MIMOFilt for MIMO")
	Conv{DomainType,H}(DomainDim,h)
end
Conv{H}(x::H, h::H) = Conv(eltype(x), size(x), h)

# Mappings

function A_mul_B!{T,H}(y::H,A::Conv{T,H},b::H)
		y .= conv(A.h,b)
end

function Ac_mul_B!{T,H}(y::H,A::Conv{T,H},b::H)
		y .= xcorr(b,A.h)[size(A,1)[1]:end-length(A.h)+1]
end

# Properties

domainType{T}(L::Conv{T}) = T
codomainType{T}(L::Conv{T}) = T

size(L::Conv) = (L.dim_in[1]+length(L.h)-1,), L.dim_in

fun_name(A::Conv)  = "★"
