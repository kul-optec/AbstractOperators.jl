export Xcorr

immutable Xcorr{T,H <:AbstractVector{T}} <: LinearOperator
	dim_in::Tuple{Int}
	h::H
end

# Constructors
function Xcorr{H<:AbstractVector, N}(DomainType::Type, DomainDim::NTuple{N,Int},  h::H) 
	eltype(h) != DomainType && error("eltype(h) is $(eltype(h)), should be $(DomainType)")
	N != 1 && error("Xcorr treats only SISO, check Filt and MIMOFilt for MIMO")
	Xcorr{DomainType,H}(DomainDim,h)
end
Xcorr{H}(x::H, h::H) = Xcorr(eltype(x), size(x), h)

# Mappings

function A_mul_B!{T,H}(y::H,A::Xcorr{T,H},b::H)
	y .= xcorr(b,A.h)
end

function Ac_mul_B!{T,H}(y::H,A::Xcorr{T,H},b::H)
	l =floor(Int64,size(A,1)[1]/2)
	idx = l+1:l+length(y)
	y .= conv(b,A.h)[idx]
end

# Properties

domainType{T}(L::Xcorr{T}) = T
codomainType{T}(L::Xcorr{T}) = T

#TODO find out a way to verify this, 
is_full_row_rank(L::Xcorr)    = true
is_full_column_rank(L::Xcorr) = true

size(L::Xcorr) = ( 2*max(L.dim_in[1], length(L.h))-1, ), L.dim_in

fun_name(A::Xcorr)  = "◎"
