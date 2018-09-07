export Xcorr
#TODO make more efficient

"""
`Xcorr([domainType=Float64::Type,] dim_in::Tuple, h::AbstractVector)`

`Xcorr(x::AbstractVector, h::AbstractVector)`

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractVector`, returns the cross correlation between `x` and `h`. Uses `xcross`. 

"""
struct Xcorr{T,H <:AbstractVector{T}} <: LinearOperator
	dim_in::Tuple{Int}
	h::H
end

# Constructors
function Xcorr(DomainType::Type, DomainDim::NTuple{N,Int},  h::H) where {H<:AbstractVector, N} 
	eltype(h) != DomainType && error("eltype(h) is $(eltype(h)), should be $(DomainType)")
	N != 1 && error("Xcorr treats only SISO, check Filt and MIMOFilt for MIMO")
	Xcorr{DomainType,H}(DomainDim,h)
end
Xcorr(x::H, h::H) where {H} = Xcorr(eltype(x), size(x), h)

# Mappings

function mul!(y::H,A::Xcorr{T,H},b::H) where {T,H}
	y .= xcorr(b,A.h)
end

function mul!(y::H,L::AdjointOperator{Xcorr{T,H}},b::H) where {T,H}
    A = L.A
	l =floor(Int64,size(A,1)[1]/2)
	idx = l+1:l+length(y)
	y .= conv(b,A.h)[idx]
end

# Properties

domainType(L::Xcorr{T}) where {T} = T
codomainType(L::Xcorr{T}) where {T} = T

#TODO find out a way to verify this, 
is_full_row_rank(L::Xcorr)    = true
is_full_column_rank(L::Xcorr) = true

size(L::Xcorr) = ( 2*max(L.dim_in[1], length(L.h))-1, ), L.dim_in

fun_name(A::Xcorr)  = "◎"
