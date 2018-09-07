export Conv

"""
`Conv([domainType=Float64::Type,] dim_in::Tuple, h::AbstractVector)`

`Conv(x::AbstractVector, h::AbstractVector)`

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractVector`, returns the convolution between `x` and `h`. Uses `conv` and hence FFT algorithm. 

"""
struct Conv{T,
	    H  <: AbstractVector{T},
	    Hc <: AbstractVector{Complex{T}},
	    } <: LinearOperator
	dim_in::Tuple{Int}
	h::H
	buf::H
	buf_c1::Hc
	buf_c2::Hc
	R::AbstractFFTs.Plan
	I::AbstractFFTs.Plan
end

# Constructors

###standard constructor
function Conv(DomainType::Type, dim_in::Tuple{Int},  h::H) where {H<:AbstractVector} 
	eltype(h) != DomainType && error("eltype(h) is $(eltype(h)), should be $(DomainType)")

	buf = zeros(dim_in[1]+length(h)-1)
	R = plan_rfft(buf)
	buf_c1 = zeros(Complex{eltype(h)}, div(dim_in[1]+length(h)-1,2)+1)
	buf_c2 = zeros(Complex{eltype(h)}, div(dim_in[1]+length(h)-1,2)+1)
	I = plan_irfft(buf_c1,dim_in[1]+length(h)-1)
	Conv{DomainType, H, typeof(buf_c1)}(dim_in,h,buf,buf_c1,buf_c2,R,I)
end

Conv(dim_in::NTuple{N,Int},  h::H) where {H<:AbstractVector, N} =  Conv(eltype(h), dim_in, h)
Conv(x::H, h::H) where {H} = Conv(eltype(x), size(x), h)

# Mappings

function mul!(y::H, A::Conv{T,H}, b::H) where {T, H}
	#y .= conv(A.h,b) #naive implementation
	for i in eachindex(A.buf)
		A.buf[i] = i <= length(A.h) ? A.h[i] : zero(T) 
	end
	mul!(A.buf_c1, A.R, A.buf)
	for i in eachindex(A.buf)
		A.buf[i] = i <= length(b) ? b[i] : zero(T) 
	end
	mul!(A.buf_c2, A.R, A.buf)
	A.buf_c2 .*= A.buf_c1
	mul!(y,A.I,A.buf_c2)

end

function mul!(y::H, L::AdjointOperator{C}, b::H) where {T, H, C <: Conv{T,H}}
	#y .= xcorr(b,L.A.h)[size(L.A,1)[1]:end-length(L.A.h)+1] #naive implementation
	for i in eachindex(L.A.buf)
		ii = length(L.A.buf)-i+1
		L.A.buf[ii] = i <= length(L.A.h) ? L.A.h[i] : zero(T) 
	end
	mul!(L.A.buf_c1, L.A.R, L.A.buf)
	for i in eachindex(L.A.buf)
		L.A.buf[i] = b[i] 
	end
	mul!(L.A.buf_c2, L.A.R, L.A.buf)
	L.A.buf_c2 .*= L.A.buf_c1
	mul!(L.A.buf,L.A.I,L.A.buf_c2)
	y[1] = L.A.buf[end]
	for i = 2:length(y)
		y[i] = L.A.buf[i-1]
	end
end

# Properties

domainType(L::Conv{T}) where {T} = T
codomainType(L::Conv{T}) where {T} = T

#TODO find out a way to verify this, 
is_full_row_rank(L::Conv)    = true
is_full_column_rank(L::Conv) = true

size(L::Conv) = (L.dim_in[1]+length(L.h)-1,), L.dim_in

fun_name(A::Conv)  = "★"
