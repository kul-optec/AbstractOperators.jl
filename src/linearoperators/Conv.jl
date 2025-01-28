export Conv

abstract type AbstractConv{T,N,H<:AbstractArray} <: LinearOperator end

"""
	Conv([domainType=Float64::Type,] dim_in::Tuple, h::AbstractVector)
	Conv(x::AbstractVector, h::AbstractVector)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractVector`, returns the convolution between `x` and `h`. Uses `conv` and hence FFT algorithm.

```jldoctest
julia> Conv((10,),randn(5))
★  ℝ^10 -> ℝ^14
	
```
"""
struct Conv{T,N,H<:AbstractArray{T,N},Hc} <: AbstractConv{T,N,H}
	dim_in::NTuple{N,Int}
	h::H
	buf::H
	buf_c1::Hc
	buf_c2::Hc
	R::AbstractFFTs.Plan
	I::AbstractFFTs.Plan
end

# Constructors

isTypeReal(::Type{T}) where {T} = T <: Real

###standard constructor
function Conv(domainType::Type, dim_in::NTuple{N,Int}, h::H) where {N,H<:AbstractArray}
	eltype(h) != domainType && error("eltype(h) is $(eltype(h)), should be $(domainType)")

	buf = similar(h, domainType, dim_in .+ size(h) .- 1)
	if isTypeReal(domainType)
		R = plan_rfft(buf)
		complex_type = Complex{domainType}
		buf_size = ntuple(d -> d == 1 ? size(buf, d) >> 1 + 1 : size(buf, d), Val(N))
		buf_c1 = similar(h, Complex{domainType}, buf_size)
		I = plan_irfft(buf_c1, size(buf, 1))
	else
		R = plan_fft(buf)
		buf_c1 = similar(buf)
		complex_type = domainType
		I = FFTW.plan_inv(R)
	end
	buf_c2 = similar(buf_c1)
	return Conv{domainType,N,H,typeof(buf_c1)}(dim_in, h, buf, buf_c1, buf_c2, R, I)
end

Conv(dim_in::NTuple{N,Int}, h::H) where {H<:AbstractVector,N} = Conv(eltype(h), dim_in, h)
Conv(x::H, h::H) where {H} = Conv(eltype(x), size(x), h)
Conv(dim_in::NTuple{N,Int}, h::H) where {H<:AbstractArray,N} = Conv(eltype(h), dim_in, h)
Conv(x::H, h::H) where {H<:AbstractArray} = Conv(eltype(x), size(x), h)

# Mappings
function mul!(
	y::AbstractArray{T,N}, A::AbstractConv{T,N}, b::AbstractArray{T,N}
) where {T,N}
	#y .= conv(A.h,b) #naive implementation
	fill!(A.buf, zero(T))
	A.buf[CartesianIndices(A.h)] .= A.h
	mul!(A.buf_c1, A.R, A.buf)
	fill!(A.buf, zero(T))
	A.buf[CartesianIndices(b)] .= b
	mul!(A.buf_c2, A.R, A.buf)
	A.buf_c2 .*= A.buf_c1
	return mul!(y, A.I, A.buf_c2)
end

function mul!(
	y::AbstractArray{T,N}, L::AdjointOperator{C}, b::AbstractArray{T,N}
) where {T,N,C<:AbstractConv{T,N}}
	#y .= xcorr(b,L.A.h)[size(L.A.h,1)[1]:end-length(L.A.h)+1] #naive implementation
	fill!(L.A.buf, zero(T))
	L.A.buf[CartesianIndices(L.A.h)] .= L.A.h
	mul!(L.A.buf_c1, L.A.R, L.A.buf)
	fill!(L.A.buf, zero(T))
	L.A.buf[CartesianIndices(b)] .= b
	mul!(L.A.buf_c2, L.A.R, L.A.buf)
	L.A.buf_c2 .*= conj.(L.A.buf_c1)
	mul!(L.A.buf, L.A.I, L.A.buf_c2)
	return y .= L.A.buf[CartesianIndices(y)]
end

# Properties

domainType(::AbstractConv{T}) where {T} = T
codomainType(::AbstractConv{T}) where {T} = T
domain_storage_type(::AbstractConv{T,N,H}) where {T,N,H} = H
codomain_storage_type(::AbstractConv{T,N,H}) where {T,N,H} = H
is_thread_safe(::Conv) = false

#TODO find out a way to verify this,
is_full_row_rank(L::Conv) = true
is_full_column_rank(L::Conv) = true
is_full_row_rank(::AbstractConv) = true
is_full_column_rank(::AbstractConv) = true

size(L::Conv) = (L.dim_in[1] + length(L.h) - 1,), L.dim_in
size(L::AbstractConv) = (L.dim_in[1] + length(L.h) - 1,), L.dim_in

fun_name(A::Conv) = "★"
fun_name(::AbstractConv) = "★"
