export SoftMax

"""
`SoftMax([domainType=Float64::Type,] dim_in::Tuple)`

Creates the softmax non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\frac{e^{\\mathbf{x} }}{ \\sum e^{\\mathbf{x} } }
```

"""
struct SoftMax{T,N} <: NonLinearOperator
	dim::NTuple{N,Int}
	buf::AbstractArray{T,N}
end

function SoftMax(x::AbstractArray{T,N}) where {T,N}
    SoftMax{N,T}(size(x),similar(x))
end

function SoftMax(DomainType::Type, DomainDim::NTuple{N,Int}) where {N}
	SoftMax{DomainType,N}(DomainDim,zeros(DomainType,DomainDim))
end

SoftMax(DomainDim::NTuple{N,Int}) where {N} = SoftMax(Float64,DomainDim)

function mul!(y::AbstractArray{T,N}, L::SoftMax{T,N}, x::AbstractArray{T,N}) where {T,N}
	y .= exp.(x.-maximum(x))
	y ./= sum(y)
end

function mul!(y::AbstractArray,
              J::AdjointOperator{Jacobian{A,TT}},
              b::AbstractArray) where {T, N, A<: SoftMax{T,N}, TT <: AbstractArray{T,N} }
    L = J.A
	fill!(y,zero(T))
	L.A.buf .= exp.(L.x.-maximum(L.x))
	L.A.buf ./= sum(L.A.buf)
	for i in eachindex(y)
		y[i] = -L.A.buf[i]*dot(L.A.buf,b)
		y[i] += L.A.buf[i]*b[i]
	end
	return y
end

fun_name(L::SoftMax) = "σ"

size(L::SoftMax) = (L.dim, L.dim)

domainType(L::SoftMax{T,N}) where {T,N} = T
codomainType(L::SoftMax{T,N}) where {T,N} = T
