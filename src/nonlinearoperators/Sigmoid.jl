export Sigmoid

"""
	Sigmoid([domainType=Float64::Type,] dim_in::Tuple, γ = 1.)

Creates the sigmoid non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\frac{1}{1+e^{-\\gamma \\mathbf{x} } }
```

"""
struct Sigmoid{T,N,G<:Real} <: NonLinearOperator
	dim::NTuple{N,Int}
	gamma::G
end

function Sigmoid(DomainType::Type, DomainDim::NTuple{N,Int}, gamma::G=1.0) where {N,G<:Real}
	return Sigmoid{DomainType,N,G}(DomainDim, gamma)
end

function Sigmoid(DomainDim::NTuple{N,Int}, gamma::G=1.0) where {N,G}
	return Sigmoid{Float64,N,G}(DomainDim, gamma)
end

function mul!(y::AbstractArray{T,N}, L::Sigmoid{T,N,G}, x::AbstractArray{T,N}) where {T,N,G}
	return y .= (1 .+ exp.(-L.gamma .* x)) .^ (-1)
end

function mul!(
	y::AbstractArray, J::AdjointOperator{Jacobian{A,TT}}, b::AbstractArray
) where {T,N,G,A<:Sigmoid{T,N,G},TT<:AbstractArray{T,N}}
	L = J.A
	y .= exp.(-L.A.gamma .* L.x)
	y ./= (1 .+ y) .^ 2
	y .= conj.(L.A.gamma .* y)
	return y .*= b
end

fun_name(L::Sigmoid) = "σ"

size(L::Sigmoid) = (L.dim, L.dim)

domainType(::Sigmoid{T,N,D}) where {T,N,D} = T
codomainType(::Sigmoid{T,N,D}) where {T,N,D} = T
