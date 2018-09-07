export Sigmoid

"""
`Sigmoid([domainType=Float64::Type,] dim_in::Tuple, γ = 1.)`

Creates the sigmoid non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\frac{1}{1+e^{-\\gamma \\mathbf{x} } }
```

"""
struct Sigmoid{T,N,G<:Real} <: NonLinearOperator
	dim::NTuple{N,Int}
	gamma::G
end

function Sigmoid(DomainType::Type, DomainDim::NTuple{N,Int}, gamma::G=1.) where {N, G <: Real} 
	Sigmoid{DomainType,N,G}(DomainDim,gamma)
end

Sigmoid(DomainDim::NTuple{N,Int}, gamma::G=1.) where {N,G} = Sigmoid{Float64,N,G}(DomainDim,gamma)

function A_mul_B!(y::AbstractArray{T,N}, L::Sigmoid{T,N,G}, x::AbstractArray{T,N}) where {T,N,G}
	y .= (1 .+exp.(-L.gamma.*x)).^(-1)
end


function Ac_mul_B!(y::AbstractArray{T,N}, 
		   L::Jacobian{A}, 
		   b::AbstractArray{T,N}) where {T,N,G, A<: Sigmoid{T,N,G}}
	y .= exp.(-L.A.gamma.*L.x)
	y ./= (1 .+y).^2 
	y .= conj.(L.A.gamma.*y)
	y .*= b
end

fun_name(L::Sigmoid) = "σ"

size(L::Sigmoid) = (L.dim, L.dim)

domainType(L::Sigmoid{T,N,D}) where {T,N,D} = T
codomainType(L::Sigmoid{T,N,D}) where {T,N,D} = T
