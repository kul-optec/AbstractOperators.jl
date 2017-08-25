export Sigmoid

"""
`Sigmoid([domainType=Float64::Type,] dim_in::Tuple, γ = 100.)`

Creates the sigmoid non-linear operator with input dimensions `dim_in`.

```math
\sigma(\mathbf{x}) = \frac{1}{1+e^{-\gamma \mathbf{x} } }
```

"""
immutable Sigmoid{T,N,G<:Real} <: NonLinearOperator
	dim::NTuple{N,Int}
	gamma::G
end

function Sigmoid{N, G <: Real}(DomainType::Type, DomainDim::NTuple{N,Int}, gamma::G=100.)  
	Sigmoid{DomainType,N,G}(DomainDim,gamma)
end

Sigmoid{N,G}(DomainDim::NTuple{N,Int}, gamma::G=100) = Sigmoid{Float64,N,G}(DomainDim,gamma)

function A_mul_B!{T,N,G}(y::AbstractArray{T,N}, L::Sigmoid{T,N,G}, x::AbstractArray{T,N})
	y .= (1.+exp.(-L.gamma.*x)).^(-1)
end


function Ac_mul_B!{T,N,G, A<: Sigmoid{T,N,G}}(y::AbstractArray{T,N}, L::Jacobian{A}, b::AbstractArray{T,N})
	y .= exp.(-L.A.gamma.*L.x)
	y ./= (1.+y).^2 
	y .= conj.(L.A.gamma.*y)
	y .*= b
end

fun_name(L::Sigmoid) = "σ"

size(L::Sigmoid) = (L.dim, L.dim)

domainType{T,N,D}(L::Sigmoid{T,N,D}) = T
codomainType{T,N,D}(L::Sigmoid{T,N,D}) = T
