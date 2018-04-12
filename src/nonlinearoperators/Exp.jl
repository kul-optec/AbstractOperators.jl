export Exp

"""
`Exp([domainType=Float64::Type,] dim_in::Tuple)`

Creates the exponential non-linear operator with input dimensions `dim_in`:
```math
e^{ \\mathbf{x} }.
```

"""
struct Exp{T,N} <: NonLinearOperator
	dim::NTuple{N,Int}
end

function Exp(DomainType::Type, DomainDim::NTuple{N,Int}) where {N} 
	Exp{DomainType,N}(DomainDim)
end

Exp(DomainDim::NTuple{N,Int}) where {N} = Exp{Float64,N}(DomainDim)
Exp(DomainDim::Vararg{Int}) = Exp{Float64,length(DomainDim)}(DomainDim)

function A_mul_B!(y::AbstractArray{T,N}, L::Exp{T,N}, x::AbstractArray{T,N}) where {T,N}
	y .= exp.(x)
end

function Ac_mul_B!(y::AbstractArray{T,N}, 
		   L::Jacobian{A}, 
		   b::AbstractArray{T,N}) where {T,N, A<: Exp{T,N}}
    y .= conj.(exp.(L.x)).*b
end

fun_name(L::Exp) = "e"

size(L::Exp) = (L.dim, L.dim)

domainType(L::Exp{T,N}) where {T,N} = T
codomainType(L::Exp{T,N}) where {T,N} = T
