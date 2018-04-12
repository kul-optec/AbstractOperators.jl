export Atan

"""
`Atan([domainType=Float64::Type,] dim_in::Tuple)`

Creates an inverse tangent non-linear operator with input dimensions `dim_in`:
```math
\\text{atan} ( \\mathbf{x} ).
```

"""
struct Atan{T,N} <: NonLinearOperator
	dim::NTuple{N,Int}
end

function Atan(DomainType::Type, DomainDim::NTuple{N,Int}) where {N} 
	Atan{DomainType,N}(DomainDim)
end

Atan(DomainDim::NTuple{N,Int}) where {N} = Atan{Float64,N}(DomainDim)
Atan(DomainDim::Vararg{Int}) = Atan{Float64,length(DomainDim)}(DomainDim)

function A_mul_B!(y::AbstractArray{T,N}, L::Atan{T,N}, x::AbstractArray{T,N}) where {T,N}
	y .= atan.(x)
end

function Ac_mul_B!(y::AbstractArray{T,N}, 
		   L::Jacobian{A}, 
		   b::AbstractArray{T,N}) where {T,N, A<: Atan{T,N}}
    y .= conj.(1.0./(1.0.+ L.x.^2)).*b
end

fun_name(L::Atan) = "atan"

size(L::Atan) = (L.dim, L.dim)

domainType(L::Atan{T,N}) where {T,N} = T
codomainType(L::Atan{T,N}) where {T,N} = T
