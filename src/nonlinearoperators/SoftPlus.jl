export SoftPlus

"""
`SoftPlus([domainType=Float64::Type,] dim_in::Tuple)`

Creates the softplus non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\log (1 + e^{x} )
```

"""
struct SoftPlus{T,N} <: NonLinearOperator
	dim::NTuple{N,Int}
end

function SoftPlus(DomainType::Type, DomainDim::NTuple{N,Int}) where {N} 
	SoftPlus{DomainType,N}(DomainDim)
end

SoftPlus(DomainDim::NTuple{N,Int}) where {N} = SoftPlus{Float64,N}(DomainDim)

function mul!(y::AbstractArray{T,N}, L::SoftPlus{T,N}, x::AbstractArray{T,N}) where {T,N}
	y .= log.(1 .+exp.(x))
end

function mul!(y::AbstractArray, 
              J::AdjointOperator{Jacobian{A,TT}}, 
              b::AbstractArray) where {T, N, A <: SoftPlus{T,N}, TT <: AbstractArray{T,N} }
    L = J.A
	y .= 1 ./(1 .+exp.(-L.x)).*b
end

fun_name(L::SoftPlus) = "σ"

size(L::SoftPlus) = (L.dim, L.dim)

domainType(L::SoftPlus{T,N}) where {T,N} = T
codomainType(L::SoftPlus{T,N}) where {T,N} = T
