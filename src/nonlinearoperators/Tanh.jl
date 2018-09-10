export Tanh

"""
`Tanh([domainType=Float64::Type,] dim_in::Tuple)`

Creates an hyperbolic tangent non-linear operator with input dimensions `dim_in`:
```math
\\text{tanh} ( \\mathbf{x} ).
```

"""
struct Tanh{T,N} <: NonLinearOperator
	dim::NTuple{N,Int}
end

function Tanh(DomainType::Type, DomainDim::NTuple{N,Int}) where {N} 
	Tanh{DomainType,N}(DomainDim)
end

Tanh(DomainDim::NTuple{N,Int}) where {N} = Tanh{Float64,N}(DomainDim)
Tanh(DomainDim::Vararg{Int}) = Tanh{Float64,length(DomainDim)}(DomainDim)

function mul!(y::AbstractArray{T,N}, L::Tanh{T,N}, x::AbstractArray{T,N}) where {T,N}
	y .= tanh.(x)
end

function mul!(y::AbstractArray, 
              J::AdjointOperator{Jacobian{A,TT}}, 
              b::AbstractArray) where {T,N, A<: Tanh{T,N}, TT <: AbstractArray{T,N}}
    L = J.A
    y .= conj.(sech.(L.x).^2).*b
end

fun_name(L::Tanh) = "tanh"

size(L::Tanh) = (L.dim, L.dim)

domainType(L::Tanh{T,N}) where {T,N} = T
codomainType(L::Tanh{T,N}) where {T,N} = T
