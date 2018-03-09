export Cos

"""
`Cos([domainType=Float64::Type,] dim_in::Tuple)`

Creates a cosine non-linear operator with input dimensions `dim_in`:
```math
\\cos (\\mathbf{x} ).
```

"""
struct Cos{T,N} <: NonLinearOperator
	dim::NTuple{N,Int}
end

function Cos(DomainType::Type, DomainDim::NTuple{N,Int}) where {N} 
	Cos{DomainType,N}(DomainDim)
end

Cos(DomainDim::NTuple{N,Int}) where {N} = Cos{Float64,N}(DomainDim)

function A_mul_B!(y::AbstractArray{T,N}, L::Cos{T,N}, x::AbstractArray{T,N}) where {T,N}
	y .= cos.(x)
end

function Ac_mul_B!(y::AbstractArray{T,N}, 
		   L::Jacobian{A}, 
		   b::AbstractArray{T,N}) where {T,N, A<: Cos{T,N}}
    y .= -conj.(sin.(L.x)).*b
end

fun_name(L::Cos) = "cos"

size(L::Cos) = (L.dim, L.dim)

domainType(L::Cos{T,N}) where {T,N} = T
codomainType(L::Cos{T,N}) where {T,N} = T
