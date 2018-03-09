export Sin

"""
`Sin([domainType=Float64::Type,] dim_in::Tuple)`

Creates a sinusoid non-linear operator with input dimensions `dim_in`:
```math
\\sin( \\mathbf{x} ).
```

"""
struct Sin{T,N} <: NonLinearOperator
	dim::NTuple{N,Int}
end

function Sin(DomainType::Type, DomainDim::NTuple{N,Int}) where {N} 
	Sin{DomainType,N}(DomainDim)
end

Sin(DomainDim::NTuple{N,Int}) where {N} = Sin{Float64,N}(DomainDim)

function A_mul_B!(y::AbstractArray{T,N}, L::Sin{T,N}, x::AbstractArray{T,N}) where {T,N}
	y .= sin.(x)
end

function Ac_mul_B!(y::AbstractArray{T,N}, 
		   L::Jacobian{A}, 
		   b::AbstractArray{T,N}) where {T,N, A<: Sin{T,N}}
    y .= conj.(cos.(L.x)).*b
end

fun_name(L::Sin) = "sin"

size(L::Sin) = (L.dim, L.dim)

domainType(L::Sin{T,N}) where {T,N} = T
codomainType(L::Sin{T,N}) where {T,N} = T
