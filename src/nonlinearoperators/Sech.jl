export Sech

"""
`Sech([domainType=Float64::Type,] dim_in::Tuple)`

Creates an hyperbolic secant non-linear operator with input dimensions `dim_in`:
```math
\\text{sech} ( \\mathbf{x} ).
```

"""
struct Sech{T,N} <: NonLinearOperator
	dim::NTuple{N,Int}
end

function Sech(DomainType::Type, DomainDim::NTuple{N,Int}) where {N}
	Sech{DomainType,N}(DomainDim)
end

Sech(DomainDim::NTuple{N,Int}) where {N} = Sech{Float64,N}(DomainDim)
Sech(DomainDim::Vararg{Int}) = Sech{Float64,length(DomainDim)}(DomainDim)

function mul!(y::AbstractArray{T,N}, L::Sech{T,N}, x::AbstractArray{T,N}) where {T,N}
	y .= sech.(x)
end

function mul!(y::AbstractArray,
              J::AdjointOperator{Jacobian{A,TT}},
              b::AbstractArray) where {T,N, A<: Sech{T,N}, TT <: AbstractArray{T,N}}
    L = J.A
    y .= -conj.( tanh.(L.x) .* sech.(L.x) ).*b
end

fun_name(L::Sech) = "sech"

size(L::Sech) = (L.dim, L.dim)

domainType(L::Sech{T,N}) where {T,N} = T
codomainType(L::Sech{T,N}) where {T,N} = T
