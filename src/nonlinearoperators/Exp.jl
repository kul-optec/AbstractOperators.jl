export Exp

"""
	Exp([domainType=Float64::Type,] dim_in::Tuple)

Creates the exponential non-linear operator with input dimensions `dim_in`:
```math
e^{ \\mathbf{x} }.
```

"""
struct Exp{T,N} <: NonLinearOperator
	dim::NTuple{N,Int}
end

function Exp(DomainType::Type, DomainDim::NTuple{N,Int}) where {N}
	return Exp{DomainType,N}(DomainDim)
end

Exp(DomainDim::NTuple{N,Int}) where {N} = Exp{Float64,N}(DomainDim)
Exp(DomainDim::Vararg{Int}) = Exp{Float64,length(DomainDim)}(DomainDim)

function mul!(y::AbstractArray{T,N}, L::Exp{T,N}, x::AbstractArray{T,N}) where {T,N}
	return y .= exp.(x)
end

function mul!(
	y::AbstractArray, J::AdjointOperator{Jacobian{A,TT}}, b::AbstractArray
) where {T,N,A<:Exp{T,N},TT<:AbstractArray{T,N}}
	L = J.A
	return y .= conj.(exp.(L.x)) .* b
end

fun_name(L::Exp) = "e"

size(L::Exp) = (L.dim, L.dim)

domainType(::Exp{T,N}) where {T,N} = T
codomainType(::Exp{T,N}) where {T,N} = T
is_thread_safe(::Exp) = true
