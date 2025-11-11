export Cos

"""
	Cos([domain_type=Float64::Type,] dim_in::Tuple)

Creates a cosine non-linear operator with input dimensions `dim_in`:
```math
\\cos (\\mathbf{x} ).
```

"""
struct Cos{T,N} <: NonLinearOperator
	dim::NTuple{N,Int}
end

function Cos(domain_type::Type, DomainDim::NTuple{N,Int}) where {N}
	return Cos{domain_type,N}(DomainDim)
end

Cos(DomainDim::NTuple{N,Int}) where {N} = Cos{Float64,N}(DomainDim)
Cos(DomainDim::Vararg{Int}) = Cos{Float64,length(DomainDim)}(DomainDim)

function mul!(y::AbstractArray{T,N}, ::Cos{T,N}, x::AbstractArray{T,N}) where {T,N}
	return y .= cos.(x)
end

function mul!(
	y::AbstractArray, J::AdjointOperator{Jacobian{A,TT}}, b::AbstractArray
) where {T,N,A<:Cos{T,N},TT<:AbstractArray{T,N}}
	L = J.A
	return y .= -conj.(sin.(L.x)) .* b
end

fun_name(L::Cos) = "cos"

size(L::Cos) = (L.dim, L.dim)

domain_type(::Cos{T,N}) where {T,N} = T
codomain_type(::Cos{T,N}) where {T,N} = T
is_thread_safe(::Cos) = true
