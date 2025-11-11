export Pow

"""
	Pow([domain_type=Float64::Type,] dim_in::Tuple)

Elementwise power `p` non-linear operator with input dimensions `dim_in`.

"""
struct Pow{T,N,I<:Real} <: NonLinearOperator
	dim::NTuple{N,Int}
	p::I
end

function Pow(domain_type::Type, DomainDim::NTuple{N,Int}, p::I) where {N,I<:Real}
	return Pow{domain_type,N,I}(DomainDim, p)
end

Pow(DomainDim::NTuple{N,Int}, p::I) where {N,I<:Real} = Pow{Float64,N,I}(DomainDim, p)

function mul!(y::AbstractArray{T,N}, L::Pow{T,N,I}, x::AbstractArray{T,N}) where {T,N,I}
	return y .= x .^ L.p
end

function mul!(
	y::AbstractArray, J::AdjointOperator{Jacobian{Pow{T,N,I},TT}}, b::AbstractArray
) where {T,N,I,TT<:AbstractArray{T,N}}
	L = J.A
	return y .= conj.(L.A.p .* (L.x) .^ (L.A.p - 1)) .* b
end

fun_name(L::Pow) = "『"

size(L::Pow) = (L.dim, L.dim)

domain_type(::Pow{T,N}) where {T,N} = T
codomain_type(::Pow{T,N}) where {T,N} = T
is_thread_safe(::Pow) = true
