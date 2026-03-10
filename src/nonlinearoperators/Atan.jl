export Atan

"""
	Atan([domain_type=Float64::Type,] dim_in::Tuple)

Creates an inverse tangent non-linear operator with input dimensions `dim_in`:
```math
\\text{atan} ( \\mathbf{x} ).
```

"""
struct Atan{T, N} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Atan(domain_type::Type, DomainDim::NTuple{N, Int}) where {N}
    return Atan{domain_type, N}(DomainDim)
end

Atan(DomainDim::NTuple{N, Int}) where {N} = Atan{Float64, N}(DomainDim)
Atan(DomainDim::Vararg{Int}) = Atan{Float64, length(DomainDim)}(DomainDim)

function mul!(y::AbstractArray, L::Atan, x::AbstractArray)
    check(y, L, x)
    return y .= atan.(x)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Atan}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    return y .= conj.(1.0 ./ (1.0 .+ L.x .^ 2)) .* b
end

fun_name(L::Atan) = "atan"

size(L::Atan) = (L.dim, L.dim)

domain_type(::Atan{T, N}) where {T, N} = T
codomain_type(::Atan{T, N}) where {T, N} = T
is_thread_safe(::Atan) = true
