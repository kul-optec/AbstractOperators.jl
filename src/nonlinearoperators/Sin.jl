export Sin

"""
	Sin([domain_type=Float64::Type,] dim_in::Tuple)

Creates a sinusoid non-linear operator with input dimensions `dim_in`:
```math
\\sin( \\mathbf{x} ).
```

"""
struct Sin{T, N} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Sin(domain_type::Type, DomainDim::NTuple{N, Int}) where {N}
    return Sin{domain_type, N}(DomainDim)
end

Sin(DomainDim::NTuple{N, Int}) where {N} = Sin{Float64, N}(DomainDim)
Sin(DomainDim::Vararg{Int}) = Sin{Float64, length(DomainDim)}(DomainDim)

function mul!(y::AbstractArray, L::Sin, x::AbstractArray)
    check(y, L, x)
    return y .= sin.(x)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Sin}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    return y .= conj.(cos.(L.x)) .* b
end

fun_name(L::Sin) = "sin"

size(L::Sin) = (L.dim, L.dim)

domain_type(::Sin{T, N}) where {T, N} = T
codomain_type(::Sin{T, N}) where {T, N} = T
is_thread_safe(::Sin) = true
