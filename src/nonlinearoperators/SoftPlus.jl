export SoftPlus

"""
	SoftPlus([domain_type=Float64::Type,] dim_in::Tuple)

Creates the softplus non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\log (1 + e^{x} )
```

"""
struct SoftPlus{T, N, S <: AbstractArray{T}, Th} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function SoftPlus(
        domain_type::Type{T}, DomainDim::NTuple{N, Int};
        array_type::Type = Array{T}, threaded = nothing
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return SoftPlus{T, N, S, _elementwise_threaded(SoftPlus, threaded, T, DomainDim, S)}(DomainDim)
end

function SoftPlus(DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}) where {N}
    return SoftPlus(Float64, DomainDim; array_type)
end

function SoftPlus(
        x::AbstractArray{T}; array_type::Type = _array_wrapper(x), threaded = nothing
    ) where {T}
    S = _normalize_array_type(array_type, T)
    return SoftPlus{T, ndims(x), S, _elementwise_threaded(SoftPlus, threaded, T, size(x), S)}(size(x))
end

function mul!(y::AbstractArray, L::SoftPlus{T, N, S, false}, x::AbstractArray) where {T, N, S}
    check(y, L, x)
    return y .= log.(1 .+ exp.(x))
end

function mul!(y::AbstractArray, L::SoftPlus{T, N, S, true}, x::AbstractArray) where {T, N, S}
    check(y, L, x)
    return @.. thread = true y = log(1 + exp(x))
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:SoftPlus{T, N, S, false}}}, b::AbstractArray
    ) where {T, N, S}
    check(y, J, b)
    L = J.A
    return y .= 1 ./ (1 .+ exp.(-L.x)) .* b
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:SoftPlus{T, N, S, true}}}, b::AbstractArray
    ) where {T, N, S}
    check(y, J, b)
    L = J.A
    return @.. thread = true y = 1 / (1 + exp(-L.x)) * b
end

fun_name(L::SoftPlus) = "σ"

size(L::SoftPlus) = (L.dim, L.dim)

domain_type(::SoftPlus{T, N}) where {T, N} = T
codomain_type(::SoftPlus{T, N}) where {T, N} = T
domain_array_type(::SoftPlus{T, N, S}) where {T, N, S} = S
codomain_array_type(::SoftPlus{T, N, S}) where {T, N, S} = S
is_thread_safe(::SoftPlus) = true
is_threaded(::SoftPlus{T, N, S, Th}) where {T, N, S, Th} = Th
threading_threshold(::Type{<:SoftPlus}) = THRESHOLD_ELEMENTWISE_TRANSCENDENTAL

function _copy_operator_impl(
        op::SoftPlus{T, N, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, S, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return SoftPlus(T, op.dim; array_type = new_at, threaded = new_threaded)
end
supports_threading(::SoftPlus) = true
