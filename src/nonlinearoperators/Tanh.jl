export Tanh

"""
	Tanh([domain_type=Float64::Type,] dim_in::Tuple)

Creates an hyperbolic tangent non-linear operator with input dimensions `dim_in`:
```math
\\text{tanh} ( \\mathbf{x} ).
```

"""
struct Tanh{T, N, S <: AbstractArray{T}, Th} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Tanh(
        domain_type::Type{T}, DomainDim::NTuple{N, Int};
        array_type::Type = Array{T}, threaded = nothing
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return Tanh{T, N, S, _elementwise_threaded(Tanh, threaded, T, DomainDim, S)}(DomainDim)
end

function Tanh(
        DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}, threaded = nothing
    ) where {N}
    return Tanh(Float64, DomainDim; array_type, threaded)
end
function Tanh(DomainDim::Vararg{Int}; array_type::Type = Array{Float64}, threaded = nothing)
    return Tanh(Float64, DomainDim; array_type, threaded)
end

function Tanh(
        x::AbstractArray{T}; array_type::Type = _array_wrapper(x), threaded = nothing
    ) where {T}
    S = _normalize_array_type(array_type, T)
    return Tanh{T, ndims(x), S, _elementwise_threaded(Tanh, threaded, T, size(x), S)}(size(x))
end

function mul!(y::AbstractArray, L::Tanh{T, N, S, false}, x::AbstractArray) where {T, N, S}
    check(y, L, x)
    return y .= tanh.(x)
end

function mul!(y::AbstractArray, L::Tanh{T, N, S, true}, x::AbstractArray) where {T, N, S}
    check(y, L, x)
    return @.. thread = true y = tanh(x)
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Tanh{T, N, S, false}}}, b::AbstractArray
    ) where {T, N, S}
    check(y, J, b)
    L = J.A
    return y .= conj.(sech.(L.x) .^ 2) .* b
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Tanh{T, N, S, true}}}, b::AbstractArray
    ) where {T, N, S}
    check(y, J, b)
    L = J.A
    return @.. thread = true y = conj(sech(L.x) ^ 2) * b
end

fun_name(L::Tanh) = "tanh"

size(L::Tanh) = (L.dim, L.dim)

domain_type(::Tanh{T, N}) where {T, N} = T
codomain_type(::Tanh{T, N}) where {T, N} = T
domain_array_type(::Tanh{T, N, S}) where {T, N, S} = S
codomain_array_type(::Tanh{T, N, S}) where {T, N, S} = S
is_thread_safe(::Tanh) = true
is_threaded(::Tanh{T, N, S, Th}) where {T, N, S, Th} = Th
threading_threshold(::Type{<:Tanh}) = THRESHOLD_ELEMENTWISE_TRANSCENDENTAL

function _copy_operator_impl(
        op::Tanh{T, N, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, S, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Tanh(T, op.dim; array_type = new_at, threaded = new_threaded)
end
supports_threading(::Tanh) = true
