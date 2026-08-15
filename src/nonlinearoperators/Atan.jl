export Atan

"""
	Atan([domain_type=Float64::Type,] dim_in::Tuple)

Creates an inverse tangent non-linear operator with input dimensions `dim_in`:
```math
\\text{atan} ( \\mathbf{x} ).
```

"""
struct Atan{T, N, S <: AbstractArray{T}, Th} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Atan(
        domain_type::Type{T}, DomainDim::NTuple{N, Int};
        array_type::Type = Array{T}, threaded = nothing
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return Atan{T, N, S, _elementwise_threaded(Atan, threaded, T, DomainDim, S)}(DomainDim)
end

function Atan(
        DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}, threaded = nothing
    ) where {N}
    return Atan(Float64, DomainDim; array_type, threaded)
end
function Atan(DomainDim::Vararg{Int}; array_type::Type = Array{Float64}, threaded = nothing)
    return Atan(Float64, DomainDim; array_type, threaded)
end

function Atan(
        x::AbstractArray{T}; array_type::Type = _array_wrapper(x), threaded = nothing
    ) where {T}
    S = _normalize_array_type(array_type, T)
    return Atan{T, ndims(x), S, _elementwise_threaded(Atan, threaded, T, size(x), S)}(size(x))
end

function mul!(y::AbstractArray, L::Atan{T, N, S, false}, x::AbstractArray) where {T, N, S}
    check(y, L, x)
    return y .= atan.(x)
end

function mul!(y::AbstractArray, L::Atan{T, N, S, true}, x::AbstractArray) where {T, N, S}
    check(y, L, x)
    return @.. thread = true y = atan(x)
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Atan{T, N, S, false}}}, b::AbstractArray
    ) where {T, N, S}
    check(y, J, b)
    L = J.A
    return y .= conj.(1.0 ./ (1.0 .+ L.x .^ 2)) .* b
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Atan{T, N, S, true}}}, b::AbstractArray
    ) where {T, N, S}
    check(y, J, b)
    L = J.A
    return @.. thread = true y = conj(1.0 / (1.0 + L.x ^ 2)) * b
end

fun_name(L::Atan) = "atan"

size(L::Atan) = (L.dim, L.dim)

domain_type(::Atan{T, N}) where {T, N} = T
codomain_type(::Atan{T, N}) where {T, N} = T
domain_array_type(::Atan{T, N, S}) where {T, N, S} = S
codomain_array_type(::Atan{T, N, S}) where {T, N, S} = S
is_thread_safe(::Atan) = true
is_threaded(::Atan{T, N, S, Th}) where {T, N, S, Th} = Th
threading_threshold(::Type{<:Atan}) = THRESHOLD_ELEMENTWISE_TRANSCENDENTAL

function _copy_operator_impl(
        op::Atan{T, N, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, S, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Atan(T, op.dim; array_type = new_at, threaded = new_threaded)
end
supports_threading(::Atan) = true
