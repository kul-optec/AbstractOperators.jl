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
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    Th = _fbthread(_elementwise_threaded(Tanh, threaded, T, DomainDim, S))
    return Tanh{T, N, S, Th}(DomainDim)
end

function Tanh(
        DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N}
    return Tanh(Float64, DomainDim; array_type, threaded)
end
function Tanh(DomainDim::Vararg{Int}; array_type::Type = Array{Float64}, threaded::Bool = true)
    return Tanh(Float64, DomainDim; array_type, threaded)
end

function Tanh(
        x::AbstractArray{T}; array_type::Type = _array_wrapper(x), threaded::Bool = true
    ) where {T}
    S = _normalize_array_type(array_type, T)
    Th = _fbthread(_elementwise_threaded(Tanh, threaded, T, size(x), S))
    return Tanh{T, ndims(x), S, Th}(size(x))
end

# One method per direction, parameterized by `Th`, rather than a `false`/`true` pair:
# `Th` is `FastBroadcast.True()`/`False()` (see `_fbthread`), so `@.. thread = Th` resolves
# to the same specialized code either way -- see `Scale`/`DiagOp` for the same pattern.
function mul!(y::AbstractArray, L::Tanh{T, N, S, Th}, x::AbstractArray) where {T, N, S, Th}
    check(y, L, x)
    return @.. thread = Th y = tanh(x)
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Tanh{T, N, S, Th}}}, b::AbstractArray
    ) where {T, N, S, Th}
    check(y, J, b)
    L = J.A
    return @.. thread = Th y = conj(sech(L.x)^2) * b
end

fun_name(L::Tanh) = "tanh"

size(L::Tanh) = (L.dim, L.dim)

domain_type(::Tanh{T, N}) where {T, N} = T
codomain_type(::Tanh{T, N}) where {T, N} = T
domain_array_type(::Tanh{T, N, S}) where {T, N, S} = S
codomain_array_type(::Tanh{T, N, S}) where {T, N, S} = S
is_thread_safe(::Tanh) = true
is_threaded(::Tanh{T, N, S, Th}) where {T, N, S, Th} = _fbbool(Th)
# PROVENANCE: measured per-operator, benchmark/operator_thresholds.jl.
# Crossover of this operator's real `mul!`: Float64 2^9, Float32 2^9.
threading_threshold(::Type{<:Tanh}) = 2^9

function _copy_operator_impl(
        op::Tanh{T, N, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, S, Th}
    new_threaded = threaded === nothing ? _fbbool(Th) : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Tanh(T, op.dim; array_type = new_at, threaded = new_threaded)
end
supports_threading(::Tanh) = true
