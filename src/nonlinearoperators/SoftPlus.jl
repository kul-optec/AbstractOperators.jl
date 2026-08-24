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
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    Th = _fbthread(_elementwise_threaded(SoftPlus, threaded, T, DomainDim, S))
    return SoftPlus{T, N, S, Th}(DomainDim)
end

function SoftPlus(DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}) where {N}
    return SoftPlus(Float64, DomainDim; array_type)
end

function SoftPlus(
        x::AbstractArray{T}; array_type::Type = _array_wrapper(x), threaded::Bool = true
    ) where {T}
    S = _normalize_array_type(array_type, T)
    Th = _fbthread(_elementwise_threaded(SoftPlus, threaded, T, size(x), S))
    return SoftPlus{T, ndims(x), S, Th}(size(x))
end

# One method per direction, parameterized by `Th`, rather than a `false`/`true` pair:
# `Th` is `FastBroadcast.True()`/`False()` (see `_fbthread`), so `@.. thread = Th` resolves
# to the same specialized code either way -- see `Scale`/`DiagOp` for the same pattern.
function mul!(y::AbstractArray, L::SoftPlus{T, N, S, Th}, x::AbstractArray) where {T, N, S, Th}
    check(y, L, x)
    return @.. thread = Th y = log(1 + exp(x))
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:SoftPlus{T, N, S, Th}}}, b::AbstractArray
    ) where {T, N, S, Th}
    check(y, J, b)
    L = J.A
    return @.. thread = Th y = 1 / (1 + exp(-L.x)) * b
end

fun_name(L::SoftPlus) = "σ"

size(L::SoftPlus) = (L.dim, L.dim)

domain_type(::SoftPlus{T, N}) where {T, N} = T
codomain_type(::SoftPlus{T, N}) where {T, N} = T
domain_array_type(::SoftPlus{T, N, S}) where {T, N, S} = S
codomain_array_type(::SoftPlus{T, N, S}) where {T, N, S} = S
is_thread_safe(::SoftPlus) = true
is_threaded(::SoftPlus{T, N, S, Th}) where {T, N, S, Th} = _fbbool(Th)
# PROVENANCE: measured per-operator, benchmark/operator_thresholds.jl.
# Crossover of this operator's real `mul!`: Float64 2^8, Float32 2^8 -- the earliest crossover of any operator.
threading_threshold(::Type{<:SoftPlus}) = 2^8

function _copy_operator_impl(
        op::SoftPlus{T, N, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, S, Th}
    new_threaded = threaded === nothing ? _fbbool(Th) : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return SoftPlus(T, op.dim; array_type = new_at, threaded = new_threaded)
end
supports_threading(::SoftPlus) = true
