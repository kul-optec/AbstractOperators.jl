export Sech

"""
	Sech([domain_type=Float64::Type,] dim_in::Tuple)

Creates an hyperbolic secant non-linear operator with input dimensions `dim_in`:
```math
\\text{sech} ( \\mathbf{x} ).
```

"""
struct Sech{T, N, S <: AbstractArray{T}, Th} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Sech(
        domain_type::Type{T}, DomainDim::NTuple{N, Int};
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    Th = _fbthread(_elementwise_threaded(Sech, threaded, T, DomainDim, S))
    return Sech{T, N, S, Th}(DomainDim)
end

function Sech(
        DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N}
    return Sech(Float64, DomainDim; array_type, threaded)
end
function Sech(DomainDim::Vararg{Int}; array_type::Type = Array{Float64}, threaded::Bool = true)
    return Sech(Float64, DomainDim; array_type, threaded)
end

function Sech(
        x::AbstractArray{T}; array_type::Type = _array_wrapper(x), threaded::Bool = true
    ) where {T}
    S = _normalize_array_type(array_type, T)
    Th = _fbthread(_elementwise_threaded(Sech, threaded, T, size(x), S))
    return Sech{T, ndims(x), S, Th}(size(x))
end

# One method per direction, parameterized by `Th`, rather than a `false`/`true` pair:
# `Th` is `FastBroadcast.True()`/`False()` (see `_fbthread`), so `@.. thread = Th` resolves
# to the same specialized code either way -- see `Scale`/`DiagOp` for the same pattern.
function mul!(y::AbstractArray, L::Sech{T, N, S, Th}, x::AbstractArray) where {T, N, S, Th}
    check(y, L, x)
    return @.. thread = Th y = sech(x)
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Sech{T, N, S, Th}}}, b::AbstractArray
    ) where {T, N, S, Th}
    check(y, J, b)
    L = J.A
    return @.. thread = Th y = -conj(tanh(L.x) * sech(L.x)) * b
end

fun_name(L::Sech) = "sech"

size(L::Sech) = (L.dim, L.dim)

domain_type(::Sech{T, N}) where {T, N} = T
codomain_type(::Sech{T, N}) where {T, N} = T
domain_array_type(::Sech{T, N, S}) where {T, N, S} = S
codomain_array_type(::Sech{T, N, S}) where {T, N, S} = S
is_thread_safe(::Sech) = true
is_threaded(::Sech{T, N, S, Th}) where {T, N, S, Th} = _fbbool(Th)
# PROVENANCE: measured per-operator, benchmark/operator_thresholds.jl.
# Crossover of this operator's real `mul!`: Float64 2^9, Float32 2^9.
threading_threshold(::Type{<:Sech}) = 2^9

function _copy_operator_impl(
        op::Sech{T, N, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, S, Th}
    new_threaded = threaded === nothing ? _fbbool(Th) : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Sech(T, op.dim; array_type = new_at, threaded = new_threaded)
end
supports_threading(::Sech) = true
