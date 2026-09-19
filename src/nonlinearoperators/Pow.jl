export Pow

"""
	Pow([domain_type=Float64::Type,] dim_in::Tuple)

Elementwise power `p` non-linear operator with input dimensions `dim_in`.

"""
struct Pow{T, N, I <: Real, S <: AbstractArray{T}, Th} <: NonLinearOperator
    dim::NTuple{N, Int}
    p::I
end

function Pow(
        domain_type::Type{T}, DomainDim::NTuple{N, Int}, p::I;
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N, I <: Real}
    S = _normalize_array_type(array_type, T)
    Th = _fbthread(_elementwise_threaded(Pow{T, N, I}, threaded, T, DomainDim, S))
    return Pow{T, N, I, S, Th}(DomainDim, p)
end

function Pow(
        DomainDim::NTuple{N, Int}, p::I;
        array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N, I <: Real}
    return Pow(Float64, DomainDim, p; array_type, threaded)
end

function Pow(
        x::AbstractArray{T}, p::I;
        array_type::Type = _array_wrapper(x), threaded::Bool = true
    ) where {T, I <: Real}
    S = _normalize_array_type(array_type, T)
    N = ndims(x)
    Th = _fbthread(_elementwise_threaded(Pow{T, N, I}, threaded, T, size(x), S))
    return Pow{T, N, I, S, Th}(size(x), p)
end

# One method per direction, parameterized by `Th`, rather than a `false`/`true` pair:
# `Th` is `FastBroadcast.True()`/`False()` (see `_fbthread`), so `@.. thread = Th` resolves
# to the same specialized code either way -- see `Scale`/`DiagOp` for the same pattern.
function mul!(y::AbstractArray, L::Pow{T, N, I, S, Th}, x::AbstractArray) where {T, N, I, S, Th}
    check(y, L, x)
    p = L.p
    return @.. thread = Th y = x^p
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Pow{T, N, I, S, Th}}}, b::AbstractArray
    ) where {T, N, I, S, Th}
    check(y, J, b)
    L = J.A
    p = L.A.p
    Lx = L.x
    return @.. thread = Th y = conj(p * Lx^(p - 1)) * b
end

fun_name(L::Pow) = "『"

size(L::Pow) = (L.dim, L.dim)

domain_type(::Pow{T, N}) where {T, N} = T
codomain_type(::Pow{T, N}) where {T, N} = T
domain_array_type(::Pow{T, N, I, S}) where {T, N, I, S} = S
codomain_array_type(::Pow{T, N, I, S}) where {T, N, I, S} = S
is_thread_safe(::Pow) = true
is_threaded(::Pow{T, N, I, S, Th}) where {T, N, I, S, Th} = _fbbool(Th)
# PROVENANCE: measured per-operator, benchmark/operator_thresholds.jl. The exponent kind
# matters enough to split the method: integer `x^2` crosses over at Float64 2^10 / Float32
# 2^11, while fractional `x^0.5` -- which lowers to `exp(p*log(x))` -- crosses at 2^8 for
# both. Taking the conservative value within each kind.
threading_threshold(::Type{<:Pow{T, N, I}}) where {T, N, I <: Integer} = 2^11
threading_threshold(::Type{<:Pow}) = 2^8

function _copy_operator_impl(
        op::Pow{T, N, I, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, I, S, Th}
    new_threaded = threaded === nothing ? _fbbool(Th) : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Pow(T, op.dim, op.p; array_type = new_at, threaded = new_threaded)
end
supports_threading(::Pow) = true
