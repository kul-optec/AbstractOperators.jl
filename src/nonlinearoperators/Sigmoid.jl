export Sigmoid

"""
	Sigmoid([domain_type=Float64::Type,] dim_in::Tuple, γ = 1.)

Creates the sigmoid non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\frac{1}{1+e^{-\\gamma \\mathbf{x} } }
```

"""
struct Sigmoid{T, N, G <: Real, S <: AbstractArray{T}, Th} <: NonLinearOperator
    dim::NTuple{N, Int}
    gamma::G
end

function Sigmoid(
        domain_type::Type{T},
        DomainDim::NTuple{N, Int},
        gamma::G = 1.0;
        array_type::Type = Array{T},
        threaded::Bool = true,
    ) where {T, N, G <: Real}
    S = _normalize_array_type(array_type, T)
    Th = _fbthread(_elementwise_threaded(Sigmoid, threaded, T, DomainDim, S))
    return Sigmoid{T, N, G, S, Th}(DomainDim, gamma)
end

function Sigmoid(
        DomainDim::NTuple{N, Int}, gamma::G = 1.0;
        array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N, G}
    return Sigmoid(Float64, DomainDim, gamma; array_type, threaded)
end

function Sigmoid(
        x::AbstractArray{T};
        gamma::G = 1.0, array_type::Type = _array_wrapper(x), threaded::Bool = true
    ) where {T, G <: Real}
    S = _normalize_array_type(array_type, T)
    Th = _fbthread(_elementwise_threaded(Sigmoid, threaded, T, size(x), S))
    return Sigmoid{T, ndims(x), G, S, Th}(size(x), gamma)
end

# One method per direction, parameterized by `Th`, rather than a `false`/`true` pair:
# `Th` is `FastBroadcast.True()`/`False()` (see `_fbthread`), so `@.. thread = Th` resolves
# to the same specialized code either way -- see `Scale`/`DiagOp` for the same pattern.
function mul!(y::AbstractArray, L::Sigmoid{T, N, G, S, Th}, x::AbstractArray) where {T, N, G, S, Th}
    check(y, L, x)
    gamma = L.gamma
    return @.. thread = Th y = (1 + exp(-gamma * x))^(-1)
end

# The per-element body of the Jacobian-adjoint, as a function rather than spelled out in the
# broadcast: `exp(-gamma * x)` appears twice in the derivative, and a broadcast expression
# evaluates each occurrence separately. Writing it inline cost a second `exp` per element --
# measured at ~2.1x on the `Jacobian/sigmoid-adjoint` benchmark, which is the whole kernel,
# since `exp` dominates it. Binding it to a local here keeps the single fused pass over the
# arrays while evaluating the transcendental once.
@inline function _sigmoid_jac_adj(gamma, x, b)
    e = exp(-gamma * x)
    return conj(gamma * (e / (1 + e)^2)) * b
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Sigmoid{T, N, G, S, Th}}}, b::AbstractArray
    ) where {T, N, G, S, Th}
    check(y, J, b)
    L = J.A
    gamma = L.A.gamma
    Lx = L.x
    # Single fused expression rather than a multi-statement in-place sequence: `@..` fuses
    # it anyway, and this keeps the serial and threaded bodies identical but for the
    # `thread` flag.
    return @.. thread = Th y = _sigmoid_jac_adj(gamma, Lx, b)
end

fun_name(L::Sigmoid) = "σ"

size(L::Sigmoid) = (L.dim, L.dim)

domain_type(::Sigmoid{T, N, D}) where {T, N, D} = T
codomain_type(::Sigmoid{T, N, D}) where {T, N, D} = T
domain_array_type(::Sigmoid{T, N, D, S}) where {T, N, D, S} = S
codomain_array_type(::Sigmoid{T, N, D, S}) where {T, N, D, S} = S
is_thread_safe(::Sigmoid) = true
is_threaded(::Sigmoid{T, N, G, S, Th}) where {T, N, G, S, Th} = _fbbool(Th)
# PROVENANCE: measured per-operator, benchmark/operator_thresholds.jl.
# Crossover of this operator's real `mul!`: Float64 2^10, Float32 2^9.
threading_threshold(::Type{<:Sigmoid}) = 2^10

function _copy_operator_impl(
        op::Sigmoid{T, N, G, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, G, S, Th}
    new_threaded = threaded === nothing ? _fbbool(Th) : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Sigmoid(T, op.dim, op.gamma; array_type = new_at, threaded = new_threaded)
end
supports_threading(::Sigmoid) = true
