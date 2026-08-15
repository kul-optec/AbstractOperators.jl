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
    return Sigmoid{T, N, G, S, _elementwise_threaded(Sigmoid, threaded, T, DomainDim, S)}(
        DomainDim, gamma
    )
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
    return Sigmoid{T, ndims(x), G, S, _elementwise_threaded(Sigmoid, threaded, T, size(x), S)}(
        size(x), gamma
    )
end

function mul!(y::AbstractArray, L::Sigmoid{T, N, G, S, false}, x::AbstractArray) where {T, N, G, S}
    check(y, L, x)
    return y .= (1 .+ exp.(-L.gamma .* x)) .^ (-1)
end

function mul!(y::AbstractArray, L::Sigmoid{T, N, G, S, true}, x::AbstractArray) where {T, N, G, S}
    check(y, L, x)
    gamma = L.gamma
    return @.. thread = true y = (1 + exp(-gamma * x)) ^ (-1)
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Sigmoid{T, N, G, S, false}}}, b::AbstractArray
    ) where {T, N, G, S}
    check(y, J, b)
    L = J.A
    y .= exp.(-L.A.gamma .* L.x)
    y ./= (1 .+ y) .^ 2
    y .= conj.(L.A.gamma .* y)
    return y .*= b
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Sigmoid{T, N, G, S, true}}}, b::AbstractArray
    ) where {T, N, G, S}
    check(y, J, b)
    L = J.A
    gamma = L.A.gamma
    Lx = L.x
    # Single fused pass rather than the four-statement in-place sequence the serial path
    # uses: `@..` fuses it anyway, and reading `y` back mid-sequence would be a per-element
    # dependency across threads.
    return @.. thread = true y = conj(gamma * (exp(-gamma * Lx) / (1 + exp(-gamma * Lx))^2)) * b
end

fun_name(L::Sigmoid) = "σ"

size(L::Sigmoid) = (L.dim, L.dim)

domain_type(::Sigmoid{T, N, D}) where {T, N, D} = T
codomain_type(::Sigmoid{T, N, D}) where {T, N, D} = T
domain_array_type(::Sigmoid{T, N, D, S}) where {T, N, D, S} = S
codomain_array_type(::Sigmoid{T, N, D, S}) where {T, N, D, S} = S
is_thread_safe(::Sigmoid) = true
is_threaded(::Sigmoid{T, N, G, S, Th}) where {T, N, G, S, Th} = Th
# PROVENANCE: measured per-operator, benchmark/operator_thresholds.jl.
# Crossover of this operator's real `mul!`: Float64 2^10, Float32 2^9.
threading_threshold(::Type{<:Sigmoid}) = 2^10

function _copy_operator_impl(
        op::Sigmoid{T, N, G, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, G, S, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Sigmoid(T, op.dim, op.gamma; array_type = new_at, threaded = new_threaded)
end
supports_threading(::Sigmoid) = true
