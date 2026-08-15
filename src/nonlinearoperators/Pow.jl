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
        array_type::Type = Array{T}, threaded = nothing
    ) where {T, N, I <: Real}
    S = _normalize_array_type(array_type, T)
    return Pow{T, N, I, S, _elementwise_threaded(Pow, threaded, T, DomainDim, S)}(DomainDim, p)
end

function Pow(
        DomainDim::NTuple{N, Int}, p::I;
        array_type::Type = Array{Float64}, threaded = nothing
    ) where {N, I <: Real}
    return Pow(Float64, DomainDim, p; array_type, threaded)
end

function Pow(
        x::AbstractArray{T}, p::I;
        array_type::Type = _array_wrapper(x), threaded = nothing
    ) where {T, I <: Real}
    S = _normalize_array_type(array_type, T)
    return Pow{T, ndims(x), I, S, _elementwise_threaded(Pow, threaded, T, size(x), S)}(size(x), p)
end

function mul!(y::AbstractArray, L::Pow{T, N, I, S, false}, x::AbstractArray) where {T, N, I, S}
    check(y, L, x)
    return y .= x .^ L.p
end

function mul!(y::AbstractArray, L::Pow{T, N, I, S, true}, x::AbstractArray) where {T, N, I, S}
    check(y, L, x)
    p = L.p
    return @.. thread = true y = x ^ p
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Pow{T, N, I, S, false}}}, b::AbstractArray
    ) where {T, N, I, S}
    check(y, J, b)
    L = J.A
    return y .= conj.(L.A.p .* (L.x) .^ (L.A.p - 1)) .* b
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Pow{T, N, I, S, true}}}, b::AbstractArray
    ) where {T, N, I, S}
    check(y, J, b)
    L = J.A
    p = L.A.p
    Lx = L.x
    return @.. thread = true y = conj(p * Lx ^ (p - 1)) * b
end

fun_name(L::Pow) = "『"

size(L::Pow) = (L.dim, L.dim)

domain_type(::Pow{T, N}) where {T, N} = T
codomain_type(::Pow{T, N}) where {T, N} = T
domain_array_type(::Pow{T, N, I, S}) where {T, N, I, S} = S
codomain_array_type(::Pow{T, N, I, S}) where {T, N, I, S} = S
is_thread_safe(::Pow) = true
is_threaded(::Pow{T, N, I, S, Th}) where {T, N, I, S, Th} = Th
# `x^p` for non-integer `p` lowers to `exp(p*log(x))`, i.e. transcendental cost; integer
# powers are cheaper but share the constant rather than splitting the type on `isinteger`.
threading_threshold(::Type{<:Pow}) = THRESHOLD_ELEMENTWISE_TRANSCENDENTAL

function _copy_operator_impl(
        op::Pow{T, N, I, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, I, S, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Pow(T, op.dim, op.p; array_type = new_at, threaded = new_threaded)
end
supports_threading(::Pow) = true
