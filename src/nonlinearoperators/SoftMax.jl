export SoftMax

"""
	SoftMax([domain_type=Float64::Type,] dim_in::Tuple)

Creates the softmax non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\frac{e^{\\mathbf{x} }}{ \\sum e^{\\mathbf{x} } }
```

- `threaded`: `false` forces `mul!` to run serially; `true` (default) threads the max/sum
  reductions and the elementwise passes via `Polyester.@batch reduction=...`, subject to
  the size policy. See the module note above `mul!` for why this needed a rewrite rather
  than a plain `@..`.

"""
struct SoftMax{T, N, B <: AbstractArray{T, N}, Th} <: NonLinearOperator
    dim::NTuple{N, Int}
    buf::B
end

function SoftMax(
        x::AbstractArray{T, N}; array_type::Type = _array_wrapper(x), threaded::Bool = true
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    buf = similar(S, size(x))
    th = _elementwise_threaded(SoftMax, threaded, T, size(x), S)
    return SoftMax{T, N, typeof(buf), th}(size(x), buf)
end

function SoftMax(
        domain_type::Type{T}, DomainDim::NTuple{N, Int};
        array_type::Type = Array{T}, threaded::Bool = true,
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    buf = similar(S, DomainDim)
    fill!(buf, zero(T))
    th = _elementwise_threaded(SoftMax, threaded, T, DomainDim, S)
    return SoftMax{T, N, typeof(buf), th}(DomainDim, buf)
end

function SoftMax(
        DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N}
    return SoftMax(Float64, DomainDim; array_type, threaded)
end

# ─── Threading ────────────────────────────────────────────────────────────────
#
# `mul!` is two reductions (the stabilizing `maximum`, then the normalizing `sum`) around
# an elementwise `exp`, and the Jacobian adjoint has a third (a `dot` product) -- not a
# plain elementwise map, so the `@..`-based split every other nonlinear operator uses does
# not apply directly. Polyester's `@batch reduction=((op, var), ...)` (`+`, `max` here) is
# exactly the "multi-pass rewrite with its own reductions" that used to make this look
# infeasible: each reduction becomes its own `@batch` loop over a plain local accumulator,
# isbits and initialized before the loop as the macro requires.

function mul!(y::AbstractArray, L::SoftMax{T, N, B, false}, x::AbstractArray) where {T, N, B}
    check(y, L, x)
    mx = maximum(x)
    @.. thread = false y = exp(x - mx)
    s = sum(y)
    @.. thread = false y = y / s
    return y
end

function mul!(y::AbstractArray, L::SoftMax{T, N, B, true}, x::AbstractArray) where {T, N, B}
    check(y, L, x)
    mx = T(-Inf)
    @batch reduction = ((max, mx),) for i in eachindex(x)
        mx = max(mx, x[i])
    end
    s = zero(T)
    @batch reduction = ((+, s),) for i in eachindex(x, y)
        y[i] = exp(x[i] - mx)
        s += y[i]
    end
    @batch for i in eachindex(y)
        y[i] /= s
    end
    return y
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:SoftMax{T, N, B, false}}}, b::AbstractArray
    ) where {T, N, B}
    check(y, J, b)
    L = J.A
    mx = maximum(L.x)
    @.. thread = false L.A.buf = exp(L.x - mx)
    s = sum(L.A.buf)
    @.. thread = false L.A.buf = L.A.buf / s
    d = dot(L.A.buf, b)
    @.. thread = false y = L.A.buf * (b - d)
    return y
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:SoftMax{T, N, B, true}}}, b::AbstractArray
    ) where {T, N, B}
    check(y, J, b)
    L = J.A
    buf = L.A.buf
    Lx = L.x
    mx = T(-Inf)
    @batch reduction = ((max, mx),) for i in eachindex(Lx)
        mx = max(mx, Lx[i])
    end
    s = zero(T)
    @batch reduction = ((+, s),) for i in eachindex(Lx, buf)
        buf[i] = exp(Lx[i] - mx)
        s += buf[i]
    end
    @batch for i in eachindex(buf)
        buf[i] /= s
    end
    d = zero(T)
    @batch reduction = ((+, d),) for i in eachindex(buf, b)
        d += buf[i] * b[i]
    end
    @batch for i in eachindex(y)
        y[i] = buf[i] * (b[i] - d)
    end
    return y
end

fun_name(L::SoftMax) = "σ"

size(L::SoftMax) = (L.dim, L.dim)

domain_type(::SoftMax{T}) where {T} = T
codomain_type(::SoftMax{T}) where {T} = T
domain_array_type(::SoftMax{T, N, B}) where {T, N, B} = _array_wrapper_type(B){T}
codomain_array_type(::SoftMax{T, N, B}) where {T, N, B} = _array_wrapper_type(B){T}
# `buf` is shared, mutable operator state written by both the forward and (especially) the
# Jacobian-adjoint path, so two calls on the same instance from different threads can
# interleave regardless of whether either individual call threads internally.
is_thread_safe(::SoftMax) = false

is_threaded(::SoftMax{T, N, B, Th}) where {T, N, B, Th} = Th
supports_threading(::SoftMax) = true

# PROVENANCE: measured, with the size policy bypassed to find the true crossover (sweeping
# through the exported `threaded = true` keyword alone would only ever reproduce whatever
# this constant already was, since `threaded` is a permission the policy can decline).
# Real `mul!` crossover: Float64 2^9, Float32 2^11. Real Jacobian-adjoint crossover (one
# more reduction pass than forward -- max, exp+sum, divide, dot, then the final elementwise
# product): Float64 2^12, Float32 2^11. Taking the latest of all four, as the package does
# everywhere a single threshold has to cover multiple directions/element types: whichever
# crosses over last decides.
threading_threshold(::Type{<:SoftMax}) = 2^12

function _copy_operator_impl(
        op::SoftMax{T, N, B, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, B, Th}
    # `buf` is a working buffer, so it is always freshly allocated rather than shared --
    # sharing it is exactly what makes SoftMax not thread-safe.
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(B) : storage_type
    return SoftMax(T, op.dim; array_type = new_at, threaded = new_threaded)
end
