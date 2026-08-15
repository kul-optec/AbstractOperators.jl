export Sin

"""
	Sin([domain_type=Float64::Type,] dim_in::Tuple)

Creates a sinusoid non-linear operator with input dimensions `dim_in`:
```math
\\sin( \\mathbf{x} ).
```

"""
struct Sin{T, N, S <: AbstractArray{T}, Th} <: NonLinearOperator
    dim::NTuple{N, Int}
end

function Sin(
        domain_type::Type{T}, DomainDim::NTuple{N, Int};
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return Sin{T, N, S, _elementwise_threaded(Sin, threaded, T, DomainDim, S)}(DomainDim)
end

function Sin(
        DomainDim::NTuple{N, Int}; array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N}
    return Sin(Float64, DomainDim; array_type, threaded)
end
function Sin(DomainDim::Vararg{Int}; array_type::Type = Array{Float64}, threaded::Bool = true)
    return Sin(Float64, DomainDim; array_type, threaded)
end

function Sin(
        x::AbstractArray{T}; array_type::Type = _array_wrapper(x), threaded::Bool = true
    ) where {T}
    S = _normalize_array_type(array_type, T)
    return Sin{T, ndims(x), S, _elementwise_threaded(Sin, threaded, T, size(x), S)}(size(x))
end

function mul!(y::AbstractArray, L::Sin{T, N, S, false}, x::AbstractArray) where {T, N, S}
    check(y, L, x)
    return y .= sin.(x)
end

function mul!(y::AbstractArray, L::Sin{T, N, S, true}, x::AbstractArray) where {T, N, S}
    check(y, L, x)
    return @.. thread = true y = sin(x)
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Sin{T, N, S, false}}}, b::AbstractArray
    ) where {T, N, S}
    check(y, J, b)
    L = J.A
    return y .= conj.(cos.(L.x)) .* b
end

function mul!(
        y::AbstractArray, J::AdjointOperator{<:Jacobian{<:Sin{T, N, S, true}}}, b::AbstractArray
    ) where {T, N, S}
    check(y, J, b)
    L = J.A
    return @.. thread = true y = conj(cos(L.x)) * b
end

fun_name(L::Sin) = "sin"

size(L::Sin) = (L.dim, L.dim)

domain_type(::Sin{T, N}) where {T, N} = T
codomain_type(::Sin{T, N}) where {T, N} = T
domain_array_type(::Sin{T, N, S}) where {T, N, S} = S
codomain_array_type(::Sin{T, N, S}) where {T, N, S} = S
is_thread_safe(::Sin) = true
is_threaded(::Sin{T, N, S, Th}) where {T, N, S, Th} = Th
# PROVENANCE: measured per-operator, benchmark/operator_thresholds.jl.
# Crossover of this operator's real `mul!`: Float64 2^10, Float32 2^10.
threading_threshold(::Type{<:Sin}) = 2^10

function _copy_operator_impl(
        op::Sin{T, N, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, S, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Sin(T, op.dim; array_type = new_at, threaded = new_threaded)
end
supports_threading(::Sin) = true
