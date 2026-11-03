export Eye

"""
	Eye([domain_type=Float64::Type,] dim_in::Tuple)
	Eye([domain_type=Float64::Type,] dims...)

Create the identity operator.

```jldoctest
julia> op = Eye(Float64,(4,))
I  ℝ^4 -> ℝ^4

julia> op = Eye(2,3,4)
I  ℝ^(2, 3, 4) -> ℝ^(2, 3, 4)

julia> op*ones(2,3,4) == ones(2,3,4)
true
	
```
"""
struct Eye{T, N, S <: AbstractArray{T}} <: LinearOperator
    dim::NTuple{N, Int}
end

# Constructors
@inline _make_eye(::Type{T}, dims::NTuple{N, Int}, ::Type{S}) where {T, N, S <: AbstractArray} =
    Eye{T, N, _normalize_array_type(S, T)}(dims)

function Eye(
        domain_type::Type{T}, domainDim::NTuple{N, <:Integer}; array_type::Type{<:AbstractArray} = Array{T}
    ) where {N, T}
    return _make_eye(domain_type, map(Int, domainDim), array_type)
end

function Eye(
        t::Type{T}, dims::Vararg{Integer}; array_type::Type{<:AbstractArray} = Array{T}
    ) where {T}
    return _make_eye(t, map(Int, dims), array_type)
end
function Eye(dims::NTuple{N, Integer}; array_type::Type{<:AbstractArray} = Array{Float64}) where {N}
    return _make_eye(Float64, map(Int, dims), array_type)
end
function Eye(dims::Vararg{Integer}; array_type::Type{<:AbstractArray} = Array{Float64})
    return _make_eye(Float64, map(Int, dims), array_type)
end
function Eye(x::A) where {A <: AbstractArray}
    return _make_eye(eltype(x), size(x), typeof(x isa SubArray ? parent(x) : x))
end

# Mappings

function mul!(y::AbstractArray, L::Eye, b::AbstractArray)
    check(y, L, b)
    y .= b
    return y
end

# Properties
diag(::Eye) = 1.0
diag_AcA(::Eye) = 1.0
diag_AAc(::Eye) = 1.0

domain_type(::Eye{T, N}) where {T, N} = T
codomain_type(::Eye{T, N}) where {T, N} = T
domain_array_type(::Eye{T, N, S}) where {T, N, S} = S
codomain_array_type(::Eye{T, N, S}) where {T, N, S} = S
is_thread_safe(::Eye) = true

size(L::Eye) = (L.dim, L.dim)

fun_name(::Eye) = "I"

is_eye(::Eye) = true
is_diagonal(::Eye) = true
is_orthogonal(::Eye) = true
is_invertible(::Eye) = true
is_full_row_rank(::Eye) = true
is_full_column_rank(::Eye) = true
is_symmetric(::Eye) = true
is_positive_definite(::Eye) = true
is_positive_semidefinite(::Eye) = true

has_optimized_normalop(::Eye) = true
get_normal_op(L::Eye) = L

has_fast_opnorm(::Eye) = true
LinearAlgebra.opnorm(L::Eye) = one(real(domain_type(L)))
AdjointOperator(L::Eye) = L

# No threaded execution path (see `supports_threading`), so `threaded` is accepted purely so
# that forwarders can pass it down uniformly, and has no effect here. `storage_type` is
# honoured: it is the whole reason this method exists rather than the deepcopy fallback.

function _copy_operator_impl(
        op::Eye{T, N, S}; storage_type = nothing, threaded = nothing
    ) where {T, N, S}
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return _make_eye(T, op.dim, new_at)
end
