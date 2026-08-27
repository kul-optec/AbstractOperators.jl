export ZeroPad

"""
	ZeroPad([domain_type::Type,] dim_in::Tuple, zp::Tuple)
	ZeroPad(x::AbstractArray, zp::Tuple)

Create a `LinearOperator` which, when multiplied to an array `x` of size `dim_in`, returns an expanded array `y` of size `dim_in .+ zp` where `y[1:dim_in[1], 1:dim_in[2] ... ] = x` and zero elsewhere.

```jldoctest
julia> Z = ZeroPad((2,2),(0,2))
[I;0]  ℝ^(2, 2) -> ℝ^(2, 4)

julia> Z*ones(2,2)
2×4 Matrix{Float64}:
 1.0  1.0  0.0  0.0
 1.0  1.0  0.0  0.0
	
```
"""
struct ZeroPad{N, T, S <: AbstractArray{T}} <: LinearOperator
    dim_in::NTuple{N, Int}
    zp::NTuple{N, Int}
end

# Constructors
#standard constructor
function ZeroPad(
        domain_type::Type{T}, dim_in::NTuple{N, Int}, zp::NTuple{M, Int};
        array_type::Type = Array{T}
    ) where {T, N, M}
    M != N && error("dim_in and zp must have the same length")
    any([zp...] .< 0) && error("zero padding cannot be negative")
    S = _normalize_array_type(array_type, T)
    return ZeroPad{N, T, S}(dim_in, zp)
end

function ZeroPad(dim_in::Tuple, zp::NTuple{N, Int}; array_type::Type = Array{Float64}) where {N}
    return ZeroPad(Float64, dim_in, zp; array_type)
end
function ZeroPad(
        domain_type::Type{T}, dim_in::Tuple, zp::Vararg{Int, N}; array_type::Type = Array{T}
    ) where {T, N}
    return ZeroPad(domain_type, dim_in, zp; array_type)
end
function ZeroPad(dim_in::Tuple, zp::Vararg{Int, N}; array_type::Type = Array{Float64}) where {N}
    return ZeroPad(Float64, dim_in, zp; array_type)
end
function ZeroPad(x::AbstractArray{T}, zp::NTuple{N, Int}) where {T, N}
    S = _normalize_array_type(_array_wrapper(x), T)
    return ZeroPad{N, T, S}(size(x), zp)
end
function ZeroPad(x::AbstractArray{T}, zp::Vararg{Int, N}) where {T, N}
    S = _normalize_array_type(_array_wrapper(x), T)
    return ZeroPad{N, T, S}(size(x), zp)
end

# Mappings
@generated function mul!(y::AbstractArray, L::ZeroPad{N, T}, b::AbstractArray) where {N, T}
    z = zero(T)  # compile-time literal zero for type T
    vars = [Symbol("i$i") for i in 1:N]
    # Condition: i1 <= size(b,1) && i2 <= size(b,2) && ...
    cond = if N == 1
        :($(vars[1]) <= size(b, 1))
    else
        Expr(:&&, [:($(vars[i]) <= size(b, $i)) for i in 1:N]...)
    end
    # Inner assignment: @inbounds y[i1,i2,...] = cond ? b[i1,i2,...] : z
    assign = :(@inbounds y[$(vars...)] = $cond ? b[$(vars...)] : $z)
    # Wrap in nested for loops: outermost=dim N, innermost=dim 1 (column-major)
    body = assign
    for i in 1:N
        body = :(
            for $(vars[i]) in 1:size(y, $i)
                ;$body
            end
        )
    end
    return quote
        check(y, L, b)
        $body
        return y
    end
end

function mul!(y::AbstractArray, L::AdjointOperator{<:ZeroPad}, b::AbstractArray)
    check(y, L, b)
    copyto!(y, view(b, ntuple(i -> 1:L.A.dim_in[i], length(L.A.dim_in))...))
    return y
end

function get_normal_op(L::ZeroPad{N, T, S}) where {N, T, S}
    return Eye(domain_type(L), size(L, 2); array_type = S)
end

# Properties

domain_type(::ZeroPad{<:Any, T}) where {T} = T
codomain_type(::ZeroPad{<:Any, T}) where {T} = T
domain_array_type(::ZeroPad{N, T, S}) where {N, T, S} = S
codomain_array_type(::ZeroPad{N, T, S}) where {N, T, S} = S
is_thread_safe(::ZeroPad) = true

size(L::ZeroPad) = L.dim_in .+ L.zp, L.dim_in

fun_name(L::ZeroPad) = "[I;0]"
is_AAc_diagonal(L::ZeroPad) = true
function diag_AAc(L::ZeroPad)
    input = allocate_in_domain(L)
    fill!(input, 1)
    return L * L' * input
end
is_AcA_diagonal(L::ZeroPad) = true
diag_AcA(L::ZeroPad) = 1

is_full_column_rank(L::ZeroPad) = true

has_fast_opnorm(::ZeroPad) = true
LinearAlgebra.opnorm(L::ZeroPad) = one(real(domain_type(L)))

# No threaded execution path (see `supports_threading`), so `threaded` is accepted purely so
# that forwarders can pass it down uniformly, and has no effect here. `storage_type` is
# honoured: it is the whole reason this method exists rather than the deepcopy fallback.

function _copy_operator_impl(
        op::ZeroPad{N, T, S}; storage_type = nothing, threaded = nothing
    ) where {N, T, S}
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return ZeroPad(T, op.dim_in, op.zp; array_type = new_at)
end
