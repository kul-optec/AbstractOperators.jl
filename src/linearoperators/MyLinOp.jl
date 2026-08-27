export MyLinOp

"""
	MyLinOp(domain_type::Type, dim_in::Tuple, [domain_type::Type,] dim_out::Tuple, Fwd!::Function, Adj!::Function)

Construct a user defined `LinearOperator` by specifing its linear mapping `Fwd!` and its adjoint `Adj!`. The functions `Fwd!` and `Adj` must be in-place functions consistent with the given dimensions `dim_in` and `dim_out` and the domain and codomain types.

```jldoctest
julia> n,m = 5,4;

julia> A = randn(n,m);

julia> op = MyLinOp(Float64, (m,),(n,), (y,x) -> mul!(y,A,x), (y,x) -> mul!(y,A',x))
A  ℝ^4 -> ℝ^5

julia> op = MyLinOp(Float64, (m,), Float64, (n,), (y,x) -> mul!(y,A,x), (y,x) -> mul!(y,A',x))
A  ℝ^4 -> ℝ^5
	
```
"""
struct MyLinOp{N, M, C, D, F <: Function, G <: Function, dS, cS} <: LinearOperator
    dim_out::NTuple{N, Int}
    dim_in::NTuple{M, Int}
    Fwd!::F
    Adj!::G
end

# Constructors

function MyLinOp(
        domain_type::Type{T},
        dim_in::NTuple{N, Int},
        dim_out::NTuple{M, Int},
        Fwd!::F,
        Adj!::G,
        ; array_type::Type = Array{T},
    ) where {T, N, M, F <: Function, G <: Function}
    dS = _normalize_array_type(array_type, domain_type)
    return MyLinOp{N, M, domain_type, domain_type, F, G, dS, dS}(dim_out, dim_in, Fwd!, Adj!)
end

function MyLinOp(
        domain_type::Type{T},
        dim_in::NTuple{N, Int},
        codomain_type::Type{C},
        dim_out::NTuple{M, Int},
        Fwd!::F,
        Adj!::G,
        ; array_type::Type = Array{T},
    ) where {T, C, N, M, F <: Function, G <: Function}
    dS = _normalize_array_type(array_type, domain_type)
    cS = _normalize_array_type(array_type, codomain_type)
    return MyLinOp{N, M, domain_type, codomain_type, F, G, dS, cS}(dim_out, dim_in, Fwd!, Adj!)
end

# Mappings

function mul!(y::AbstractArray, L::MyLinOp, b::AbstractArray)
    check(y, L, b)
    L.Fwd!(y, b)
    return y
end
function mul!(y::AbstractArray, L::AdjointOperator{<:MyLinOp}, b::AbstractArray)
    check(y, L, b)
    L.A.Adj!(y, b)
    return y
end

# Properties

size(L::MyLinOp) = (L.dim_out, L.dim_in)

codomain_type(::MyLinOp{N, M, C}) where {N, M, C} = C
domain_type(::MyLinOp{N, M, C, D}) where {N, M, C, D} = D
domain_array_type(::MyLinOp{N, M, C, D, F, G, dS}) where {N, M, C, D, F, G, dS} = dS
codomain_array_type(::MyLinOp{N, M, C, D, F, G, dS, cS}) where {N, M, C, D, F, G, dS, cS} = cS

fun_name(L::MyLinOp) = "A"

# No threaded execution path (see `supports_threading`), so `threaded` is accepted purely so
# that forwarders can pass it down uniformly, and has no effect here. `storage_type` is
# honoured: it is the whole reason this method exists rather than the deepcopy fallback.

function _copy_operator_impl(
        op::MyLinOp{N, M, C, D, F, G, dS, cS}; storage_type = nothing, threaded = nothing
    ) where {N, M, C, D, F, G, dS, cS}
    new_at = storage_type === nothing ? _array_wrapper_type(dS) : storage_type
    return MyLinOp(D, op.dim_in, C, op.dim_out, op.Fwd!, op.Adj!; array_type = new_at)
end
