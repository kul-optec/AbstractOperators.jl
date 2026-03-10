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
struct MyLinOp{N, M, C, D, F <: Function, G <: Function} <: LinearOperator
    dim_out::NTuple{N, Int}
    dim_in::NTuple{M, Int}
    Fwd!::F
    Adj!::G
end

# Constructors

function MyLinOp(
        domain_type::Type,
        dim_in::NTuple{N, Int},
        dim_out::NTuple{M, Int},
        Fwd!::F,
        Adj!::G,
    ) where {N, M, F <: Function, G <: Function}
    return MyLinOp{N, M, domain_type, domain_type, F, G}(dim_out, dim_in, Fwd!, Adj!)
end

function MyLinOp(
        domain_type::Type,
        dim_in::NTuple{N, Int},
        codomain_type::Type,
        dim_out::NTuple{M, Int},
        Fwd!::F,
        Adj!::G,
    ) where {N, M, F <: Function, G <: Function}
    return MyLinOp{N, M, domain_type, codomain_type, F, G}(dim_out, dim_in, Fwd!, Adj!)
end

# Mappings

function mul!(y::AbstractArray, L::MyLinOp, b::AbstractArray)
    check(y, L, b)
    return L.Fwd!(y, b)
end
function mul!(y::AbstractArray, L::AdjointOperator{<:MyLinOp}, b::AbstractArray)
    check(y, L, b)
    return L.A.Adj!(y, b)
end

# Properties

size(L::MyLinOp) = (L.dim_out, L.dim_in)

codomain_type(::MyLinOp{N, M, C}) where {N, M, C} = C
domain_type(::MyLinOp{N, M, C, D}) where {N, M, C, D} = D

fun_name(L::MyLinOp) = "A"
