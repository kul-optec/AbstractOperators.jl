export MyLinOp

"""
	MyLinOp(domainType::Type, dim_in::Tuple, [domainType::Type,] dim_out::Tuple, Fwd!::Function, Adj!::Function)

Construct a user defined `LinearOperator` by specifing its linear mapping `Fwd!` and its adjoint `Adj!`. The functions `Fwd!` and `Adj` must be in-place functions consistent with the given dimensions `dim_in` and `dim_out` and the domain and codomain types.

```julia
julia> n,m = 5,4;

julia> A = randn(n,m);

julia> op = MyLinOp(Float64, (m,),(n,), (y,x) -> mul!(y,A,x), (y,x) -> mul!(y,A',x))
A  ℝ^4 -> ℝ^5

julia> op = MyLinOp(Float64, (m,), Float64, (n,), (y,x) -> mul!(y,A,x), (y,x) -> mul!(y,A',x))
A  ℝ^4 -> ℝ^5
	
```
"""
struct MyLinOp{N,M,C,D} <: LinearOperator
	dim_out::NTuple{N,Int}
	dim_in::NTuple{M,Int}
	Fwd!::Function
	Adj!::Function
end

# Constructors

function MyLinOp(
	domainType::Type,
	dim_in::NTuple{N,Int},
	dim_out::NTuple{M,Int},
	Fwd!::Function,
	Adj!::Function,
) where {N,M}
	return MyLinOp{N,M,domainType,domainType}(dim_out, dim_in, Fwd!, Adj!)
end

function MyLinOp(
	domainType::Type,
	dim_in::NTuple{N,Int},
	codomainType::Type,
	dim_out::NTuple{M,Int},
	Fwd!::Function,
	Adj!::Function,
) where {N,M}
	return MyLinOp{N,M,domainType,codomainType}(dim_out, dim_in, Fwd!, Adj!)
end

# Mappings

function mul!(
	y::AbstractArray{C,N}, L::MyLinOp{N,M,C,D}, b::AbstractArray{D,M}
) where {N,M,C,D}
	return L.Fwd!(y, b)
end
function mul!(
	y::AbstractArray{C,N}, L::AdjointOperator{MyLinOp{N,M,C,D}}, b::AbstractArray{D,M}
) where {N,M,C,D}
	return L.A.Adj!(y, b)
end

# Properties

size(L::MyLinOp) = (L.dim_out, L.dim_in)

codomainType(::MyLinOp{N,M,C}) where {N,M,C} = C
domainType(::MyLinOp{N,M,C,D}) where {N,M,C,D} = D

fun_name(L::MyLinOp) = "A"
