export MyLinOp

"""
`MyLinOp(domainType::Type, dim_in::Tuple, [domainType::Type,] dim_out::Tuple, Fwd!::Function, Adj!::Function)`

Construct a user defined `LinearOperator` by specifing its linear mapping `Fwd!` and its adjoint `Adj!`. The functions `Fwd!` and `Adj` must be in-place functions consistent with the given dimensions `dim_in` and `dim_out` and the domain and codomain types.

```julia
julia> n,m = 5,4;

julia> A = randn(n,m);

julia> op = MyLinOp(Float64, (m,),(n,), (y,x) -> A_mul_B!(y,A,x), (y,x) -> Ac_mul_B!(y,A,x))
A  ℝ^4 -> ℝ^5

julia> op = MyLinOp(Float64, (m,), Float64, (n,), (y,x) -> A_mul_B!(y,A,x), (y,x) -> Ac_mul_B!(y,A,x))
A  ℝ^4 -> ℝ^5

```

"""

immutable MyLinOp{N,M,C,D} <: LinearOperator
	dim_out::NTuple{N,Int}
	dim_in::NTuple{M,Int}
	Fwd!::Function
	Adj!::Function
end

# Constructors

MyLinOp{N,M}(domainType::Type, dim_in::NTuple{N,Int}, dim_out::NTuple{M,Int},
	   Fwd!::Function, Adj!::Function ) =
MyLinOp{N,M, domainType, domainType}(dim_out, dim_in, Fwd!, Adj! )

MyLinOp{N,M}(domainType::Type, dim_in::NTuple{N,Int}, codomainType::Type, dim_out::NTuple{M,Int},
	   Fwd!::Function, Adj!::Function ) =
MyLinOp{N,M, domainType, codomainType}(dim_out, dim_in, Fwd!, Adj! )

# Mappings

A_mul_B!{N,M,C,D}( y::Array{C,N}, L::MyLinOp{N,M,C,D}, b::Array{D,M}) = L.Fwd!(y,b)
Ac_mul_B!{N,M,C,D}(y::Array{C,N}, L::MyLinOp{N,M,C,D}, b::Array{D,M}) = L.Adj!(y,b)

# Properties

size(L::MyLinOp) = (L.dim_out, L.dim_in)

codomainType{N,M,C,D}(L::MyLinOp{N,M,C,D}) = C
  domainType{N,M,C,D}(L::MyLinOp{N,M,C,D}) = D

fun_name(L::MyLinOp)  = "A"
