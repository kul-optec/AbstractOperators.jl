export Zeros

"""
`Zeros(domainType::Type, dim_in::Tuple, [codomainType::Type,] dim_out::Tuple)`

Create a `LinearOperator` which, when multiplied with an array `x` of size `dim_in`, returns an array `y` of size `dim_out` filled with zeros.

For convenience `Zeros` can be constructed from any `AbstractOperator`.

```julia
julia> Zeros(Eye(10,20))
0  ℝ^(10, 20) -> ℝ^(10, 20)

julia> Zeros([Eye(10,20) Eye(10,20)])
[0,0]  ℝ^(10, 20)  ℝ^(10, 20) -> ℝ^(10, 20)
```

"""
struct Zeros{C,N,D,M} <: LinearOperator
	dim_out::NTuple{N, Int}
	dim_in::NTuple{M, Int}
end

# Constructors
#default
Zeros(domainType::Type, dim_in::NTuple{M,Int},
      codomainType::Type, dim_out::NTuple{N,Int}) where {N,M} =
Zeros{codomainType,N,domainType,M}(dim_out,dim_in)

Zeros(domainType::Type, dim_in::NTuple{M,Int}, dim_out::NTuple{N,Int}) where {N,M} =
Zeros{domainType,N,domainType,M}(dim_out,dim_in)

function Zeros(domainType::NTuple{NN,Type}, dim_in::NTuple{NN,Tuple},
	       codomainType::Type, dim_out::Tuple) where {NN}
	HCAT([Zeros(domainType[i], dim_in[i], codomainType, dim_out) for i =1:NN]...)
end

function Zeros(domainType::Type, dim_in::Tuple,
	       codomainType::NTuple{NN,Type}, dim_out::NTuple{NN,Tuple}) where {NN}
	VCAT([Zeros(domainType, dim_in, codomainType[i], dim_out[i]) for i =1:NN]...)
end

Zeros(A::AbstractOperator) = Zeros(domainType(A),size(A,2),codomainType(A),size(A,1))

# Mappings

mul!(y::AbstractArray{C,N}, A::Zeros{C,N,D,M}, b::AbstractArray{D,M}) where {C,N,D,M}= fill!(y,zero(C))
mul!(y::AbstractArray{D,M}, A::AdjointOperator{Zeros{C,N,D,M}}, b::AbstractArray{C,N}) where {C,N,D,M}= fill!(y,zero(D))

# Properties

  domainType(L::Zeros{C,N,D,M}) where {C,N,D,M} = D
codomainType(L::Zeros{C,N,D,M}) where {C,N,D,M} = C

size(L::Zeros) = (L.dim_out, L.dim_in)

fun_name(A::Zeros)  = "0"

is_null(L::Zeros) = true
