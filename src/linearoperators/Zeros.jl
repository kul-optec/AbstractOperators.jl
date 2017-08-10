export Zeros

immutable Zeros{C,N,D,M} <: LinearOperator
	dim_out::NTuple{N, Int}
	dim_in::NTuple{M, Int}
end

# Constructors
#default 
Zeros{N,M}(domainType::Type, dim_in::NTuple{M,Int}, codomainType::Type, dim_out::NTuple{N,Int}) = 
Zeros{codomainType,N,domainType,M}(dim_out,dim_in)

Zeros{N,M}(domainType::Type, dim_in::NTuple{M,Int}, dim_out::NTuple{N,Int}) = 
Zeros{domainType,N,domainType,M}(dim_out,dim_in)

function Zeros{NN}(domainType::NTuple{NN,Type}, dim_in::NTuple{NN,Tuple}, 
		   codomainType::Type, dim_out::Tuple) 
	HCAT([Zeros(domainType[i], dim_in[i], codomainType, dim_out) for i =1:NN]...)
end

function Zeros{NN}(domainType::Type, dim_in::Tuple, 
		   codomainType::NTuple{NN,Type}, dim_out::NTuple{NN,Tuple}) 
	VCAT([Zeros(domainType, dim_in, codomainType[i], dim_out[i]) for i =1:NN]...)
end

Zeros(A::AbstractOperator) = Zeros(domainType(A),size(A,2),codomainType(A),size(A,1))

# Mappings

A_mul_B!{C,N,D,M}(y::AbstractArray{C,N}, A::Zeros{C,N,D,M}, b::AbstractArray{D,M}) = fill!(y,zero(C))
Ac_mul_B!{C,N,D,M}(y::AbstractArray{D,M}, A::Zeros{C,N,D,M}, b::AbstractArray{C,N}) = fill!(y,zero(D))

# Properties

domainType{C,N,D,M}(L::Zeros{C,N,D,M}) = D
codomainType{C,N,D,M}(L::Zeros{C,N,D,M}) = C

size(L::Zeros) = (L.dim_out, L.dim_in)

fun_name(A::Zeros)  = "0"

is_null(L::Zeros) = true
