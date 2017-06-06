export Zeros

immutable Zeros{C,N,D,M} <: LinearOperator
	dim_out::NTuple{N, Int}
	dim_in::NTuple{M, Int}
end

# Constructors
#default 
Zeros{N,M}(domainType::Type, dim_in::NTuple{M,Int}, codomainType::Type, dim_out::NTuple{N,Int}) = 
Zeros{codomainType,N,domainType,M}(dim_out,dim_in)

# Mappings

function A_mul_B!{C,N,D,M}(y::AbstractArray{C,N}, A::Zeros{C,N,D,M}, b::AbstractArray{D,M})
	y .= zero(C)
end

function Ac_mul_B!{C,N,D,M}(y::AbstractArray{D,M}, A::Zeros{C,N,D,M}, b::AbstractArray{C,N})
	y .= zero(D)
end

# Properties

domainType{C,N,D,M}(L::Zeros{C,N,D,M}) = D
codomainType{C,N,D,M}(L::Zeros{C,N,D,M}) = C

size(L::Zeros) = (L.dim_out, L.dim_in)

fun_name(A::Zeros)  = "0"

is_null(L::Zeros) = true
is_diagonal(L::Zeros) = true
is_gram_diagonal(L::Zeros) = true
