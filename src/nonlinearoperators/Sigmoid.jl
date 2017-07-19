export Sigmoid

immutable Sigmoid{T,N,G<:Real} <: NonLinearOperator
	dim::NTuple{N,Int}
	gamma::G
end

mutable struct JacSigmoid{T,N,G<:Real} <: LinearOperator
	dim::NTuple{N,Int}
	gamma::G
	x::AbstractArray{T,N}
end

function Sigmoid{N, G <: Real}(DomainType::Type, DomainDim::NTuple{N,Int}, gamma::G)  
	Sigmoid{DomainType,N,G}(DomainDim,gamma)
end

function A_mul_B!{T,N,G}(y::AbstractArray{T,N}, L::Sigmoid{T,N,G}, x::AbstractArray{T,N})
	y .= (1.+exp.(-L.gamma.*x)).^(-1)
end

jacobian{T,N,G}(L::Sigmoid{T,N,G}, x0::AbstractArray{T,N}) = JacSigmoid{T,N,G}(L.dim,L.gamma,x0)

function A_mul_B!{T,N,G}(y::AbstractArray{T,N}, L::JacSigmoid{T,N,G}, b::AbstractArray{T,N})
	y .= (L.gamma.*(1.+exp.(-L.gamma*L.x)).^(-2).*exp.(-L.gamma.*L.x)).*b
end

function Ac_mul_B!{T,N,G}(y::AbstractArray{T,N}, L::JacSigmoid{T,N,G}, b::AbstractArray{T,N})
	y .= conj.(L.gamma.*(1.+exp.(-L.gamma*L.x)).^(-2).*exp.(-L.gamma.*L.x)).*b
end

fun_name(L::Sigmoid) = "σ"
fun_name(L::JacSigmoid) = "J(σ)"

size(L::Sigmoid) = (L.dim, L.dim)
size(L::JacSigmoid) = (L.dim, L.dim)

domainType{T,N,D}(L::Sigmoid{T,N,D}) = T
domainType{T,N,D}(L::JacSigmoid{T,N,D}) = T
codomainType{T,N,D}(L::Sigmoid{T,N,D}) = T
codomainType{T,N,D}(L::JacSigmoid{T,N,D}) = T
