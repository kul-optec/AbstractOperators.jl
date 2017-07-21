export Sigmoid

immutable Sigmoid{T,N,G<:Real} <: NonLinearOperator
	dim::NTuple{N,Int}
	gamma::G
end

function Sigmoid{N, G <: Real}(DomainType::Type, DomainDim::NTuple{N,Int}, gamma::G)  
	Sigmoid{DomainType,N,G}(DomainDim,gamma)
end

function A_mul_B!{T,N,G}(y::AbstractArray{T,N}, L::Sigmoid{T,N,G}, x::AbstractArray{T,N})
	y .= (1.+exp.(-L.gamma.*x)).^(-1)
end

function A_mul_B!{T,N,G, A <: Sigmoid{T,N,G}}(y::AbstractArray{T,N}, L::Jacobian{A}, b::AbstractArray{T,N})
	y .= (L.A.gamma.*(1.+exp.(-L.A.gamma*L.x)).^(-2).*exp.(-L.A.gamma.*L.x)).*b
end

function Ac_mul_B!{T,N,G, A<: Sigmoid{T,N,G}}(y::AbstractArray{T,N}, L::Jacobian{A}, b::AbstractArray{T,N})
	y .= conj.(L.A.gamma.*(1.+exp.(-L.A.gamma*L.x)).^(-2).*exp.(-L.A.gamma.*L.x)).*b
end

fun_name(L::Sigmoid) = "σ"

size(L::Sigmoid) = (L.dim, L.dim)

domainType{T,N,D}(L::Sigmoid{T,N,D}) = T
codomainType{T,N,D}(L::Sigmoid{T,N,D}) = T
