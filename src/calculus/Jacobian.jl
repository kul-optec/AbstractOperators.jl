export Jacobian

immutable Jacobian{T <: NonLinearOperator, X<:Union{AbstractArray,NTuple}} <: LinearOperator
	A::T
	x::X
end

#Jacobian of LinearOperator 
Jacobian{T<:LinearOperator,X<:Union{AbstractArray,NTuple}}(L::T,x::X) = L
#Jacobian of Scale
Jacobian{T,L,T2<:Scale{T,L}}(S::T2,x::AbstractArray) = Scale(S.coeff,Jacobian(S.A,x)) 
#Jacobian of DCAT
Jacobian{N,C,D}(L::DCAT{N,C,D},x::D) = DCAT(Jacobian.(L.A,x)...) 
#Jacobian of HCAT
function Jacobian{M,N,L,P,C,D}(H::HCAT{M,N,L,P,C},x::D)  
	A = ()
	c = 0
	for i = 1:N
		A = length(H.idxs[i]) == 1 ?
		(A...,jacobian(H.A[i],x[c+1])) :
		(A...,jacobian(H.A[i],x[c+1:c+length(H.idxs[i])])) 
		c += length(H.idxs[i])
	end
	HCAT(A,H.idxs,H.mid,M)
end
#Jacobian of VCAT
Jacobian{M,N,L,P,C,D}(V::VCAT{M,N,L,P,C},x::D) = VCAT(([Jacobian(a,x) for a in V.A]...), V.idxs,  V.mid, M) 
#Jacobian of Compose 
function Jacobian{X<:Union{AbstractArray,NTuple}}(L::Compose, x::X)  
	Compose(Jacobian.(L.A,(x,L.mid...)),L.mid)
end
#Jacobian of Reshape
Jacobian{N,L}(R::Reshape{N,L},x::AbstractArray) = Reshape(Jacobian(R.A,x),R.dim_out) 
#Jacobian of Sum
Jacobian{M,N,K,C,D}(S::Sum{M,N,K,C,D},x::D) = Sum(([Jacobian(a,x) for a in S.A]...),S.midC,S.midD,M,N)

# Properties

fun_name(L::Jacobian)  = "J("*fun_name(L.A)*")"
size(L::Jacobian) = size(L.A,1), size(L.A,2)

domainType(L::Jacobian) = domainType(L.A)
codomainType(L::Jacobian) = codomainType(L.A)
