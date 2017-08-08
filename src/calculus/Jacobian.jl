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
Jacobian{M,N,C,D}(L::HCAT{M,N,C,D},x::D) = HCAT(Jacobian.(L.A,x), L.mid, M) 
#Jacobian of VCAT
Jacobian{M,N,C,D}(L::VCAT{M,N,C,D},x::D) = VCAT(([Jacobian(a,x) for a in L.A]...), L.mid, M) 
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
