export Jacobian


"""
`Jacobian(A::AbstractOperator,x)`

Shorthand constructor: 

`jacobian(A::AbstractOperator,x)`

Returns the jacobian of `A` evaluated at `x` (which in the case of a `LinearOperator` is `A` itself).

```julia
julia> Jacobian(DFT(10),randn(10))
ℱ  ℝ^10 -> ^ℂ10

julia> Jacobian(Sigmoid((10,)),randn(10))
J(σ)  ℝ^10 -> ℝ^10

```

"""
immutable Jacobian{T <: NonLinearOperator, X<:Union{AbstractArray,NTuple}} <: LinearOperator
	A::T
	x::X
end

#Jacobian of LinearOperator 
Jacobian{T<:LinearOperator,X<:Union{AbstractArray,NTuple}}(L::T,x::X) = L
#Jacobian of Scale
Jacobian{T,L,T2<:Scale{T,L}}(S::T2,x::AbstractArray) = Scale(S.coeff,Jacobian(S.A,x)) 
##Jacobian of DCAT
function Jacobian{N,L,P1,P2}(H::DCAT{N,L,P1,P2},x)  
	A = ()
	c = 0
	for i = 1:N
		A = length(H.idxD[i]) == 1 ?
		(A...,jacobian(H.A[i],x[c+1])) :
		(A...,jacobian(H.A[i],x[c+1:c+length(H.idxD[i])])) 
		c += length(H.idxD[i])
	end
	DCAT(A,H.idxD,H.idxC)
end
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
	HCAT(A,H.idxs,H.buf,M)
end
#Jacobian of VCAT
Jacobian{M,N,L,P,C,D}(V::VCAT{M,N,L,P,C},x::D) = VCAT(([Jacobian(a,x) for a in V.A]...), V.idxs,  V.buf, M) 
#Jacobian of Compose 
function Jacobian{X<:AbstractArray}(L::Compose, x::X)  
	Compose(Jacobian.(L.A,(x,L.buf...)),L.buf)
end

function Jacobian{N,X<:NTuple{N,AbstractArray}}(L::Compose, x::X)  
	Compose(Jacobian.(L.A,(x,L.buf...)),L.buf)
end
#Jacobian of Reshape
Jacobian{N,L}(R::Reshape{N,L},x::AbstractArray) = Reshape(Jacobian(R.A,x),R.dim_out) 
#Jacobian of Sum
Jacobian{M,N,K,C,D}(S::Sum{M,N,K,C,D},x::D) = Sum(([Jacobian(a,x) for a in S.A]...),S.bufC,S.bufD,M,N)

# Properties

fun_name(L::Jacobian)  = "J("*fun_name(L.A)*")"
size(L::Jacobian) = size(L.A,1), size(L.A,2)

domainType(L::Jacobian) = domainType(L.A)
codomainType(L::Jacobian) = codomainType(L.A)
