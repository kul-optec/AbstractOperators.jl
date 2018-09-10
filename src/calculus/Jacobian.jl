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
struct Jacobian{T <: NonLinearOperator, X<:Union{AbstractArray,Tuple}} <: LinearOperator
	A::T
	x::X
end

#Jacobian of LinearOperator 
Jacobian(L::T,x::X) where {T<:LinearOperator,X<:Union{AbstractArray,NTuple}} = L
#Jacobian of Scale
Jacobian(S::T2,x::AbstractArray) where {T,L,T2<:Scale{T,L}} = Scale(S.coeff,Jacobian(S.A,x)) 
##Jacobian of DCAT
function Jacobian(H::DCAT{N,L,P1,P2},x) where {N,L,P1,P2} 
	A = ()
	c = 0
    for (k, idx) in enumerate(H.idxD)
		if length(idx) == 1 
            A = (A...,jacobian(H.A[k],x[idx])) 
        else
            xx = ([x[i] for i in idx]...,)
            A = (A...,jacobian(H.A[k],xx)) 
        end
	end
	DCAT(A,H.idxD,H.idxC)
end
#Jacobian of HCAT
function Jacobian(H::HCAT{M,N,L,P,C},x::D) where {M,N,L,P,C,D} 
	A = ()
    for (k, idx) in enumerate(H.idxs)
		if length(idx) == 1 
            A = (A...,jacobian(H.A[k],x[idx])) 
        else
            xx = ([x[i] for i in idx]...,)
            A = (A...,jacobian(H.A[k],xx)) 
        end
	end
	HCAT(A,H.idxs,H.buf,M)
end
#Jacobian of VCAT
function Jacobian(V::VCAT{M,N,L,P,C},x::D) where {M,N,L,P,C,D}
    JJ = ([Jacobian(a,x) for a in V.A]...,)
    VCAT{M,N,typeof(JJ),P,C}(JJ, V.idxs, V.buf)
end
#Jacobian of Compose 
function Jacobian(L::Compose, x::X) where {X<:AbstractArray} 
	Compose(Jacobian.(L.A,(x,L.buf...)),L.buf)
end

function Jacobian(L::Compose, x::X) where {N,X<:NTuple{N,AbstractArray}} 
	Compose(Jacobian.(L.A,(x,L.buf...)),L.buf)
end
#Jacobian of Reshape
Jacobian(R::Reshape{N,L},x::AbstractArray) where {N,L} = Reshape(Jacobian(R.A,x),R.dim_out) 
#Jacobian of Sum
Jacobian(S::Sum{M,N,K,C,D},x::D) where {M,N,K,C,D} = 
Sum(([Jacobian(a,x) for a in S.A]...,),S.bufC,S.bufD,M,N)
#Jacobian of Transpose
Jacobian(T::Transpose{A}, x::AbstractArray) where {A <: AbstractOperator} = T 
#Jacobian of BroadCast
Jacobian(B::A, x::AbstractArray) where {A <: BroadCast} = BroadCast(Jacobian(B.A,x),B.dim_out) 
#Jacobian of AffineAdd
Jacobian(B::A, x) where {A <: AffineAdd} = Jacobian(B.A,x)

# Properties

fun_name(L::Jacobian)  = "J("*fun_name(L.A)*")"
size(L::Jacobian) = size(L.A,1), size(L.A,2)

domainType(L::Jacobian) = domainType(L.A)
codomainType(L::Jacobian) = codomainType(L.A)
