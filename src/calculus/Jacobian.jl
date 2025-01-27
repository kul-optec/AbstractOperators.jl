export Jacobian

"""
	Jacobian(A::AbstractOperator,x)

Shorthand constructor:

	jacobian(A::AbstractOperator,x)

Returns the jacobian of `A` evaluated at `x` (which in the case of a `LinearOperator` is `A` itself).

```julia
julia> Jacobian(DFT(10),randn(10))
ℱ  ℝ^10 -> ^ℂ10

julia> Jacobian(Sigmoid((10,)),randn(10))
J(σ)  ℝ^10 -> ℝ^10
	
```
"""
struct Jacobian{T<:NonLinearOperator,X<:AbstractArray} <: LinearOperator
	A::T
	x::X
end

#Jacobian of LinearOperator
Jacobian(L::T, x::X) where {T<:LinearOperator,X<:AbstractArray} = L
#Jacobian of Scale
function Jacobian(S::T2, x::AbstractArray) where {T,L,T2<:Scale{T,L}}
	return Scale(S.coeff, Jacobian(S.A, x))
end
##Jacobian of DCAT
function Jacobian(H::DCAT{N,L,P1,P2}, b) where {N,L,P1,P2}
	x = b.x
	A = ()
	c = 0
	for (k, idx) in enumerate(H.idxD)
		if length(idx) == 1
			A = (A..., jacobian(H.A[k], x[idx]))
		else
			xx = ([x[i] for i in idx]...,)
			A = (A..., jacobian(H.A[k], xx))
		end
	end
	return DCAT(A, H.idxD, H.idxC)
end
#Jacobian of HCAT
function Jacobian(H::HCAT{N,L,P,C}, b::ArrayPartition) where {N,L,P,C}
	x = b.x
	A = ()
	for (k, idx) in enumerate(H.idxs)
		if length(idx) == 1
			A = (A..., jacobian(H.A[k], x[idx]))
		else
			xx = ArrayPartition([x[i] for i in idx]...)
			A = (A..., jacobian(H.A[k], xx))
		end
	end
	return HCAT(A, H.idxs, H.buf)
end
#Jacobian of VCAT
function Jacobian(V::VCAT{N,L,P,C}, x::AbstractArray) where {N,L,P,C}
	JJ = ([Jacobian(a, x) for a in V.A]...,)
	return VCAT(JJ, V.idxs, V.buf)
end
#Jacobian of Compose
function Jacobian(L::Compose, x::X) where {X<:AbstractArray}
	return Compose(Jacobian.(L.A, (x, L.buf...)), L.buf)
end

function Jacobian(L::Compose, x::X) where {N,X<:NTuple{N,AbstractArray}}
	return Compose(Jacobian.(L.A, (x, L.buf...)), L.buf)
end
#Jacobian of Reshape
function Jacobian(R::Reshape{N,L}, x::AbstractArray) where {N,L}
	return Reshape(Jacobian(R.A, x), R.dim_out)
end
#Jacobian of Sum
function Jacobian(S::Sum{K,C,D}, x::D) where {K,C,D}
	return Sum(([Jacobian(a, x) for a in S.A]...,), S.bufC, S.bufD)
end
#Jacobian of Transpose
Jacobian(T::Transpose{A}, x::AbstractArray) where {A<:AbstractOperator} = T
#Jacobian of BroadCast
function Jacobian(B::A, x::AbstractArray) where {A<:BroadCast}
	return BroadCast(Jacobian(B.A, x), B.dim_out)
end
#Jacobian of AffineAdd
Jacobian(B::A, x) where {A<:AffineAdd} = Jacobian(B.A, x)

# Properties

fun_name(L::Jacobian) = "J(" * fun_name(L.A) * ")"
size(L::Jacobian) = size(L.A, 1), size(L.A, 2)

domainType(L::Jacobian) = domainType(L.A)
codomainType(L::Jacobian) = codomainType(L.A)
