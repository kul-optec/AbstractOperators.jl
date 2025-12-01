export Jacobian

"""
	Jacobian(A::AbstractOperator,x)

Shorthand constructor:

	jacobian(A::AbstractOperator,x)

Returns the jacobian of `A` evaluated at `x` (which in the case of a `LinearOperator` is `A` itself).

```jldoctest
julia> Jacobian(DiagOp(rand(ComplexF64, 10)),randn(10))
╲  ℂ^10 -> ℂ^10

julia> Jacobian(Sigmoid((10,)),randn(10))
J(σ)  ℝ^10 -> ℝ^10
	
```
"""
struct Jacobian{T<:NonLinearOperator,X<:AbstractArray} <: LinearOperator
	A::T
	x::X
end

#Jacobian of LinearOperator
Jacobian(L::LinearOperator, ::AbstractArray) = L
#Jacobian of Scale
function Jacobian(S::Scale, x::AbstractArray)
	return Scale(S.coeff, Jacobian(S.A, x))
end
##Jacobian of DCAT
function Jacobian(H::DCAT, b)
	x = b.x
	A = ()
	c = 0
	for (k, idx) in enumerate(H.idxD)
		if length(idx) == 1
			A = (A..., jacobian(H.A[k], x[idx]))
		else
			xx = Tuple(x[i] for i in idx)
			A = (A..., jacobian(H.A[k], xx))
		end
	end
	return DCAT(A, H.idxD, H.idxC)
end
#Jacobian of HCAT
function Jacobian(H::HCAT, b::ArrayPartition)
	return Jacobian(H, b.x)
end
function Jacobian(H::HCAT, x::Tuple)
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
function Jacobian(V::VCAT, x)
	JJ = ([Jacobian(a, x) for a in V.A]...,)
	return VCAT(JJ, V.idxs, V.buf)
end
#Jacobian of Compose
function Jacobian(L::Compose, x::AbstractArray)
	x_vec = AbstractArray[x]
	for A in L.A[1:end-1]
		push!(x_vec, A * x_vec[end])
	end
	return Compose(Jacobian.(L.A, tuple(x_vec...)), L.buf)
end

function Jacobian(L::Compose, x::X) where {N,X<:NTuple{N,AbstractArray}}
	x_vec = AbstractArray[x]
	for A in L.A[1:end-1]
		push!(x_vec, A * x_vec[end])
	end
	return Compose(Jacobian.(L.A, tuple(x_vec...)), L.buf)
end
#Jacobian of Reshape
function Jacobian(R::Reshape, x::AbstractArray)
	return Reshape(Jacobian(R.A, x), R.dim_out)
end
#Jacobian of Sum
function Jacobian(S::Sum{K,C,D}, x::D) where {K,C,D}
	return Sum(([Jacobian(a, x) for a in S.A]...,), S.bufC, S.bufD)
end
#Jacobian of Transpose
Jacobian(T::Transpose{<:AbstractOperator}, ::AbstractArray) = T
#Jacobian of BroadCast
Jacobian(L::NoOperatorBroadCast, ::AbstractArray) = L
function Jacobian(B::OperatorBroadCast{T,N,M,false}, x::AbstractArray) where {T,N,M}
	return OperatorBroadCast(Jacobian(B.A, x), B.dim_out, threaded=false)
end
function Jacobian(B::OperatorBroadCast{T,N,M,true}, x::AbstractArray) where {T,N,M}
	return OperatorBroadCast(Jacobian(B.A[1], x), B.dim_out, threaded=true)
end
#Jacobian of AffineAdd
Jacobian(B::AffineAdd, x) = Jacobian(B.A, x)

# Properties
Base.:(==)(L1::Jacobian{L}, L2::Jacobian{L}) where {L} = L1.A == L2.A && L1.x == L2.x
fun_name(L::Jacobian) = "J(" * fun_name(L.A) * ")"
size(L::Jacobian) = size(L.A, 1), size(L.A, 2)

domain_type(L::Jacobian) = domain_type(L.A)
codomain_type(L::Jacobian) = codomain_type(L.A)
domain_storage_type(L::Jacobian) = domain_storage_type(L.A)
codomain_storage_type(L::Jacobian) = codomain_storage_type(L.A)
