import Base: blkdiag, transpose, *, +, -, getindex, hcat, vcat, reshape

###### blkdiag ######
blkdiag(L::Vararg{LinearOperator}) = DCAT(L...)

###### ' ######
transpose{T <: LinearOperator}(L::T) = Transpose(L)

######+,-######
(+){T <: LinearOperator}(L::T) = L
(-){T <: LinearOperator}(L::T) = Scale(-1.0, L)
(+)(L1::LinearOperator, L2::LinearOperator) = Sum(L1,  L2 )
(-)(L1::LinearOperator, L2::LinearOperator) = Sum(L1, -L2 )

###### * ######
function (*){T <: Union{AbstractArray, Tuple}}(L::LinearOperator, b::T)
	y = deepzeros(codomainType(L), size(L, 1))
	A_mul_B!(y, L, b)
	return y
end

*{T<:Number}(coeff::T, L::LinearOperator) = Scale(coeff,L)

*{S<:Scale}(L1::S, L2::LinearOperator) = L1.coeff*(L1.A*L2)
*{S<:Scale}(L1::LinearOperator, L2::S) = L2.coeff*(L1*L2.A)
*{S1<:Scale,S2<:Scale}(L1::S1, L2::S2) = (L1.coeff*L2.coeff)*(L1.A*L2.A)

*(L1::LinearOperator, L2::LinearOperator) = Compose(L1,L2)

*{E<:Eye}(L1::E, L2::LinearOperator) = L2
*{E<:Eye}(L1::LinearOperator, L2::E) = L1
*{E1<:Eye,E2<:Eye}(L1::E1, L2::E2) = L1

# redefine .*
Base.broadcast(::typeof(*), d::AbstractArray, L::LinearOperator) = DiagOp(codomainType(L), d)*L
Base.broadcast(::typeof(*), d::AbstractArray, L::Scale)          = DiagOp(L.coeff*d)*L.A

# getindex
# slice only output 
function getindex(A::LinearOperator,idx...) 
	Gout = GetIndex(codomainType(A),size(A,1),idx)
	return Gout*A
end

# slice output and input e.g. G[idx_out...][idx_in...] 
function getindex{L<:Compose}(A::L,idx...) 
	if typeof(A.A[end]) <: GetIndex
		Gin = GetIndex(codomainType(A.A[1]),size(A.A[1],2),idx)
		return A*Gin'
	else
		Gout = GetIndex(codomainType(A),size(A,1),idx)
		return Gout*A
	end
end

hcat(L::Vararg{LinearOperator}) = HCAT(L...)
vcat(L::Vararg{LinearOperator}) = VCAT(L...)

###### reshape ######
reshape{N,A<:LinearOperator}(L::A, idx::NTuple{N,Int}) = Reshape(L,idx)
reshape{A<:LinearOperator}(L::A, idx::Vararg{Int}) = Reshape(L,idx)