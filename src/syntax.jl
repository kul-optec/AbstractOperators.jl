import Base: blkdiag, transpose, *, +, -, getindex, hcat, vcat, reshape
export jacobian

###### blkdiag ######
blkdiag(L::Vararg{AbstractOperator}) = DCAT(L...)

###### ' ######
transpose{T <: AbstractOperator}(L::T) = Transpose(L)

######+,-######
(+){T <: AbstractOperator}(L::T) = L
(-){T <: AbstractOperator}(L::T) = Scale(-1.0, L)
(+)(L1::AbstractOperator, L2::AbstractOperator) = Sum(L1,  L2 )
(-)(L1::AbstractOperator, L2::AbstractOperator) = Sum(L1, -L2 )

###### * ######
function (*){T <: BlockArray}(L::AbstractOperator, b::T)
	y = blockzeros(codomainType(L), size(L, 1))
	A_mul_B!(y, L, b)
	return y
end

*{T<:Number}(coeff::T, L::AbstractOperator) = Scale(coeff,L)
*(L1::AbstractOperator, L2::AbstractOperator) = Compose(L1,L2)

# redefine .*
Base.broadcast(::typeof(*), d::AbstractArray, L::AbstractOperator) = DiagOp(codomainType(L), size(d), d)*L
Base.broadcast(::typeof(*), d::AbstractArray, L::Scale) = DiagOp(L.coeff*d)*L.A

# getindex
function getindex(A::AbstractOperator,idx...)
	if ndoms(A,2) == 1
		Gout = GetIndex(codomainType(A),size(A,1),idx)
		return Gout*A
	elseif length(idx) == 1  && ndoms(A,2) == length(idx[1])
		return permute(A,idx[1])
	else
		error("cannot split operator of type $(typeof(H.A[i]))")
	end
end

#get index of HCAT returns HCAT (or Operator)
function getindex{M,N,L,P,C,A<:HCAT{M,N,L,P,C}}(H::A, idx::Union{AbstractArray,Int})

	unfolded = vcat([[i... ] for i in H.idxs]...)
	if length(idx) == length(unfolded)
		return permute(H,idx)
	else
		new_H = ()
		for i in idx
			for ii in eachindex(H.idxs)
				if i in H.idxs[ii]
					if typeof(H.idxs[ii]) <: Int
						new_H = (new_H...,H.A[ii])
					else
					error("cannot split operator")
					end
				end
			end
		end
		return HCAT(new_H,H.buf)
	end
end


#get index of HCAT returns HCAT (or Operator)
function getindex{M,N,L,P,C,A<:VCAT{M,N,L,P,C}}(H::A, idx::Union{AbstractArray,Int})

	unfolded = vcat([[i... ] for i in H.idxs]...)
	if length(idx) == length(unfolded)
		return permute(H,idx)
	else
		new_H = ()
		for i in idx
			for ii in eachindex(H.idxs)
				if i in H.idxs[ii]
					if typeof(H.idxs[ii]) <: Int
						new_H = (new_H...,H.A[ii])
					else
					error("cannot split operator")
					end
				end
			end
		end
		return VCAT(new_H,H.buf)
	end
end

getindex(H::A, idx::Union{AbstractArray,Int}) where {L <: HCAT, D, S, A<: AffineAdd{L,D,S}} = 
AffineAdd(getindex(H.A, idx), H.d, S) 

# get index of scale
getindex{T, L, S <:Scale{T,L}}(A::S,idx...) = Scale(A.coeff,A.coeff_conj,getindex(A.A,idx...))

hcat(L::Vararg{AbstractOperator}) = HCAT(L...)
vcat(L::Vararg{AbstractOperator}) = VCAT(L...)

###### reshape ######
reshape{N,A<:AbstractOperator}(L::A, idx::NTuple{N,Int}) = Reshape(L,idx)
reshape{A<:AbstractOperator}(L::A, idx::Vararg{Int}) = Reshape(L,idx)

###### jacobian ######
jacobian = Jacobian
