import Base: adjoint, *, +, -, getindex, hcat, vcat, reshape
export jacobian

###### ' ######
adjoint(L::T) where {T <: AbstractOperator} = AdjointOperator(L)

######+,-######
(+)(L::T) where {T <: AbstractOperator} = L
(-)(L::T) where {T <: AbstractOperator} = Scale(-1.0, L)
(+)(L1::AbstractOperator, L2::AbstractOperator) = Sum(L1,  L2 )
(-)(L1::AbstractOperator, L2::AbstractOperator) = Sum(L1, -L2 )

###### * ######
function (*)(L::AbstractOperator, b::AbstractArray)
  C = codomainType(L)
  if typeof(C) <: Tuple
    y = ArrayPartition(zeros.(codomainType(L), size(L, 1))...)
  else
	  y = zeros(codomainType(L), size(L, 1))
  end
	mul!(y, L, b)
	return y
end

#(*)(L::AbstractOperator, b::Tuple) = (*)(L, ArrayPartition(b...))

*(coeff::T, L::AbstractOperator) where {T<:Number} = Scale(coeff,L)
*(L1::AbstractOperator, L2::AbstractOperator) = Compose(L1,L2)

# getindex
function getindex(A::AbstractOperator,idx...)
	if ndoms(A,2) == 1
		Gout = GetIndex(codomainType(A),size(A,1),idx)
		return Gout*A
	elseif length(idx) == 1  && ndoms(A,2) == length(idx[1])
		return permute(A,idx[1])
	else
		error("cannot split operator of type $(typeof(A))")
	end
end

#get index of HCAT returns HCAT (or Operator)
function getindex(H::HCAT, idx::Union{AbstractArray,Int})

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
function getindex(H::VCAT, idx::Union{AbstractArray,Int})

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
getindex(A::S,idx...) where {T, L, S <:Scale{T,L}} = Scale(A.coeff,A.coeff_conj,getindex(A.A,idx...))

hcat(L::Vararg{AbstractOperator}) = HCAT(L...)
vcat(L::Vararg{AbstractOperator}) = VCAT(L...)

###### reshape ######
reshape(L::A, idx::NTuple{N,Int}) where {N,A<:AbstractOperator} = Reshape(L,idx)
reshape(L::A, idx::Vararg{Int}) where {A<:AbstractOperator} = Reshape(L,idx)

###### jacobian ######
jacobian = Jacobian
